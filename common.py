"""
common.py
=========
Shared helpers for the three experiments. Everything here is pure
torch/numpy and runs on CPU -- the heavy GPU work happened in capture.py.
"""

import os
import json

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_run(run_dir):
    blob = torch.load(os.path.join(run_dir, "capture_store.pt"), map_location="cpu")
    return blob["layers"], blob["meta"]


def layer_kind(layer_idx, meta):
    """'double' or 'single' for a given captured layer index."""
    return "double" if layer_idx < meta["n_double"] else "single"


# --------------------------------------------------------------------------- #
# Tensor reshaping into the attention "output space"                           #
# --------------------------------------------------------------------------- #
def merge_heads(x):
    """(B, H, S, D) -> (S, H*D). Concatenating heads is the output space that
    feeds attn.to_out -- this is what ConceptAttention projects in."""
    x = x.float()
    b, h, s, d = x.shape
    return x[0].permute(1, 0, 2).reshape(s, h * d)


def split_txt_img(vec_seq, meta):
    """(S, C) -> (txt, C), (img, C)."""
    t = meta["txt_len"]
    img_len = meta["img_len"]
    return vec_seq[:t], vec_seq[t : t + img_len]


def to_grid(patch_vec, meta):
    """(img_len,) -> (h_patches, w_patches), row-major (FLUX packing order)."""
    return np.asarray(patch_vec).reshape(meta["h_patches"], meta["w_patches"])


def patch_rc(idx, meta):
    """image-patch index -> (row, col) on the patch grid."""
    w = meta["w_patches"]
    return idx // w, idx % w


def rc_patch(r, c, meta):
    return r * meta["w_patches"] + c


# --------------------------------------------------------------------------- #
# Concept word  ->  T5 token positions                                         #
# --------------------------------------------------------------------------- #
def _clean(tok):
    # T5 SentencePiece marks word starts with U+2581 ('▁')
    return tok.replace("▁", "").strip().lower()


def concept_positions(concept, meta):
    """Indices into the text portion whose token matches `concept`.

    A token matches if its cleaned form equals the concept or starts with it
    (so 'cloud'->'cloudy', 'mountain'->'mountains', 'light'->'lights'). Using
    startswith (not substring) avoids false hits like 'tree' in 'street' or the
    article 'a' in 'apple'.
    """
    concept = concept.lower().strip()
    toks = meta["t5_tokens"][: meta["txt_len"]]
    pos = []
    for i, t in enumerate(toks):
        c = _clean(t)
        if c and (c == concept or c.startswith(concept)):
            pos.append(i)
    return pos, [toks[i] for i in pos]


# --------------------------------------------------------------------------- #
# FLUX RoPE per-channel frequency table                                        #
# --------------------------------------------------------------------------- #
def flux_channel_frequencies(axes_dims=(16, 56, 56), theta=10000.0):
    """
    Angular frequency (omega) for each of the head_dim channels, matching
    diffusers' get_1d_rotary_pos_embed(repeat_interleave_real=True):

        per axis of dim d:  omega_j = theta^(-(2j)/d), j=0..d/2-1
        then each omega is repeat_interleaved x2  (channels 2j, 2j+1 share it)

    Channels are concatenated across the 3 axes (axis0=text/seq, axis1=H, axis2=W).
    Large omega = high frequency = position-sensitive (local).
    Small omega = low  frequency = position-insensitive (global/semantic).
    Returns: (head_dim,) float tensor.
    """
    freqs = []
    for d in axes_dims:
        j = torch.arange(0, d, 2).float()          # [0,2,...,d-2], length d/2
        omega = theta ** (-(j / d))                # (d/2,)
        omega = omega.repeat_interleave(2)         # (d,)
        freqs.append(omega)
    return torch.cat(freqs)                         # (sum(axes_dims),)


def frequency_band_masks(meta, hi_frac=0.5, spatial_only=False):
    """
    Boolean channel masks (head_dim,) for high- and low-frequency bands.
    Splits channels by their RoPE omega: top `hi_frac` by omega = high band.

    spatial_only=True: drop the text/temporal RoPE axis (axes[0], the first
    `axes[0]` channels) so the bands are built ONLY from the H and W image
    axes. Use this for a purely-spatial locality test. Excluded channels are
    False in both masks (never used in the band logits).
    """
    axes = tuple(meta.get("axes_dims_rope", [16, 56, 56]))
    theta = meta.get("rope_theta", 10000.0)
    omega = flux_channel_frequencies(axes, theta)          # (head_dim,)
    n = omega.numel()

    cand = torch.arange(n)
    if spatial_only:
        cand = cand[axes[0]:]                              # keep H,W axes only
    order = cand[torch.argsort(omega[cand], descending=True)]   # high -> low
    n_hi = int(round(hi_frac * cand.numel()))
    hi = torch.zeros(n, dtype=torch.bool)
    lo = torch.zeros(n, dtype=torch.bool)
    hi[order[:n_hi]] = True
    lo[order[n_hi:]] = True
    return hi, lo, omega


# --------------------------------------------------------------------------- #
# Attention math (re-derived from captured q,k)                                #
# --------------------------------------------------------------------------- #
def attn_logits(q, k, channel_mask=None):
    """
    q,k: (H, S, D). Returns mean-over-heads logits (Sq, Sk) using the model's
    scale 1/sqrt(D_used). If channel_mask given, restrict to those channels
    (a RoPE frequency band).
    """
    q = q.float()
    k = k.float()
    if channel_mask is not None:
        q = q[..., channel_mask]
        k = k[..., channel_mask]
    d = q.shape[-1]
    logits = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)  # (H,Sq,Sk)
    return logits.mean(0)                                       # (Sq,Sk)


def softmax_rows(logits):
    return torch.softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
# Spatial-locality metrics for an attention-over-image-patches vector          #
# --------------------------------------------------------------------------- #
def spatial_spread(attn_over_patches, query_rc, meta):
    """
    Mean Euclidean distance (in patch units) of attention mass from the query
    patch. Small = local, large = global. attn_over_patches: (img_len,) >=0.
    """
    a = np.asarray(attn_over_patches, dtype=np.float64)
    a = a / (a.sum() + 1e-12)
    qr, qc = query_rc
    H, W = meta["h_patches"], meta["w_patches"]
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dist = np.sqrt((rr - qr) ** 2 + (cc - qc) ** 2).reshape(-1)
    return float((a * dist).sum())


def radial_power_spectrum(grid):
    """
    2D FFT power of an attention map, averaged into radial bins. Honors the
    'FFT to decompose by frequency' framing: local maps put more energy at
    high spatial frequencies. Returns (radii, power).
    """
    g = np.asarray(grid, dtype=np.float64)
    g = g - g.mean()
    F2 = np.fft.fftshift(np.fft.fft2(g))
    P = np.abs(F2) ** 2
    H, W = g.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = r.max()
    radial = np.array([P[r == i].mean() if np.any(r == i) else 0.0 for i in range(rmax + 1)])
    return np.arange(rmax + 1), radial


def normalize01(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)