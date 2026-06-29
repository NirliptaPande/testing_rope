"""
exp2_rope_frequency.py
======================
EXPERIMENT 2 -- RoPE frequency decomposition.

Untwisting RoPE shows the attention logit q.k is an additive sum over RoPE
channel pairs, and each pair carries a distinct angular frequency omega:
  * high omega  -> steep similarity drop with distance -> LOCAL
  * low  omega  -> position-insensitive               -> GLOBAL / semantic

Because the RoPE channels ARE the frequency basis, we don't need an FFT to
decompose the logit: we just restrict q,k to the high-omega channels vs the
low-omega channels and recompute the (image-query -> image-key) attention map.
We then verify locality two ways:
  (1) spatial spread = mean distance of attention mass from the query patch
  (2) a 2D radial power spectrum of the attention map (the literal 'FFT'
      view: local maps carry energy at higher spatial frequencies).

Outputs:
  exp2_layerL<idx>_maps.png   full / high-band / low-band maps for a query patch
  exp2_spread_vs_layer.png    mean spread (hi vs lo band) across analyzed layers
  exp2_metrics.json
"""

import os
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


def img_qk(layer, meta):
    """Image-token q,k as (H, img_len, D)."""
    t, n = meta["txt_len"], meta["img_len"]
    q = layer["q"][0][:, t : t + n, :]
    k = layer["k"][0][:, t : t + n, :]
    return q, k


def attn_row(q, k, query_idx, mask, meta):
    """Attention distribution of one image query over all image keys (grid)."""
    logits = C.attn_logits(q, k, channel_mask=mask)     # (img,img)
    p = C.softmax_rows(logits)[query_idx].numpy()       # (img,)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/run0")
    ap.add_argument("--layers", default=None,
                    help="comma-sep layer indices; default = spread across depth")
    ap.add_argument("--detail-layer", type=int, default=None,
                    help="layer to render full/hi/lo maps for (default: a mid double block)")
    ap.add_argument("--hi-frac", type=float, default=0.5,
                    help="fraction of channels (by omega) in the high band")
    args = ap.parse_args()

    layers, meta = C.load_run(args.run_dir)
    hi, lo, omega = C.frequency_band_masks(meta, hi_frac=args.hi_frac)
    H, W = meta["h_patches"], meta["w_patches"]

    if args.layers:
        layer_ids = [int(x) for x in args.layers.split(",")]
    else:
        nd, nl = meta["n_double"], meta["n_layers"]
        layer_ids = sorted(set([0, nd // 2, nd - 1, nd, nd + (nl - nd) // 2, nl - 1]))
    detail = args.detail_layer if args.detail_layer is not None else meta["n_double"] // 2

    # query patches: a 3x3 grid of points, plus the center for the detail figure
    qpts = [(int(round(r * (H - 1))), int(round(c * (W - 1))))
            for r in (0.25, 0.5, 0.75) for c in (0.25, 0.5, 0.75)]
    center = (H // 2, W // 2)

    # ---- across-layer spread trend (averaged over query points) ----
    metrics = {"layers": [], "spread_full": [], "spread_hi": [], "spread_lo": [],
               "kind": []}
    for li in layer_ids:
        q, k = img_qk(layers[li], meta)
        sf = sh = sl = 0.0
        for (qr, qc) in qpts:
            qi = C.rc_patch(qr, qc, meta)
            sf += C.spatial_spread(attn_row(q, k, qi, None, meta), (qr, qc), meta)
            sh += C.spatial_spread(attn_row(q, k, qi, hi, meta), (qr, qc), meta)
            sl += C.spatial_spread(attn_row(q, k, qi, lo, meta), (qr, qc), meta)
        n = len(qpts)
        metrics["layers"].append(int(li))
        metrics["kind"].append(C.layer_kind(li, meta))
        metrics["spread_full"].append(sf / n)
        metrics["spread_hi"].append(sh / n)
        metrics["spread_lo"].append(sl / n)
        print(f"[exp2] L{li:>2} ({C.layer_kind(li, meta):>6}): "
              f"spread full={sf/n:5.2f}  high={sh/n:5.2f}  low={sl/n:5.2f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = metrics["layers"]
    ax.plot(x, metrics["spread_hi"], "o-", label="high-freq band (expect local/small)")
    ax.plot(x, metrics["spread_lo"], "s-", label="low-freq band (expect global/large)")
    ax.plot(x, metrics["spread_full"], "x--", color="gray", label="all channels")
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean attention spread (patch units)")
    ax.set_title("Exp2: RoPE band locality vs depth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_trend = os.path.join(args.run_dir, "exp2_spread_vs_layer.png")
    fig.savefig(out_trend, dpi=140)
    plt.close(fig)

    # ---- detail maps for one layer / center query ----
    q, k = img_qk(layers[detail], meta)
    qi = C.rc_patch(*center, meta)
    maps = {
        "full": attn_row(q, k, qi, None, meta),
        "high": attn_row(q, k, qi, hi, meta),
        "low": attn_row(q, k, qi, lo, meta),
    }
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for j, (name, p) in enumerate(maps.items()):
        grid = C.to_grid(p, meta)
        ax = axes[0, j]
        im = ax.imshow(grid, cmap="magma")
        ax.plot(center[1], center[0], "c+", ms=12, mew=2)
        sp = C.spatial_spread(p, center, meta)
        ax.set_title(f"{name} band\nspread={sp:.2f}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        # radial power spectrum (FFT view)
        rad, power = C.radial_power_spectrum(grid)
        axes[1, j].semilogy(rad, power + 1e-12)
        axes[1, j].set_title(f"{name}: radial FFT power")
        axes[1, j].set_xlabel("spatial frequency (radius)")
    fig.suptitle(f"Exp2 detail: layer {detail} ({C.layer_kind(detail, meta)}), "
                 f"center query patch", fontsize=11)
    fig.tight_layout()
    out_maps = os.path.join(args.run_dir, f"exp2_layerL{detail}_maps.png")
    fig.savefig(out_maps, dpi=140)
    plt.close(fig)

    metrics["detail_layer"] = int(detail)
    metrics["hi_frac"] = args.hi_frac
    metrics["n_channels_hi"] = int(hi.sum())
    metrics["n_channels_lo"] = int(lo.sum())
    with open(os.path.join(args.run_dir, "exp2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[exp2] wrote {out_trend} and {out_maps}")
    print(f"[exp2] high band = {int(hi.sum())} channels, low band = {int(lo.sum())} "
          f"(of {omega.numel()})")
    print("[exp2] CLAIM HOLDS if high-band spread < low-band spread (high freq = local).")


if __name__ == "__main__":
    main()