"""
exp3_entanglement.py
====================
EXPERIMENT 3 -- Entanglement severity map.

The joint attention is a (txt+img) x (txt+img) matrix. For the IMAGE queries we
split where their attention mass goes:
    image -> text  (IT block)  = cross-modal "contamination" of spatial signal
    image -> image (II block)  = the self/spatial structure we care about
and we characterize the II block's structure:
    entropy   (normalized 0..1): 1 = diffuse/global, 0 = peaky/local
    spread    (patch units)    : mean distance of II mass from each query patch

Per layer we report:
    cross_fraction = mean over image queries of attention mass on text keys
    ii_entropy     = mean normalized entropy of image->image attention
    ii_spread      = mean spatial spread of image->image attention

"Structure survives best" where cross_fraction is LOW and ii_entropy/ii_spread
are LOW (localized, uncontaminated). That ranking is printed and saved.

Outputs:
  exp3_entanglement.png   line plots (double vs single shaded) + severity heatmap
  exp3_metrics.json
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


def layer_metrics(layer, meta, dist_img, n_real):
    """Average per-head joint-attention metrics for one layer.

    n_real: number of REAL text tokens (prompt + EOS, excluding T5 padding).
    Real tokens are the first n_real positions; padding follows. We count
    contamination only against real tokens so the figure isn't inflated by
    attention to padding."""
    t, n = meta["txt_len"], meta["img_len"]
    q = layer["q"][0].float()                  # (H, S, D)
    k = layer["k"][0].float()
    Hd = q.shape[0]
    d = q.shape[-1]
    scale = d ** -0.5

    cross_f = ent = spread = 0.0
    log_n = np.log(n)
    for h in range(Hd):
        logits = (q[h] @ k[h].transpose(0, 1)) * scale     # (S,S)
        p = torch.softmax(logits, dim=-1)                  # (S,S)
        p_img = p[t : t + n]                               # image queries (n, S)
        # cross-modal mass: image queries attending to REAL text keys only
        cross_f += p_img[:, :n_real].sum(-1).mean().item()
        # image->image block, renormalized over image keys
        ii = p_img[:, t : t + n]
        ii = ii / (ii.sum(-1, keepdim=True) + 1e-12)       # (n, n)
        ent += (-(ii * (ii + 1e-12).log()).sum(-1) / log_n).mean().item()
        spread += (ii.numpy() * dist_img).sum(-1).mean()
    return cross_f / Hd, ent / Hd, spread / Hd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/run0")
    args = ap.parse_args()

    layers, meta = C.load_run(args.run_dir)
    H, W, n = meta["h_patches"], meta["w_patches"], meta["img_len"]

    # precompute patch-to-patch distance matrix (img_len x img_len)
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([rr.reshape(-1), cc.reshape(-1)], 1).astype(np.float64)  # (n,2)
    dist_img = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))  # (n,n)

    # real text tokens = non-padding (T5 pads with "<pad>" after the prompt+EOS)
    toks = meta.get("t5_tokens", [])[: meta["txt_len"]]
    n_real = sum(1 for tk in toks if tk != "<pad>") if toks else meta["txt_len"]
    print(f"[exp3] counting contamination against {n_real} real text tokens "
          f"(of {meta['txt_len']} total; rest is padding)")

    cross, ent, spread, kinds = [], [], [], []
    for li, lyr in enumerate(layers):
        cf, en, sp = layer_metrics(lyr, meta, dist_img, n_real)
        cross.append(cf)
        ent.append(en)
        spread.append(sp)
        kinds.append(C.layer_kind(li, meta))
        print(f"[exp3] L{li:>2} ({kinds[-1]:>6}): cross={cf:.3f}  "
              f"ii_entropy={en:.3f}  ii_spread={sp:.2f}")

    cross = np.array(cross); ent = np.array(ent); spread = np.array(spread)
    nd = meta["n_double"]
    L = len(layers)

    # composite severity: high = more entangled / less usable structure
    sev = C.normalize01(cross) + C.normalize01(ent) + C.normalize01(spread)
    best = np.argsort(sev)[:5]
    print("[exp3] layers where structure survives best (lowest severity): "
          + ", ".join(f"L{int(i)}({kinds[int(i)][0]})" for i in best))

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(L)

    def shade(ax):
        ax.axvspan(-0.5, nd - 0.5, color="tab:blue", alpha=0.07, label="double-stream")
        ax.axvspan(nd - 0.5, L - 0.5, color="tab:orange", alpha=0.07, label="single-stream")

    axes[0, 0].plot(x, cross, "o-", color="crimson")
    shade(axes[0, 0]); axes[0, 0].set_title("image->text mass (contamination)")
    axes[0, 0].set_xlabel("layer"); axes[0, 0].set_ylabel("fraction")

    axes[0, 1].plot(x, ent, "o-", color="teal")
    shade(axes[0, 1]); axes[0, 1].set_title("image->image entropy (1=diffuse)")
    axes[0, 1].set_xlabel("layer"); axes[0, 1].set_ylabel("norm. entropy")

    axes[1, 0].plot(x, spread, "o-", color="purple")
    shade(axes[1, 0]); axes[1, 0].set_title("image->image spatial spread")
    axes[1, 0].set_xlabel("layer"); axes[1, 0].set_ylabel("patch units")
    axes[1, 0].legend(fontsize=7)

    heat = np.stack([C.normalize01(cross), C.normalize01(ent), C.normalize01(spread)])
    im = axes[1, 1].imshow(heat, aspect="auto", cmap="inferno")
    axes[1, 1].set_yticks([0, 1, 2])
    axes[1, 1].set_yticklabels(["cross", "ii_entropy", "ii_spread"])
    axes[1, 1].set_xlabel("layer")
    axes[1, 1].set_title("severity heatmap (norm.)  low=structure survives")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    fig.suptitle("Exp3: entanglement severity across layers", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.run_dir, "exp3_entanglement.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)

    with open(os.path.join(args.run_dir, "exp3_metrics.json"), "w") as f:
        json.dump(
            {
                "cross_fraction": cross.tolist(),
                "ii_entropy": ent.tolist(),
                "ii_spread": spread.tolist(),
                "kinds": kinds,
                "n_double": nd,
                "best_layers": [int(i) for i in best],
            },
            f,
            indent=2,
        )
    print(f"[exp3] wrote {out}")


if __name__ == "__main__":
    main()