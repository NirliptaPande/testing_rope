"""
exp4_intersection.py
=====================
EXPERIMENT 4 -- AV semantics  ∩  RoPE geometry.

Tests "intersection is the signal": build a band-limited attention output
    AV_high = softmax(Q_hi · K_hiᵀ / sqrt(d_hi)) · V
using ONLY the high-frequency RoPE channels for the attention weights (geometry
= local) but the FULL value vectors (semantics). Then run the Exp1 concept
projection on AV_high and compare its localization score to the full AV (Exp1)
and the low-band AV.

If AV_high localizes concepts MORE sharply than full AV, the clean control
signal lives at the intersection of semantic output space and high-frequency
RoPE locality.

Requires a store captured with --save-v (Exp4 recomputes AV from Q,K,V).

Outputs:
  exp4_score_vs_layer.png   per-layer localization: full vs high vs low band
  exp4_concepts.png         band-averaged saliency, full vs high band, per concept
  exp4_metrics.json
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
from exp1_av_localization import saliency_from_output, localization_score, upsample, parse_band


def band_limited_av(layer, mask, meta):
    """Recompute AV using only `mask` channels for Q·K, full V. -> (S, H*D)."""
    q = layer["q"][0].float()
    k = layer["k"][0].float()
    v = layer["v"][0].float()
    qm, km = q[..., mask], k[..., mask]
    d = int(mask.sum())
    logits = torch.matmul(qm, km.transpose(-1, -2)) / (d ** 0.5)  # (H,S,S)
    p = torch.softmax(logits, dim=-1)
    av = torch.matmul(p, v)                                        # (H,S,D)
    H, S, D = av.shape
    return av.permute(1, 0, 2).reshape(S, H * D)                   # (S, H*D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/run0")
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--band", default=None, help="layers to average for the figure (e.g. 19-30)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hi-frac", type=float, default=0.5)
    ap.add_argument("--spatial-only", action="store_true",
                    help="build bands from H/W RoPE axes only")
    args = ap.parse_args()

    layers, meta = C.load_run(args.run_dir)
    if "v" not in layers[0]:
        print("[exp4] ERROR: store has no V. Re-capture with --save-v.")
        return
    concepts = ([c.strip() for c in args.concepts.split(",")]
                if args.concepts else meta["concepts"])
    hi, lo, _ = C.frequency_band_masks(meta, args.hi_frac, spatial_only=args.spatial_only)

    pos_map, used = {}, []
    for c in concepts:
        p, matched = C.concept_positions(c, meta)
        if p:
            pos_map[c] = p; used.append(c)
            print(f"[exp4] '{c}' -> {matched}")
    if not used:
        print("[exp4] no concepts matched."); return

    n = len(layers)
    # per-layer scores for full / high / low readouts (mean over concepts)
    sc_full, sc_hi, sc_lo = np.zeros(n), np.zeros(n), np.zeros(n)
    corr = []  # sanity: recomputed-full AV vs stored AV
    for li, lyr in enumerate(layers):
        O_full = C.merge_heads(lyr["av"])
        O_hi = band_limited_av(lyr, hi, meta)
        O_lo = band_limited_av(lyr, lo, meta)
        for c in used:
            sc_full[li] += localization_score(saliency_from_output(O_full, pos_map[c], meta))
            sc_hi[li] += localization_score(saliency_from_output(O_hi, pos_map[c], meta))
            sc_lo[li] += localization_score(saliency_from_output(O_lo, pos_map[c], meta))
    sc_full /= len(used); sc_hi /= len(used); sc_lo /= len(used)

    # choose band
    band = parse_band(args.band)
    if band is None:
        band = sorted(np.argsort(np.maximum(sc_full, sc_hi))[::-1][: args.topk].tolist())
    band = [b for b in band if 0 <= b < n]
    print(f"[exp4] band = {band}")
    print(f"[exp4] mean score over band  full={sc_full[band].mean():.3f}  "
          f"high={sc_hi[band].mean():.3f}  low={sc_lo[band].mean():.3f}")
    verdict = ("HIGH-BAND sharper -> intersection helps"
               if sc_hi[band].mean() > sc_full[band].mean() else
               "full AV already sharper than high band here")
    print(f"[exp4] {verdict}")

    # ---- per-layer score figure ----
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, sc_full, "o-", ms=3, label="full AV (Exp1)")
    ax.plot(x, sc_hi, "o-", ms=3, label="high-band AV (semantics ∩ geometry)")
    ax.plot(x, sc_lo, "o-", ms=3, color="gray", alpha=0.6, label="low-band AV")
    nd = meta["n_double"]
    ax.axvspan(-0.5, nd - 0.5, color="tab:blue", alpha=0.06)
    ax.axvspan(nd - 0.5, n - 0.5, color="tab:orange", alpha=0.06)
    ax.set_xlabel("layer"); ax.set_ylabel("localization score (mean over concepts)")
    ax.set_title("Exp4: full vs high-band (RoPE-local) AV localization")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.run_dir, "exp4_score_vs_layer.png"), dpi=140); plt.close(fig)

    # ---- band-averaged full vs high saliency, per concept ----
    Ofull_b = sum(C.merge_heads(layers[i]["av"]) for i in band) / len(band)
    Ohi_b = sum(band_limited_av(layers[i], hi, meta) for i in band) / len(band)
    base = os.path.join(args.run_dir, "generated.png")
    img = plt.imread(base) if os.path.exists(base) else None
    up = max(meta["height"] // meta["h_patches"], 1)
    m = len(used)
    fig, axes = plt.subplots(2, m, figsize=(3.2 * m, 6.6), squeeze=False)
    for j, c in enumerate(used):
        for row, (O, tag) in enumerate([(Ofull_b, "full"), (Ohi_b, "high-band")]):
            sal = saliency_from_output(O, pos_map[c], meta)
            ax = axes[row, j]
            if img is not None:
                ax.imshow(img, extent=[0, 1, 0, 1])
            ax.imshow(upsample(C.normalize01(C.to_grid(sal, meta)), up),
                      extent=[0, 1, 0, 1], cmap="jet", alpha=0.55)
            ax.set_title(f"{c} [{tag}] {localization_score(sal):.2f}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"Exp4 band-avg saliency (layers {band[0]}–{band[-1]}): "
                 f"top=full AV, bottom=high-band AV", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.run_dir, "exp4_concepts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(args.run_dir, "exp4_metrics.json"), "w") as f:
        json.dump({"score_full": sc_full.round(4).tolist(),
                   "score_high": sc_hi.round(4).tolist(),
                   "score_low": sc_lo.round(4).tolist(),
                   "band": band,
                   "band_mean": {"full": float(sc_full[band].mean()),
                                 "high": float(sc_hi[band].mean()),
                                 "low": float(sc_lo[band].mean())},
                   "spatial_only": args.spatial_only}, f, indent=2)
    print(f"[exp4] wrote figures + exp4_metrics.json")


if __name__ == "__main__":
    main()