"""
exp6_perturb.py
===============
EXPERIMENT 6 -- frozen (gradient-free) perturbation: a causal sanity check.

Inject (or project out) a concept's AV direction into a chosen SPATIAL REGION,
across a BAND of mid layers and ALL denoising steps, then compare the generated
image to a seed-matched baseline. If the effect localizes to the region and
scales with strength, the AV signal is causally usable for spatial control --
the green light for gradient machinery.

It reuses a reference capture (--run-dir) for: the prompt/seed/size (so the
baseline matches your earlier image) and the concept direction (band-averaged
AV concept vector). Needs only AV in that store.

Usage:
  python exp6_perturb.py --run-dir runs/dragon --concept dragon \
      --band 19-26 --mode add --alphas 0,4,8,16
  # ablation: rerun with --band 15-18 to contrast structural vs semantic band

Outputs (in --outdir, default <run-dir>/exp6_<concept>_<band>):
  baseline.png, perturb_a*.png, summary.png, delta_vs_alpha.png, exp6_metrics.json
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import common as C
import capture as cap
from exp1_av_localization import parse_band


def concept_direction(layers, meta, concept_pos, band):
    """Band-averaged, unit-norm concept AV vector -> (H, D) for the hook."""
    H, D = meta["heads"], meta["head_dim"]
    acc = None
    for li in band:
        O = C.merge_heads(layers[li]["av"])           # (S, H*D)
        txt, _ = C.split_txt_img(O, meta)
        c = txt[concept_pos].mean(0)
        acc = c if acc is None else acc + c
    c = acc / len(band)
    c = c / (c.norm() + 1e-8)
    return c.reshape(H, D)                              # (H, D)


def gen_image(pipe, meta, guidance, max_seq):
    g = torch.Generator(device="cpu").manual_seed(meta["seed"])
    cap.CAP.call_count = 0                              # reset layer/step indexing
    out = pipe(prompt=meta["prompt"], height=meta["height"], width=meta["width"],
               num_inference_steps=meta["steps"], guidance_scale=guidance,
               max_sequence_length=max_seq, generator=g).images[0]
    return np.asarray(out).astype(np.float32)          # (Hpx,Wpx,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/dragon")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--band", default="19-26")
    ap.add_argument("--region", default=None,
                    help="patch box 'r0,c0,r1,c1' (inclusive). Default: central third.")
    ap.add_argument("--mode", choices=["add", "project"], default="add")
    ap.add_argument("--alphas", default="0,4,8,16")
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    layers, meta = C.load_run(args.run_dir)
    band = [b for b in parse_band(args.band) if 0 <= b < meta["n_layers"]]
    pos, matched = C.concept_positions(args.concept, meta)
    if not pos:
        print(f"[exp6] concept '{args.concept}' not in prompt tokens."); return
    print(f"[exp6] concept '{args.concept}' {matched}; band {band}; mode {args.mode}")

    H, Wp = meta["h_patches"], meta["w_patches"]
    if args.region:
        r0, c0, r1, c1 = (int(x) for x in args.region.split(","))
    else:
        r0, r1 = H // 3, 2 * H // 3 - 1
        c0, c1 = Wp // 3, 2 * Wp // 3 - 1
    patch_ids = [r * Wp + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    seq_idx = torch.tensor([meta["txt_len"] + p for p in patch_ids], dtype=torch.long)
    print(f"[exp6] region patches rows {r0}-{r1}, cols {c0}-{c1}  ({len(patch_ids)} patches)")

    vec = concept_direction(layers, meta, pos, band)   # (H,D)
    guidance = meta.get("guidance", 0.0)
    max_seq = meta.get("max_seq", meta["txt_len"])
    outdir = args.outdir or os.path.join(args.run_dir, f"exp6_{args.concept}_{args.band}")
    os.makedirs(outdir, exist_ok=True)

    print(f"[exp6] loading {meta['model_id']} ...")
    pipe = cap.load_pipeline(meta["model_id"], torch.bfloat16,
                             offload=not args.no_offload, cache_dir=args.cache_dir)
    cap.configure_model(pipe)

    # configure the write-hook (alpha set per run)
    cap.CAP.enabled = False
    cap.CAP.perturb_layers = set(band)
    cap.CAP.perturb_idx = seq_idx
    cap.CAP.perturb_vec = vec
    cap.CAP.perturb_mode = args.mode

    # baseline (perturb off)
    cap.CAP.perturb_enabled = False
    base = gen_image(pipe, meta, guidance, max_seq)
    plt.imsave(os.path.join(outdir, "baseline.png"), base.astype(np.uint8))

    # pixel mask for the region
    Hpx, Wpx = base.shape[:2]
    s = meta["height"] // H
    rmask = np.zeros((Hpx, Wpx), bool)
    rmask[r0 * s:(r1 + 1) * s, c0 * s:(c1 + 1) * s] = True

    alphas = [float(a) for a in args.alphas.split(",")]
    results, perturbed_imgs = [], []
    cap.CAP.perturb_enabled = True
    for a in alphas:
        cap.CAP.perturb_alpha = a
        img = gen_image(pipe, meta, guidance, max_seq)
        perturbed_imgs.append(img)
        plt.imsave(os.path.join(outdir, f"perturb_a{a:g}.png"), img.astype(np.uint8))
        diff = np.abs(img - base).mean(2)              # (Hpx,Wpx)
        d_in = float(diff[rmask].mean())
        d_out = float(diff[~rmask].mean())
        ratio = d_in / (d_out + 1e-6)
        results.append({"alpha": a, "delta_in": d_in, "delta_out": d_out, "ratio": ratio})
        print(f"[exp6] alpha={a:5g}  delta_in={d_in:6.2f}  delta_out={d_out:6.2f}  in/out={ratio:.2f}")
    cap.CAP.perturb_enabled = False

    # ---- summary figure: baseline + perturbed + diff, region boxed ----
    ncol = len(alphas) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 5.4))
    def box(ax): ax.add_patch(Rectangle((c0 * s, r0 * s), (c1 - c0 + 1) * s,
                                        (r1 - r0 + 1) * s, fill=False, ec="lime", lw=1.5))
    axes[0, 0].imshow(base.astype(np.uint8)); axes[0, 0].set_title("baseline", fontsize=9)
    box(axes[0, 0]); axes[1, 0].axis("off")
    for j, (a, img) in enumerate(zip(alphas, perturbed_imgs), start=1):
        axes[0, j].imshow(img.astype(np.uint8)); axes[0, j].set_title(f"α={a:g}", fontsize=9); box(axes[0, j])
        d = np.abs(img - base).mean(2)
        axes[1, j].imshow(d, cmap="inferno"); axes[1, j].set_title(f"|Δ| in/out={results[j-1]['ratio']:.2f}", fontsize=8); box(axes[1, j])
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Exp6 {args.mode} '{args.concept}' band {args.band}  "
                 f"(top: images, bottom: |perturbed-baseline|)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary.png"), dpi=140); plt.close(fig)

    # ---- delta vs alpha ----
    fig, ax = plt.subplots(figsize=(6, 4))
    al = [r["alpha"] for r in results]
    ax.plot(al, [r["delta_in"] for r in results], "o-", label="Δ inside region")
    ax.plot(al, [r["delta_out"] for r in results], "s-", label="Δ outside region")
    ax.set_xlabel("perturbation strength α"); ax.set_ylabel("mean |pixel change|")
    ax.set_title("Exp6: causal effect vs strength (want in ≫ out, monotonic)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "delta_vs_alpha.png"), dpi=140); plt.close(fig)

    with open(os.path.join(outdir, "exp6_metrics.json"), "w") as f:
        json.dump({"concept": args.concept, "band": band, "mode": args.mode,
                   "region_rc": [r0, c0, r1, c1], "results": results}, f, indent=2)
    print(f"[exp6] wrote {outdir}/ (summary.png, delta_vs_alpha.png, exp6_metrics.json)")


if __name__ == "__main__":
    main()