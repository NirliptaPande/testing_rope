"""
exp6b_concept_metric.py
=======================
EXPERIMENT 6b -- measure the perturbation in CONCEPT space, not pixels.

Exp6 showed AV edits cause localized PIXEL change, but the pixel in/out ratio
measures disruption, not concept-correctness (random scored higher than the
concept direction). 6b asks the right question: does perturbing the region's AV
move the *concept readout* the intended way, and does the concept direction do
this while a RANDOM direction does not?

It perturbs as in Exp6 (band, region, all steps) but ALSO captures AV during the
perturbed generation, then measures, per layer, the change in region-mean
dragon-saliency (cosine of image-patch AV onto the clean, unperturbed text-concept
vector). At the edited band the rise is trivial (we injected it); the informative
signal is DOWNSTREAM of the band -- does it persist?

Run the same condition you ran in Exp6, then rerun with --control-random and
compare the downstream curves:
  python exp6b_concept_metric.py --run-dir runs/dragon_v --concept dragon --band 19-26 --alphas 0,4,8
  python exp6b_concept_metric.py --run-dir runs/dragon_v --concept dragon --band 19-26 --alphas 0,4,8 --control-random

Outputs (in --outdir):
  exp6b_saliency_vs_layer.png   Δ region dragon-saliency per layer (band shaded)
  exp6b_delta_vs_alpha.png      downstream Δ saliency vs α
  exp6b_metrics.json
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
import capture as cap
from exp1_av_localization import parse_band
from exp6_perturb import concept_direction


def gen_and_capture(pipe, meta, guidance, max_seq, perturb_on):
    """Generate (optionally perturbed) and return {layer: merged AV} at the
    capture step."""
    cap.CAP.store = []
    cap.CAP.call_count = 0
    cap.CAP.enabled = True
    cap.CAP.av_only = True
    cap.CAP.all_steps = False
    cap.CAP.target_step = meta.get("capture_step", 0)
    cap.CAP.perturb_enabled = perturb_on
    g = torch.Generator(device="cpu").manual_seed(meta["seed"])
    pipe(prompt=meta["prompt"], height=meta["height"], width=meta["width"],
         num_inference_steps=meta["steps"], guidance_scale=guidance,
         max_sequence_length=max_seq, generator=g)
    cap.CAP.enabled = False
    cap.CAP.perturb_enabled = False
    return {e["layer"]: C.merge_heads(e["av"]) for e in cap.CAP.store}


def region_saliency(O, probe_unit, meta, patch_ids):
    """Mean cosine(image-patch AV, probe) over the region patches."""
    _, img = C.split_txt_img(O, meta)
    img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
    sal = (img @ probe_unit).numpy()
    return float(sal[patch_ids].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/dragon_v")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--band", default="19-26")
    ap.add_argument("--region", default=None, help="patch box 'r0,c0,r1,c1'; default central third")
    ap.add_argument("--mode", choices=["add", "project"], default="add")
    ap.add_argument("--control-random", action="store_true")
    ap.add_argument("--alphas", default="0,4,8")
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    layers, meta = C.load_run(args.run_dir)
    band = [b for b in parse_band(args.band) if 0 <= b < meta["n_layers"]]
    pos, matched = C.concept_positions(args.concept, meta)
    if not pos:
        print(f"[exp6b] concept '{args.concept}' not in prompt tokens."); return

    H, Wp = meta["h_patches"], meta["w_patches"]
    if args.region:
        r0, c0, r1, c1 = (int(x) for x in args.region.split(","))
    else:
        r0, r1, c0, c1 = H // 3, 2 * H // 3 - 1, Wp // 3, 2 * Wp // 3 - 1
    patch_ids = [r * Wp + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    seq_idx = torch.tensor([meta["txt_len"] + p for p in patch_ids], dtype=torch.long)

    # perturbation direction (concept or random control)
    if args.control_random:
        gd = torch.Generator().manual_seed(1234)
        rr = torch.randn(meta["heads"] * meta["head_dim"], generator=gd)
        vec = (rr / (rr.norm() + 1e-8)).reshape(meta["heads"], meta["head_dim"])
        dir_tag = "random"
    else:
        vec = concept_direction(layers, meta, pos, band)
        dir_tag = args.concept

    guidance = meta.get("guidance", 0.0)
    max_seq = meta.get("max_seq", meta["txt_len"])
    regtag = "r" + args.region.replace(",", "-") if args.region else "center"
    outdir = args.outdir or os.path.join(
        args.run_dir, f"exp6b_{dir_tag}_{args.mode}_{args.band}_{regtag}")
    os.makedirs(outdir, exist_ok=True)

    print(f"[exp6b] loading {meta['model_id']} ...")
    pipe = cap.load_pipeline(meta["model_id"], torch.bfloat16,
                             offload=not args.no_offload, cache_dir=args.cache_dir)
    cap.configure_model(pipe)
    cap.CAP.perturb_layers = set(band)
    cap.CAP.perturb_idx = seq_idx
    cap.CAP.perturb_vec = vec
    cap.CAP.perturb_mode = args.mode

    # baseline capture (no perturb) -> also defines the clean per-layer probe
    cap.CAP.perturb_alpha = 0.0
    base = gen_and_capture(pipe, meta, guidance, max_seq, perturb_on=False)
    nL = meta["n_layers"]
    probe = {}
    for l in range(nL):
        txt, _ = C.split_txt_img(base[l], meta)
        c = txt[pos].mean(0)
        probe[l] = c / (c.norm() + 1e-8)
    base_sal = {l: region_saliency(base[l], probe[l], meta, patch_ids) for l in range(nL)}

    alphas = [float(a) for a in args.alphas.split(",")]
    curves = {}  # alpha -> per-layer Δ saliency
    for a in alphas:
        if a == 0:
            curves[a] = np.zeros(nL); continue
        cap.CAP.perturb_alpha = a
        pert = gen_and_capture(pipe, meta, guidance, max_seq, perturb_on=True)
        d = np.array([region_saliency(pert[l], probe[l], meta, patch_ids) - base_sal[l]
                      for l in range(nL)])
        curves[a] = d
        print(f"[exp6b] α={a:g}  Δsaliency band={d[band].mean():+.3f}  "
              f"downstream={d[max(band)+1:min(max(band)+9, nL)].mean():+.3f}")

    # downstream window = the 8 layers just after the edited band
    ds = list(range(max(band) + 1, min(max(band) + 9, nL)))

    # ---- per-layer Δ saliency ----
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(nL)
    for a in alphas:
        if a == 0:
            continue
        ax.plot(x, curves[a], "o-", ms=2.5, label=f"α={a:g}")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(min(band) - 0.5, max(band) + 0.5, color="tab:red", alpha=0.12, label="edited band")
    ax.axvspan(ds[0] - 0.5, ds[-1] + 0.5, color="tab:green", alpha=0.10, label="downstream readout")
    ax.set_xlabel("layer"); ax.set_ylabel(f"Δ region '{args.concept}' saliency")
    ax.set_title(f"Exp6b: {args.mode} '{dir_tag}' — concept-space effect per layer")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "exp6b_saliency_vs_layer.png"), dpi=140); plt.close(fig)

    # ---- downstream Δ saliency vs alpha ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ds_delta = [float(curves[a][ds].mean()) for a in alphas]
    ax.plot(alphas, ds_delta, "o-")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("α"); ax.set_ylabel(f"downstream Δ '{args.concept}' saliency")
    ax.set_title(f"Exp6b downstream concept effect ({dir_tag}). "
                 f"concept↑ & random≈0 ⇒ specific")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "exp6b_delta_vs_alpha.png"), dpi=140); plt.close(fig)

    with open(os.path.join(outdir, "exp6b_metrics.json"), "w") as f:
        json.dump({"direction": dir_tag, "concept": args.concept, "mode": args.mode,
                   "band": band, "downstream_layers": ds,
                   "delta_saliency_band": {a: float(curves[a][band].mean()) for a in alphas},
                   "delta_saliency_downstream": {a: float(curves[a][ds].mean()) for a in alphas}},
                  f, indent=2)
    print(f"[exp6b] wrote {outdir}/  (compare downstream curve: concept vs --control-random)")


if __name__ == "__main__":
    main()