"""
exp5_layer_timestep.py
======================
EXPERIMENT 5 -- layer x timestep sweep.

Computes the Exp1 localization score for every (layer, denoising-step) pair and
renders a 2D heatmap, so you can read off the optimal depth AND timestep for the
cleanest concept signal (ConceptAttention found ~mid-schedule for semantics).

Requires a store captured with --all-steps (use --av-only to keep it small):
    python capture.py --all-steps --av-only --outdir runs/sweep ...
On schnell that's 4 coarse timesteps; for real timestep resolution capture on
FLUX.1-dev with --steps 50.

Outputs:
  exp5_heatmap.png     layer (x) x timestep (y) mean localization score
  exp5_scores.json
"""

import os
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
from exp1_av_localization import saliency_from_output, localization_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/sweep")
    ap.add_argument("--concepts", default=None)
    args = ap.parse_args()

    layers, meta = C.load_run(args.run_dir)
    steps = meta.get("captured_steps") or sorted({e["step"] for e in layers})
    if len(steps) < 2:
        print(f"[exp5] only {len(steps)} timestep(s) in store. "
              f"Re-capture with --all-steps (and --steps 50 on dev for resolution).")
    nL = meta["n_layers"]
    concepts = ([c.strip() for c in args.concepts.split(",")]
                if args.concepts else meta["concepts"])

    pos_map, used = {}, []
    for c in concepts:
        p, matched = C.concept_positions(c, meta)
        if p:
            pos_map[c] = p; used.append(c)
    if not used:
        print("[exp5] no concepts matched."); return

    # index store by (step, layer)
    by = {(e["step"], e["layer"]): e for e in layers}
    grid = np.full((len(steps), nL), np.nan)  # [step, layer] mean score
    for si, s in enumerate(steps):
        for l in range(nL):
            e = by.get((s, l))
            if e is None:
                continue
            O = C.merge_heads(e["av"])
            grid[si, l] = np.mean([localization_score(saliency_from_output(O, pos_map[c], meta))
                                   for c in used])

    # best operating point
    flat = np.nanargmax(grid)
    bs, bl = np.unravel_index(flat, grid.shape)
    best_step, best_layer = steps[bs], int(bl)
    best_layer_per_step = {int(s): int(np.nanargmax(grid[si])) for si, s in enumerate(steps)}
    print(f"[exp5] best (layer, step) = ({best_layer}, {best_step}) "
          f"score={grid[bs, bl]:.3f}")
    print(f"[exp5] best layer per step: {best_layer_per_step}")

    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.5 * len(steps) + 1.5)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis",
                   extent=[0, nL, len(steps) - 0.5, -0.5])
    ax.axvline(meta["n_double"], color="white", ls="--", lw=1, alpha=0.7)
    ax.set_yticks(range(len(steps))); ax.set_yticklabels(steps)
    ax.set_xlabel("layer  (dashed = double→single)"); ax.set_ylabel("denoising step")
    ax.scatter([best_layer + 0.5], [bs], marker="*", s=180, c="red", edgecolors="white")
    ax.set_title(f"Exp5 localization score (layer x timestep). "
                 f"best=(L{best_layer}, step {best_step})")
    plt.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(os.path.join(args.run_dir, "exp5_heatmap.png"), dpi=140); plt.close(fig)

    with open(os.path.join(args.run_dir, "exp5_scores.json"), "w") as f:
        json.dump({"steps": list(steps), "n_layers": nL,
                   "score_grid_step_by_layer": np.nan_to_num(grid).round(4).tolist(),
                   "best_layer": best_layer, "best_step": int(best_step),
                   "best_layer_per_step": best_layer_per_step}, f, indent=2)
    print(f"[exp5] wrote exp5_heatmap.png + exp5_scores.json")


if __name__ == "__main__":
    main()