"""
exp1_av_localization.py
=======================
EXPERIMENT 1 -- AV localization quality (the foundation test).

For each concept word in the prompt, and for EVERY captured layer, we project
that concept's attention-output (AV) vector onto every image patch's AV vector
(cosine similarity in the concatenated-head "output space"). This is the
ConceptAttention finding: the output space of DiT attention localizes concepts
far more sharply than raw cross-attention.

Outputs (per concept):
  exp1_<concept>_grid.png    saliency for all layers (overlaid on the image)
  exp1_<concept>_best.png    the single cleanest layer, larger
  exp1_scores.json           per-layer localization score (top-decile mass)

The localization score = fraction of total (positive) saliency mass that lands
in the top 10% of patches. Higher = sharper / cleaner localization. This is the
number that tells you WHICH DEPTH carries clean spatial-semantic signal.
"""

import os
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


def cosine_saliency(av, concept_pos, meta):
    """av: (B,H,S,D) tensor -> (img_len,) cosine-sim saliency for the concept."""
    O = C.merge_heads(av)                       # (S, H*D)
    txt, img = C.split_txt_img(O, meta)         # (txt,C), (img,C)
    c = txt[concept_pos].mean(0)                # (C,)
    c = c / (c.norm() + 1e-8)
    img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
    sal = (img @ c).numpy()                     # (img_len,)
    return sal


def localization_score(sal):
    """Fraction of positive saliency mass in the top 10% of patches."""
    pos = np.clip(sal, 0, None)
    if pos.sum() <= 0:
        return 0.0
    k = max(1, int(0.10 * pos.size))
    top = np.sort(pos)[::-1][:k].sum()
    return float(top / pos.sum())


def upsample(grid, factor=16):
    return np.kron(grid, np.ones((factor, factor)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/run0")
    ap.add_argument("--concepts", default=None, help="override meta concepts (comma-sep)")
    args = ap.parse_args()

    layers, meta = C.load_run(args.run_dir)
    concepts = (
        [c.strip() for c in args.concepts.split(",")]
        if args.concepts
        else meta["concepts"]
    )

    img_path = os.path.join(args.run_dir, "generated.png")
    base_img = plt.imread(img_path) if os.path.exists(img_path) else None
    up = max(meta["height"] // meta["h_patches"], 1)

    all_scores = {}
    for concept in concepts:
        pos, matched = C.concept_positions(concept, meta)
        if not pos:
            print(f"[exp1] '{concept}': no T5 tokens matched, skipping.")
            continue
        print(f"[exp1] '{concept}' -> tokens {matched} at {pos}")

        grids, scores = [], []
        for li, lyr in enumerate(layers):
            sal = cosine_saliency(lyr["av"], pos, meta)
            grids.append(C.to_grid(sal, meta))
            scores.append(localization_score(sal))
        all_scores[concept] = scores

        order = np.argsort(scores)[::-1]
        best = int(order[0])
        print(
            f"[exp1] '{concept}' cleanest layers (by top-decile mass): "
            + ", ".join(
                f"L{int(i)}({C.layer_kind(int(i), meta)[0]}){scores[int(i)]:.2f}"
                for i in order[:5]
            )
        )

        # ---- grid figure: all layers ----
        n = len(grids)
        cols = 8
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.7))
        axes = np.atleast_1d(axes).ravel()
        for li in range(len(axes)):
            ax = axes[li]
            ax.axis("off")
            if li >= n:
                continue
            if base_img is not None:
                ax.imshow(base_img, extent=[0, 1, 0, 1])
            sal_up = upsample(C.normalize01(grids[li]), up)
            ax.imshow(sal_up, extent=[0, 1, 0, 1], cmap="jet", alpha=0.55)
            kind = C.layer_kind(li, meta)[0]
            color = "yellow" if li == best else "white"
            ax.set_title(f"L{li}{kind} {scores[li]:.2f}", fontsize=6, color=color)
        fig.suptitle(f"Exp1 AV saliency '{concept}' (yellow = cleanest)", fontsize=10)
        fig.tight_layout()
        out_grid = os.path.join(args.run_dir, f"exp1_{concept}_grid.png")
        fig.savefig(out_grid, dpi=130)
        plt.close(fig)

        # ---- best-layer overlay ----
        fig, ax = plt.subplots(figsize=(4, 4))
        if base_img is not None:
            ax.imshow(base_img, extent=[0, 1, 0, 1])
        ax.imshow(upsample(C.normalize01(grids[best]), up), extent=[0, 1, 0, 1],
                  cmap="jet", alpha=0.55)
        ax.set_title(f"'{concept}' best = L{best} ({C.layer_kind(best, meta)}) "
                     f"score={scores[best]:.2f}")
        ax.axis("off")
        out_best = os.path.join(args.run_dir, f"exp1_{concept}_best.png")
        fig.savefig(out_best, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[exp1] wrote {out_grid} and {out_best}")

    with open(os.path.join(args.run_dir, "exp1_scores.json"), "w") as f:
        json.dump({"concepts": all_scores, "n_double": meta["n_double"]}, f, indent=2)
    print("[exp1] done.")


if __name__ == "__main__":
    main()