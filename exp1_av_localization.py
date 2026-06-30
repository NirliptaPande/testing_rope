"""
exp1_av_localization.py
=======================
EXPERIMENT 1 -- AV localization quality (the foundation test).

For each concept word in the prompt, and for EVERY captured layer, project that
concept's attention-output (AV) vector onto every image patch's AV vector
(cosine similarity in the concatenated-head "output space"). This is the
ConceptAttention finding: the output space of DiT attention localizes concepts
far more sharply than raw cross-attention.

Two outputs:
  * per-layer saliency grid + score (which depth localizes cleanest), and
  * a BAND-AVERAGED, multi-concept figure (ConceptAttention style): average AV
    over the best band of layers, then show every concept's map side by side.
    Band-averaging sharpens by cancelling per-layer speckle WITHOUT needing a
    cat-vs-table contrast (works for any number of concepts).

Outputs (in run dir):
  exp1_<concept>_grid.png    per-layer saliency for that concept
  exp1_concepts.png          band-averaged map for ALL concepts, side by side
  exp1_scores.json           per-layer scores + the band used

Localization score = fraction of positive saliency mass in the top 10% of
patches (0.10 == no localization; higher == sharper).
"""

import os
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


def saliency_from_output(O, concept_pos, meta):
    """O: (S, H*D) output-space tensor -> (img_len,) cosine saliency."""
    txt, img = C.split_txt_img(O, meta)
    c = txt[concept_pos].mean(0)
    c = c / (c.norm() + 1e-8)
    img = img / (img.norm(dim=-1, keepdim=True) + 1e-8)
    return (img @ c).numpy()


def localization_score(sal):
    pos = np.clip(sal, 0, None)
    if pos.sum() <= 0:
        return 0.0
    k = max(1, int(0.10 * pos.size))
    return float(np.sort(pos)[::-1][:k].sum() / pos.sum())


def upsample(grid, factor=16):
    return np.kron(grid, np.ones((factor, factor)))


def parse_band(s):
    """'19-30' -> [19..30]; '19,22,26' -> [19,22,26]; None -> None."""
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/run0")
    ap.add_argument("--concepts", default=None, help="override meta concepts (comma-sep)")
    ap.add_argument("--band", default=None,
                    help="layers to average for the multi-concept figure, e.g. "
                         "'19-30' or '19,22,26'. Default: auto top-K by score.")
    ap.add_argument("--topk", type=int, default=8,
                    help="if --band not given, average the K best-scoring layers")
    ap.add_argument("--no-grids", action="store_true",
                    help="skip the per-layer per-concept grids (faster)")
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

    # resolve concept -> text positions, drop misses
    pos_map, used = {}, []
    for concept in concepts:
        pos, matched = C.concept_positions(concept, meta)
        if pos:
            pos_map[concept] = pos
            used.append(concept)
            print(f"[exp1] '{concept}' -> tokens {matched} at {pos}")
        else:
            print(f"[exp1] '{concept}': no T5 tokens matched (is it in the prompt?), skipping.")
    if not used:
        print("[exp1] no concepts matched the prompt; nothing to do.")
        return

    # ---- pass 1: per-layer saliency grids + scores ----
    n = len(layers)
    grids = {c: [] for c in used}      # per concept: list of (h,w) grids
    scores = {c: [] for c in used}
    for li, lyr in enumerate(layers):
        O = C.merge_heads(lyr["av"])
        for c in used:
            sal = saliency_from_output(O, pos_map[c], meta)
            grids[c].append(C.to_grid(sal, meta))
            scores[c].append(localization_score(sal))

    # rank + report cleanest layers per concept
    for c in used:
        order = np.argsort(scores[c])[::-1]
        print(f"[exp1] '{c}' cleanest layers: " + ", ".join(
            f"L{int(i)}({C.layer_kind(int(i), meta)[0]}){scores[c][int(i)]:.2f}"
            for i in order[:5]))

    # ---- choose the band ----
    band = parse_band(args.band)
    if band is None:
        mean_score = np.mean([scores[c] for c in used], axis=0)   # over concepts
        band = sorted(np.argsort(mean_score)[::-1][: args.topk].tolist())
    band = [b for b in band if 0 <= b < n]
    print(f"[exp1] band-averaging over layers: {band}")

    # ---- pass 2: band-averaged output space ----
    O_band = None
    for li in band:
        O = C.merge_heads(layers[li]["av"])
        O_band = O if O_band is None else O_band + O
    O_band = O_band / len(band)

    # ---- per-layer grids (optional) ----
    if not args.no_grids:
        for c in used:
            best = int(np.argmax(scores[c]))
            cols, rows = 8, int(np.ceil(n / 8))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.7))
            axes = np.atleast_1d(axes).ravel()
            for li in range(len(axes)):
                ax = axes[li]; ax.axis("off")
                if li >= n:
                    continue
                if base_img is not None:
                    ax.imshow(base_img, extent=[0, 1, 0, 1])
                ax.imshow(upsample(C.normalize01(grids[c][li]), up),
                          extent=[0, 1, 0, 1], cmap="jet", alpha=0.55)
                col = "yellow" if li == best else "white"
                ax.set_title(f"L{li}{C.layer_kind(li, meta)[0]} {scores[c][li]:.2f}",
                             fontsize=6, color=col)
            fig.suptitle(f"Exp1 AV saliency '{c}' (yellow = cleanest layer)", fontsize=10)
            fig.tight_layout()
            out = os.path.join(args.run_dir, f"exp1_{c}_grid.png")
            fig.savefig(out, dpi=130); plt.close(fig)
            print(f"[exp1] wrote {out}")

    # ---- multi-concept band-averaged figure (ConceptAttention style) ----
    m = len(used)
    fig, axes = plt.subplots(1, m, figsize=(3.2 * m, 3.4))
    axes = np.atleast_1d(axes).ravel()
    for j, c in enumerate(used):
        sal = saliency_from_output(O_band, pos_map[c], meta)
        ax = axes[j]
        if base_img is not None:
            ax.imshow(base_img, extent=[0, 1, 0, 1])
        ax.imshow(upsample(C.normalize01(C.to_grid(sal, meta)), up),
                  extent=[0, 1, 0, 1], cmap="jet", alpha=0.55)
        ax.set_title(f"{c}  (score {localization_score(sal):.2f})", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Exp1 band-averaged saliency  (layers {band[0]}–{band[-1]}, "
                 f"{len(band)} layers)", fontsize=12)
    fig.tight_layout()
    out_multi = os.path.join(args.run_dir, "exp1_concepts.png")
    fig.savefig(out_multi, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[exp1] wrote {out_multi}")

    with open(os.path.join(args.run_dir, "exp1_scores.json"), "w") as f:
        json.dump({"concepts": {c: scores[c] for c in used},
                   "band": band, "n_double": meta["n_double"]}, f, indent=2)
    print("[exp1] done.")


if __name__ == "__main__":
    main()