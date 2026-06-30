"""
aggregate.py
============
Crunch the per-run JSON outputs of Exp1/2/3 across ALL runs into cross-image
conclusions + summary figures. Pure numpy/json -- no GPU, no store needed.

It walks --root for subdirs containing exp1_scores.json / exp2_metrics.json /
exp3_metrics.json and computes, across images:

  Exp1  mean localization score per layer (+ how consistent the best layers are)
  Exp2  hi<lo hit-rate (does the RoPE claim hold?) and mean low-minus-high gap
  Exp3  mean contamination / entropy / spread per layer (+ severity band)
  Combined  an "operating score" per layer = high Exp1 signal AND low Exp3
            severity (AND large Exp2 gap if present), to recommend the band.

Outputs (in --out):
  aggregate_summary.json      all the numbers
  aggregate_exp1.png          mean score vs layer (+/- std across images)
  aggregate_exp2.png          mean hi/lo/full spread (+ hit-rate annotation)
  aggregate_exp3.png          mean cross/entropy/spread vs layer
  aggregate_operating.png     combined operating score, recommended band shaded
And a concise text summary is printed.

Usage:
  python aggregate.py --root runs --out runs/_aggregate
"""

import os
import json
import glob
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def norm01(a):
    a = np.asarray(a, float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo + 1e-12)


def find_runs(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        if any(os.path.exists(os.path.join(d, f)) for f in
               ("exp1_scores.json", "exp2_metrics.json", "exp3_metrics.json")):
            runs.append(d)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="runs/_aggregate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = find_runs(args.root)
    if not runs:
        print(f"[agg] no run dirs with exp*_*.json under {args.root}")
        return
    names = [os.path.basename(r) for r in runs]
    print(f"[agg] {len(runs)} runs: {names}")

    summary = {"runs": names}
    n_double = None
    L = None  # number of layers

    # ---------------- Exp1 ----------------
    e1_curves, e1_best = [], {}      # list of (57,) score arrays; per (run/concept) best layer
    for r, name in zip(runs, names):
        d = _load(os.path.join(r, "exp1_scores.json"))
        if not d:
            continue
        n_double = d.get("n_double", n_double)
        for concept, scores in d["concepts"].items():
            arr = np.asarray(scores, float)
            L = len(arr) if L is None else L
            e1_curves.append(arr)
            e1_best[f"{name}/{concept}"] = int(np.argmax(arr))
    if e1_curves:
        M = np.vstack(e1_curves)                 # (n_series, L)
        e1_mean, e1_std = M.mean(0), M.std(0)
        top = np.argsort(e1_mean)[::-1][:8]
        summary["exp1"] = {
            "n_series": int(M.shape[0]),
            "mean_score_per_layer": e1_mean.round(4).tolist(),
            "top8_layers_by_mean": sorted(int(i) for i in top),
            "best_layer_per_series": e1_best,
        }
        print(f"[agg] Exp1: {M.shape[0]} curves; top layers by mean score = "
              f"{sorted(int(i) for i in top)}")

    # ---------------- Exp2 ----------------
    # per-layer accumulators (layers can differ across runs)
    acc_hi, acc_lo, acc_full = {}, {}, {}
    hits = total = 0
    for r in runs:
        d = _load(os.path.join(r, "exp2_metrics.json"))
        if not d:
            continue
        for j, ly in enumerate(d["layers"]):
            hi, lo, fu = d["spread_hi"][j], d["spread_lo"][j], d["spread_full"][j]
            acc_hi.setdefault(ly, []).append(hi)
            acc_lo.setdefault(ly, []).append(lo)
            acc_full.setdefault(ly, []).append(fu)
            total += 1
            hits += int(hi < lo)
    if total:
        lys = sorted(acc_hi)
        hit_rate = hits / total
        summary["exp2"] = {
            "hi_lt_lo_hit_rate": round(hit_rate, 4),
            "n_pairs": total,
            "layers": lys,
            "mean_hi": [round(float(np.mean(acc_hi[l])), 3) for l in lys],
            "mean_lo": [round(float(np.mean(acc_lo[l])), 3) for l in lys],
            "mean_gap_lo_minus_hi": [round(float(np.mean(acc_lo[l]) - np.mean(acc_hi[l])), 3) for l in lys],
        }
        print(f"[agg] Exp2: hi<lo holds in {hits}/{total} = {hit_rate:.1%} of (image,layer) pairs")

    # ---------------- Exp3 ----------------
    e3_cross, e3_ent, e3_spread = [], [], []
    for r in runs:
        d = _load(os.path.join(r, "exp3_metrics.json"))
        if not d:
            continue
        n_double = d.get("n_double", n_double)
        e3_cross.append(np.asarray(d["cross_fraction"], float))
        e3_ent.append(np.asarray(d["ii_entropy"], float))
        e3_spread.append(np.asarray(d["ii_spread"], float))
        L = len(d["cross_fraction"]) if L is None else L
    if e3_cross:
        C = np.vstack(e3_cross); E = np.vstack(e3_ent); S = np.vstack(e3_spread)
        cross_m, ent_m, spread_m = C.mean(0), E.mean(0), S.mean(0)
        severity = norm01(cross_m) + norm01(ent_m) + norm01(spread_m)
        low_sev = np.argsort(severity)[:8]
        summary["exp3"] = {
            "n_images": int(C.shape[0]),
            "mean_cross_per_layer": cross_m.round(4).tolist(),
            "mean_ii_entropy_per_layer": ent_m.round(4).tolist(),
            "mean_ii_spread_per_layer": spread_m.round(3).tolist(),
            "lowest_severity_layers": sorted(int(i) for i in low_sev),
        }
        print(f"[agg] Exp3: lowest-severity layers (structure survives) = "
              f"{sorted(int(i) for i in low_sev)}")

    # ---------------- Combined operating score ----------------
    if e1_curves and e3_cross and L:
        op = norm01(e1_mean) + (1.0 - norm01(severity))
        # add Exp2 gap if we can align it to all layers
        if total:
            gap_full = np.full(L, np.nan)
            for l in acc_lo:
                if 0 <= l < L:
                    gap_full[l] = np.mean(acc_lo[l]) - np.mean(acc_hi[l])
            if np.isfinite(gap_full).any():
                g = np.where(np.isfinite(gap_full), gap_full, np.nanmin(gap_full))
                op = op + norm01(g)
        best_band = sorted(int(i) for i in np.argsort(op)[::-1][:8])
        summary["operating_point"] = {
            "score_per_layer": op.round(4).tolist(),
            "recommended_band_top8": best_band,
            "recommended_range": [min(best_band), max(best_band)],
        }
        print(f"[agg] COMBINED operating band (Exp1 high + Exp3 low-severity"
              f"{' + Exp2 gap' if total else ''}): layers {best_band}")

    with open(os.path.join(args.out, "aggregate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---------------- figures ----------------
    def shade(ax):
        if n_double and L:
            ax.axvspan(-0.5, n_double - 0.5, color="tab:blue", alpha=0.07)
            ax.axvspan(n_double - 0.5, L - 0.5, color="tab:orange", alpha=0.07)

    if e1_curves:
        x = np.arange(L)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, e1_mean, "o-", ms=3, color="tab:green")
        ax.fill_between(x, e1_mean - e1_std, e1_mean + e1_std, alpha=0.2, color="tab:green")
        shade(ax)
        ax.set_title(f"Exp1 mean localization score across {M.shape[0]} image/concept series")
        ax.set_xlabel("layer"); ax.set_ylabel("top-decile mass (0.10 = none)")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "aggregate_exp1.png"), dpi=140); plt.close(fig)

    if total:
        lys = sorted(acc_hi)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lys, [np.mean(acc_hi[l]) for l in lys], "o-", label="high band (mean)")
        ax.plot(lys, [np.mean(acc_lo[l]) for l in lys], "s-", label="low band (mean)")
        ax.plot(lys, [np.mean(acc_full[l]) for l in lys], "x--", color="gray", label="all channels")
        ax.set_title(f"Exp2 mean spread across images  (hi<lo in {hit_rate:.0%} of pairs)")
        ax.set_xlabel("layer"); ax.set_ylabel("mean attention spread"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "aggregate_exp2.png"), dpi=140); plt.close(fig)

    if e3_cross:
        x = np.arange(L)
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, m, sd, t in [(axs[0], cross_m, C.std(0), "cross (contamination)"),
                             (axs[1], ent_m, E.std(0), "ii entropy"),
                             (axs[2], spread_m, S.std(0), "ii spread")]:
            ax.plot(x, m, "o-", ms=3); ax.fill_between(x, m - sd, m + sd, alpha=0.2)
            shade(ax); ax.set_title(t); ax.set_xlabel("layer")
        fig.suptitle(f"Exp3 mean +/- std across {C.shape[0]} images")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "aggregate_exp3.png"), dpi=140); plt.close(fig)

    if "operating_point" in summary:
        x = np.arange(L)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, op, "o-", ms=3, color="crimson")
        for b in best_band:
            ax.axvline(b, color="crimson", alpha=0.15)
        shade(ax)
        ax.set_title(f"Combined operating score (higher=better). Band: {summary['operating_point']['recommended_range']}")
        ax.set_xlabel("layer"); ax.set_ylabel("Exp1 high + Exp3 low-sev (+ Exp2 gap)")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "aggregate_operating.png"), dpi=140); plt.close(fig)

    print(f"[agg] wrote summary + figures to {args.out}/")


if __name__ == "__main__":
    main()