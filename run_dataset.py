"""
run_dataset.py
==============
Sweep capture + experiments over a dataset of prompts/concepts, loading FLUX
ONCE (so you don't pay the model-load cost per prompt).

Exp1 is per-image (a concept gallery) -> sweep the whole dataset.
Exp2/Exp3 characterize the network -> run them on a HANDFUL of varied images to
confirm the depth trends (hi<lo locality, the L15-26 structural band) are
content-independent. That cross-image agreement is what strengthens validity.

Examples:
  # Exp1 gallery over everything, drop stores to save disk
  python run_dataset.py --run-exp1 --drop-stores

  # Validity check: Exp2+Exp3 on a few diverse scenes (keeps 1 store at a time)
  python run_dataset.py --only dragon,city_night,beach_person,fish_coral \
      --run-exp2 --run-exp3 --drop-stores \
      --exp2-extra "--spatial-only --layers 0,9,15,19,26,38,56 --detail-layer 19"

  # Everything on one image
  python run_dataset.py --only dragon --run-exp1 --run-exp2 --run-exp3

Disk: each 512px store is ~1.5GB and the model is ~34GB on a 50GB disk.
--drop-stores deletes each entry's store after its experiments finish, so only
ONE store sits on disk at a time -- use it for any multi-entry sweep.
"""

import os
import sys
import json
import shlex
import argparse
import subprocess

import torch
import capture as cap  # importing applies the SDPA capture monkeypatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="prompts.json")
    ap.add_argument("--outroot", default="runs")
    ap.add_argument("--only", default=None, help="comma-sep entry names to run")
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--capture-step", type=int, default=2)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    # which experiments to run after each capture
    ap.add_argument("--run-exp1", action="store_true")
    ap.add_argument("--run-exp2", action="store_true")
    ap.add_argument("--run-exp3", action="store_true")
    # extra CLI args passed straight through to each experiment script
    ap.add_argument("--exp1-extra", default="", help="e.g. \"--band 19-30\"")
    ap.add_argument("--exp2-extra", default="", help="e.g. \"--spatial-only --detail-layer 19\"")
    ap.add_argument("--exp3-extra", default="")
    ap.add_argument("--drop-stores", action="store_true",
                    help="delete each store after its experiments run (keeps 1 on disk)")
    args = ap.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)
    entries = data["prompts"] if isinstance(data, dict) else data
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        entries = [e for e in entries if e["name"] in want]
    if not entries:
        print("[dataset] no matching entries.")
        return

    jobs = []
    if args.run_exp1:
        jobs.append(("exp1_av_localization.py", args.exp1_extra))
    if args.run_exp2:
        jobs.append(("exp2_rope_frequency.py", args.exp2_extra))
    if args.run_exp3:
        jobs.append(("exp3_entanglement.py", args.exp3_extra))

    print(f"[dataset] loading {args.model_id} once "
          f"({len(entries)} prompts, experiments: {[j[0].split('_')[0] for j in jobs] or 'capture only'}) ...")
    pipe = cap.load_pipeline(args.model_id, torch.bfloat16,
                             offload=not args.no_offload, cache_dir=args.cache_dir)

    for i, e in enumerate(entries):
        outdir = os.path.join(args.outroot, e["name"])
        print(f"\n[dataset] ({i+1}/{len(entries)}) {e['name']}: {e['prompt']!r}")
        cap.run_capture(
            pipe, e["prompt"], e["concepts"], outdir,
            height=args.height, width=args.width, steps=args.steps,
            capture_step=args.capture_step, guidance=args.guidance,
            max_seq=args.max_seq, seed=args.seed, model_id=args.model_id,
        )
        for script, extra in jobs:
            cmd = [sys.executable, script, "--run-dir", outdir] + shlex.split(extra)
            subprocess.run(cmd, check=False)

        if args.drop_stores and jobs:
            sp = os.path.join(outdir, "capture_store.pt")
            if os.path.exists(sp):
                os.remove(sp)
                print(f"[dataset] removed {sp} to save disk")

    print(f"\n[dataset] done. Per entry: {args.outroot}/<name>/ "
          f"(exp1_concepts.png, exp2_*.png, exp3_entanglement.png as requested)")


if __name__ == "__main__":
    main()