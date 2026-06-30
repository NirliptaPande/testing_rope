#!/usr/bin/env bash
# Minimal RunPod setup + smoke run. Use a PyTorch GPU pod with >=32GB VRAM
# (e.g. RTX PRO 4500 / RTX 5090 / L40S). 24GB cards are tight for FLUX bf16.
set -e

# --- persistent cache so the ~34GB model is downloaded ONCE -------------------
# /workspace is the persistent volume; / is wiped on pod restart.
# We use --cache-dir below to put the model on the volume (single mechanism,
# no HF_HOME needed). Verify your volume has ~34GB free quota:  du -sh /workspace
export HF_HUB_DISABLE_XET=1          # avoid xet's ~2x disk usage / quota errors
unset HF_HOME                        # avoid HF_HOME's ~2x disk usage / quota errors

pip install -r requirements.txt

# FLUX.1-schnell is ungated (Apache-2.0) -> no token needed.
# For FLUX.1-dev (gated) instead: accept the license on HF, then:
#   export HF_TOKEN=hf_xxx ; huggingface-cli login --token $HF_TOKEN
# and pass --model-id black-forest-labs/FLUX.1-dev --steps 50 --guidance 3.5 --max-seq 512

RUN=runs/run0

# capture re-runs are skipped automatically if the model is already cached.
python capture.py \
  --prompt "a cat sitting on a wooden table" \
  --concepts "cat,table" \
  --height 512 --width 512 --steps 4 --capture-step 2 \
  --outdir "$RUN"

# experiments need neither GPU nor the model -- they just read the store.
python exp1_av_localization.py --run-dir "$RUN"
python exp2_rope_frequency.py  --run-dir "$RUN" \
  --layers 0,5,9,12,15,18,19,22,26,30,38,45,56 --detail-layer 19
python exp3_entanglement.py    --run-dir "$RUN"

echo "Done. See $RUN/*.png and $RUN/*.json"