#!/usr/bin/env bash
# Minimal RunPod setup + smoke run. Use a PyTorch GPU pod (>=24GB, e.g. RTX 4090).
set -e

pip install -r requirements.txt

# FLUX.1-schnell is ungated (Apache-2.0) -> no token needed.
# For FLUX.1-dev (gated) instead: accept the license on HF, then:
#   export HF_TOKEN=hf_xxx ; huggingface-cli login --token $HF_TOKEN
# and pass --model-id black-forest-labs/FLUX.1-dev --steps 50 --guidance 3.5 --max-seq 512

RUN=runs/run0

python capture.py \
  --prompt "a cat sitting on a wooden table" \
  --concepts "cat,table" \
  --height 512 --width 512 --steps 4 --capture-step 2 \
  --outdir "$RUN"

python exp1_av_localization.py --run-dir "$RUN"
python exp2_rope_frequency.py  --run-dir "$RUN"
python exp3_entanglement.py    --run-dir "$RUN"

echo "Done. See $RUN/*.png and $RUN/*.json"