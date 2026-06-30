# Probing RoPE geometry and AV semantics in FLUX: readout works, activation steering doesn't

# RUNBOOK — FLUX RoPE / AV probe

Practical guide to set up and run every experiment. For *what they mean* and the
findings, see `WRITEUP.md`. For the conceptual overview, `README.md`.

---

## 0. TL;DR end-to-end

```bash
# 0. deps (once per pod)
pip install -r requirements.txt

# 1. smoke test (no GPU, no download) — always do this first
python selftest.py
python exp1_av_localization.py --run-dir runs/selftest
python exp4_intersection.py    --run-dir runs/selftest
python exp5_layer_timestep.py  --run-dir runs/selftest_sweep

# 2. one real capture + the observational experiments
python capture.py --save-v --outdir runs/dragon \
  --prompt "A dragon standing on a rock under a cloudy sky" \
  --concepts "dragon,rock,cloud,sky"
python exp1_av_localization.py --run-dir runs/dragon
python exp2_rope_frequency.py  --run-dir runs/dragon --spatial-only
python exp3_entanglement.py    --run-dir runs/dragon
python exp4_intersection.py    --run-dir runs/dragon --band 19-30

# 3. dataset sweep + aggregate (cross-image validity)
python run_dataset.py --run-exp1 --drop-stores
python run_dataset.py --only dragon,city_night,beach_person,fish_coral \
  --run-exp2 --run-exp3 --drop-stores
python aggregate.py --root runs --out runs/_aggregate

# 4. causal control experiments
python exp6_perturb.py        --run-dir runs/dragon --concept dragon --band 19-26 --alphas 0,4,8,16
python exp6b_concept_metric.py --run-dir runs/dragon --concept dragon --band 19-26 --alphas 0,4,8
```

---

## 1. Hardware & pod

- GPU: **≥32GB recommended** (RTX PRO 4500 / RTX 5090 / L40S). 24GB (4090) is
  tight for FLUX bf16 + cpu-offload and may OOM; if you must, lower resolution.
- Disk: the model is **~34GB**. Budget **≥50GB** wherever it caches. Each 512px
  capture store is ~1.5GB (≈2GB with `--save-v`).

## 2. Environment

```bash
pip install -r requirements.txt
```

Pinned `diffusers==0.31.0` because the capture hook relies on FLUX attention
calling `torch.nn.functional.scaled_dot_product_attention`. **If you bump
diffusers, re-run `selftest.py` and confirm the hook records 57 layers.**

FLUX.1-schnell is ungated (Apache-2.0) — no HF token. For FLUX.1-dev: accept its
license, `huggingface-cli login`, then add
`--model-id black-forest-labs/FLUX.1-dev --steps 50 --guidance 3.5 --max-seq 512`.

## 3. Smoke test (do this before paying for a capture)

```bash
python selftest.py        # builds runs/selftest (single-step+V) and runs/selftest_sweep (multi-step)
```

Then run the analysis scripts against the synthetic stores (numbers are random —
you're only checking they execute and produce figures):

```bash
python exp1_av_localization.py --run-dir runs/selftest
python exp2_rope_frequency.py  --run-dir runs/selftest
python exp3_entanglement.py    --run-dir runs/selftest
python exp4_intersection.py    --run-dir runs/selftest
python exp5_layer_timestep.py  --run-dir runs/selftest_sweep
```

(exp6/6b need the real model, not the synthetic store.)

## 4. `capture.py` — the instrumented generation

Generates one image and records per-layer tensors at the chosen step.

| flag | default | notes |
|------|---------|-------|
| `--prompt` / `--concepts` | dragon demo | concepts **must appear in the prompt** |
| `--height/--width` | 512 | image size; patch grid = size/16 |
| `--steps` | 4 | schnell=4, dev~50 |
| `--capture-step` | 2 | which denoising step to record |
| `--seed` | 0 | reproducibility |
| `--save-v` | off | **required for Exp 4** (recomputes AV from Q,K,V) |
| `--av-only` | off | store only AV (small) — for Exp 5 |
| `--all-steps` | off | record every step — **required for Exp 5** |
| `--cache-dir` | none | persistent model dir, e.g. `/workspace/hf` (see §9) |
| `--no-offload` | off | skip cpu-offload (needs ≥32GB free VRAM, faster) |
| `--outdir` | runs/run0 | output directory |

Outputs in `--outdir`: `generated.png`, `capture_store.pt`, `meta.json`.

## 5. Observational experiments (read a store; no GPU)

| script | needs in store | key flags | output |
|--------|----------------|-----------|--------|
| `exp1_av_localization.py` | AV | `--band 19-30`, `--topk`, `--concepts`, `--no-grids` | per-layer + `exp1_concepts.png`, `exp1_scores.json` |
| `exp2_rope_frequency.py` | Q,K | `--spatial-only`, `--layers 0,9,15,19,26,38,56`, `--detail-layer 19`, `--hi-frac` | `exp2_spread_vs_layer.png`, `exp2_layerL*_maps.png` |
| `exp3_entanglement.py` | Q,K | (none) | `exp3_entanglement.png`, `exp3_metrics.json` |
| `exp4_intersection.py` | Q,K,**V** | `--band 19-30`, `--spatial-only` | `exp4_*` (full vs high-band AV) |
| `exp5_layer_timestep.py` | AV, multi-step | `--concepts` | `exp5_heatmap.png` (layer × step) |

```bash
python exp1_av_localization.py --run-dir runs/dragon --band 19-30
python exp2_rope_frequency.py  --run-dir runs/dragon --spatial-only \
  --layers 0,5,9,12,15,18,19,22,26,30,38,45,56 --detail-layer 19
python exp3_entanglement.py    --run-dir runs/dragon
python exp4_intersection.py    --run-dir runs/dragon --band 19-30   # store must have --save-v
```

Exp 5 needs a special capture:
```bash
python capture.py --all-steps --av-only --outdir runs/sweep \
  --prompt "A dragon standing on a rock under a cloudy sky" --concepts "dragon,rock,cloud,sky"
python exp5_layer_timestep.py --run-dir runs/sweep
# finer timestep axis: add --model-id .../FLUX.1-dev --steps 50 --guidance 3.5 --max-seq 512
```

## 6. Dataset sweep + aggregation

`prompts.json` holds 16 prompt/concept entries (all concepts appear in their
prompt). `run_dataset.py` loads the model **once** and sweeps them.

```bash
# Exp1 gallery over all prompts (drop stores to save disk)
python run_dataset.py --run-exp1 --drop-stores

# Exp2+Exp3 on a few diverse scenes for cross-image validity
python run_dataset.py --only dragon,city_night,beach_person,fish_coral \
  --run-exp2 --run-exp3 --drop-stores \
  --exp2-extra "--spatial-only --layers 0,9,15,19,26,38,56 --detail-layer 19"

# collapse all runs into cross-image conclusions + the operating band
python aggregate.py --root runs --out runs/_aggregate
```

`run_dataset.py` flags: `--only a,b,c`, `--run-exp1/2/3`, `--exp{1,2,3}-extra "..."`,
`--drop-stores` (deletes each store after its experiments — keeps only one on
disk at a time). **Keep stores** (omit `--drop-stores`) only if you'll run more
experiments on them later.

`aggregate.py` prints `top8_layers_by_mean` (Exp1), `hi_lt_lo_hit_rate` (Exp2),
`lowest_severity_layers` (Exp3), and `recommended_range` (the fused operating
band) and writes `aggregate_*.png` + `aggregate_summary.json`.

## 7. Causal control experiments (load the model)

These perturb AV during generation. Use a run dir that **still has
`capture_store.pt`** (not one swept with `--drop-stores`).

`exp6_perturb.py` — pixel-space effect:
```bash
# inject (semantic band) / structural band ablation / suppression
python exp6_perturb.py --run-dir runs/dragon --concept dragon --band 19-26 --mode add     --alphas 0,4,8,16
python exp6_perturb.py --run-dir runs/dragon --concept dragon --band 15-18 --mode add     --alphas 0,4,8,16
python exp6_perturb.py --run-dir runs/dragon --concept dragon --band 19-26 --mode project --alphas 0,1,2,4
# controls
python exp6_perturb.py --run-dir runs/dragon --concept dragon --band 19-26 --mode add --alphas 0,4,8,16 --control-random
python exp6_perturb.py --run-dir runs/dragon --concept dragon --band 19-26 --mode add --alphas 0,4,8,16 --region 1,8,9,24   # off-object (sky box)
```
Flags: `--band a-b`, `--region r0,c0,r1,c1` (patch coords; default central third),
`--mode add|project`, `--alphas`, `--control-random`. Each run auto-names a
distinct folder (`exp6_<dir>_<mode>_<band>_<region>/`). Read `delta_vs_alpha.png`
(want inside ≫ outside, monotonic) and `summary.png`.

`exp6b_concept_metric.py` — concept-space effect (run concept + random, compare
the **downstream** curves):
```bash
python exp6b_concept_metric.py --run-dir runs/dragon --concept dragon --band 19-26 --alphas 0,4,8
python exp6b_concept_metric.py --run-dir runs/dragon --concept dragon --band 19-26 --alphas 0,4,8 --control-random
```
Concept ↑ downstream while random ≈ 0 ⇒ concept-specific. (Our result: both ≈ 0
downstream → not concept-specific; edit erased within ~1 layer.)

## 8. Recommended workflow for a fresh investigation

1. `selftest.py` + analysis on synthetic — confirm the code runs.
2. One `capture.py --save-v` on your prompt → exp1/2/3/4 to find the operating band.
3. `run_dataset.py` (exp1 all; exp2/3 a few) → `aggregate.py` → read
   `recommended_range`. That band drives everything downstream.
4. exp6 / exp6b at the recommended band, with the random + off-object controls.

## 9. Caching the model (avoid re-downloading 34GB)

The default HF cache is `/root/.cache` on the **ephemeral** container disk — wiped
on pod stop, so the model re-downloads. To persist, point at a volume **that has
≥40GB free quota**:

```bash
export HF_HUB_DISABLE_XET=1        # already set inside capture.py; avoids xet's 2x disk use
python capture.py --cache-dir /workspace/hf ...   # or export HF_HOME=/workspace/hf
```

Check the volume first: `df -h` and `du -sh /workspace`. If `/workspace` is a
quota'd network mount with no room (we hit this), just use the local container
disk (omit `--cache-dir`) and accept a re-download on restart.

## 10. Troubleshooting

| symptom | cause → fix |
|---------|-------------|
| `Disk quota exceeded (os error 122)` | cache hitting a full/quota'd disk → `export HF_HUB_DISABLE_XET=1`; cache on a disk with room (`--cache-dir`), or use local `/`. |
| `FileNotFoundError: .../capture_store.pt` | that run was swept with `--drop-stores`, or you pointed at the wrong dir → use a run dir that still has the store, or re-`capture.py`. |
| CUDA OOM on load | 24GB card + FLUX → use ≥32GB, keep cpu-offload (don't pass `--no-offload`), or lower `--height/--width`. |
| Exp4 prints "store has no V" | capture without `--save-v` → re-capture with `--save-v`. |
| Exp5 "only 1 timestep" | capture without `--all-steps` → re-capture with `--all-steps --av-only`. |
| hook records ≠ 57 layers | diffusers version changed the attention path → pin `diffusers==0.31.0` and re-run `selftest.py`. |
| concept "X" skipped | the word isn't a prompt token → put it in the prompt (simplified readout only sees prompt tokens). |

## 11. File map

```
capture.py            instrumented generation (capture + write-hook)
common.py             shared analysis helpers (readout, RoPE freqs, metrics)
exp1_av_localization  AV concept localization per layer + multi-concept figure
exp2_rope_frequency   high/low RoPE-band locality vs depth
exp3_entanglement     image↔text contamination + image self-structure
exp4_intersection     band-limited (high-freq) AV vs full AV
exp5_layer_timestep   layer × timestep localization heatmap
exp6_perturb          gradient-free AV perturbation (pixel-space)
exp6b_concept_metric  perturbation effect measured in concept space
run_dataset.py        sweep capture+experiments over prompts.json
aggregate.py          cross-run summary + operating band
prompts.json          16-entry prompt/concept dataset
selftest.py           CPU validation (synthetic stores)
WRITEUP.md            findings + takeaways + path forward
RUNBOOK.md            this file
```



*A research log. FLUX.1-schnell, 512px, single-image probes unless noted.*

## 0. One-paragraph summary

We set out to test whether the structure FLUX builds under its rotary positional
embeddings (RoPE) could be turned into a control signal for object position. We
found two clean, opposite results. **(Readout — confirmed.)** The attention
output (AV) linearly encodes *where* a concept is, sharply and at a predictable
mid-network depth, and RoPE genuinely organizes attention by spatial scale
(high-frequency channels = local, low-frequency = global). **(Control —
refuted, for this mechanism.)** A fixed AV "concept direction" is an excellent
*detector* but a useless *actuator*: injecting it is erased within ~1 layer, is
no more concept-specific than a random vector, and cannot insert a concept where
one isn't — even under sustained multi-layer injection. The honest conclusion is
*decode ≠ steer*, and the path to actual control is latent guidance, not
activation editing.

## 1. The original idea (and where the plot went)

The starting intuition, paraphrased: *RoPE's frequency structure encodes spatial
position at different scales; can I read that structure out and use it (e.g. via
its frequency decomposition) to guide where objects appear?*

That intuition bundled two separable claims:

1. **RoPE carries positional/scale structure that survives the network's
   entanglement, and AV exposes a clean semantic-spatial readout.** — This is
   **true**, and we measured it precisely (Exp 1–3).
2. **You can therefore *intervene* on that structure (band-limited attention, or
   injecting a concept direction into AV) to *control* object position.** — This
   is **false** for direct activation editing (Exp 4, 6, 6b).

So the plot didn't disappear; the project split the original idea into a part
that holds and a part that doesn't, and told us *why*. The position-control goal
is still reachable — but through a different actuator (Section 6).

## 2. Method / instrumentation

- **Model:** FLUX.1-schnell (MM-DiT): 19 double-stream blocks then 38
  single-stream blocks = 57 attention layers; 24 heads, head_dim 128; 2D RoPE
  with axes `[16, 56, 56]` (one text/temporal axis + H and W image axes).
- **Capture:** we monkeypatch `torch.nn.functional.scaled_dot_product_attention`,
  pass through unchanged (image identical), and record the post-RoPE Q/K, the
  attention output AV, and (optionally) V, filtering to FLUX-shaped calls. A call
  counter maps each call to (layer, denoising-step). The same hook was later made
  *write-capable* to perturb AV in place.
- **Readout (ConceptAttention-style):** for a concept word present in the prompt,
  take its text-token AV vector and cosine-project it onto every image patch's AV
  (in the concatenated-head "output space"). Sharpness is scored as the fraction
  of positive saliency mass in the top 10% of patches (0.10 = no localization).

## 3. Results

| Exp | Question | Result | Takeaway |
|----|----------|--------|----------|
| **1** | Does AV localize concepts, and at what depth? | Peak ~**L24–28** (score ~0.26 ≈ 2.6× uniform); elevated band **L19–32**; deep layers (38–56) ≈ floor. Across **55 image/concept series** the *peak location* is stable; only magnitude varies. | AV is a clean concept **readout**, concentrated in early single-stream blocks. Content-independent. |
| **2** | Does RoPE frequency = spatial scale (Untwisting-RoPE claim) in FLUX 2D RoPE? | High-frequency channels give **local** attention, low-frequency **global**. Holds in **95%** of (image, layer) pairs. High band most local at L15 and L26–30; low band flat (~12–14, never localizes). | RoPE **organizes attention by spatial scale**, robustly. The RoPE channels *are* the frequency basis — no literal FFT needed; band-grouping by ω is the spectral decomposition. |
| **3** | How entangled is the joint attention, and where does structure survive? | Image self-attention is cleanest (low text-contamination, low entropy, low spread) in **late double blocks L15–17**; contamination climbs through single blocks; deep layers are maximally entangled *and* high-variance. | Spatial structure **survives entanglement best mid-network**, then dissolves. |
| **agg** | Do 1–3 agree across images? | Combined operating band = **L15–30**. Late-double (15–18) = clean spatial canvas; single-entry (19–30) = peak concept localization; deep = dissolution. | A single, convergent mid-network window. The "hand-off": geometry first, semantics on top, then loss. |
| **4** | Is the signal sharper at the *intersection* of AV semantics and high-frequency RoPE geometry? | Band-limiting AV to high-frequency channels makes localization **worse** (≈ floor). Full AV is best; low-band is intermediate. | Semantics live in the **low-frequency/full** channels, geometry in the high. They are **separable axes, not an intersecting sharpener.** (Negative for the framing, consistent with Exp 2.) |
| **5** | Optimal layer × timestep? | Implemented as a 2D score heatmap (AV-only, all-steps capture). On schnell only 4 coarse timesteps; real timestep resolution needs FLUX.1-dev (50 steps). | Tooling in place; layer axis reproduces Exp 1. Timestep sweep pending a dev run. |
| **6** | Does perturbing AV causally and *specifically* control the image? (gradient-free) | Editing AV → **localized** pixel change (in/out ≈ 3.5, clean α=0). **But not concept-specific:** a *random* direction gave a *higher* in/out than the concept direction, and **off-object injection failed** (in/out < 1 — the effect leaked to the existing object). | Pixel in/out measures **disruption, not concept**. Spatial intervention is real; concept-specific control is not demonstrated. |
| **6b** | Measure in *concept space*, not pixels: does the edit move the readout, and does concept beat random? | At the edited band, inject = +0.15, suppress = −0.14, random ≈ 0 (metric valid). **Downstream of the band, all conditions → ~0**; concept ≈ random. | The edit is **erased within ~1 layer.** Fixed-direction AV injection has **no persistent, concept-specific causal effect.** |
| **6 (sustained)** | Does re-injecting at *every* layer (8–48) overcome the erasure? | Coherent localized change (in/out ≈ 5) — brute force beats one-shot erasure. **But** sustained *random* scored *higher* (in/out 6.8 > 5.0), and sustained *off-object* still failed (0.63). | Sustained injection **coherently modulates existing content**, but is still **not concept-specific** and **cannot insert.** Not usable control. |

## 4. What is confirmed

- **AV is a state-of-the-art-style readout.** It localizes concepts in output
  space far better than raw cross-attention, peaks at a predictable mid-network
  depth, and the depth is content-independent across 55 series.
- **RoPE organizes information by spatial scale.** High-frequency rotary
  channels enforce locality; low-frequency channels are position-insensitive /
  global. This is robust across images (95% hit-rate) and is the strongest direct
  support for the original "RoPE encodes position at scale" intuition.
- **Structure survives entanglement in a specific band (L15–30)**, with a clean
  hand-off from geometric self-structure (late double) to semantic localization
  (early single).

## 5. What is refuted (for this mechanism)

- **A fixed linear AV "concept direction" is not a control signal.** Three
  independent failures, all well-controlled:
  1. *Erasure* — a one-shot band edit vanishes within ~1 layer downstream (6b).
  2. *Non-specificity* — a random direction perturbs at least as much as the
     concept direction on every pixel metric, one-shot and sustained (6, sustained).
  3. *No insertion* — injecting a concept into a region that lacks it does
     nothing there; the effect routes through attention to where the concept
     already is (off-object, one-shot and sustained).
- **The likely cause:** each block's RMSNorm renormalizes and the next layer's
  attention re-mixes, so an off-manifold additive edit is absorbed and the
  representation is pulled back to its learned manifold. The model *self-corrects*
  the intervention. This is the classic **decode ≠ steer** gap, here quantified
  with an erasure depth (~1 layer) and a specificity test (concept ≈ random).

## 6. Moving forward: the goal is still reachable, via a different actuator

The position-control goal was never really an *activation-editing* problem; it's
a *guidance* problem. The right way to turn a good readout into control is
**latent guidance**: treat the AV concept-saliency as a differentiable
objective and take its gradient with respect to the latent `x_t` at each
denoising step, nudging the model's own noise prediction so the region's concept
readout goes up/down (or so an object's saliency centroid moves to a target
location). This stays **on-manifold** — the model denoises the guided latent
normally instead of fighting an off-manifold edit, which is exactly the failure
mode we kept hitting.

This is a known and validated framework — *Diffusion Self-Guidance* (Epstein et
al., 2023) controls object **position and size** using gradients of
cross-attention properties w.r.t. the latent. Our contribution would be to plug
in a **sharper readout** (the AV / ConceptAttention signal, which beats
cross-attention) and the **RoPE-scale awareness** (which layers/scales carry the
positional handle) into that loop. So the original "use RoPE/attention structure
to guide object position" idea lands here — as a guidance loss, not an
activation edit.

Concretely, three options:

1. **Build the latent-guidance prototype** (recommended). Reuse the existing hook
   to compute a region/centroid concept-saliency loss at the operating band
   (L15–30), enable gradients through one denoising step, and add a guidance term
   to the latent update. Smallest viable test: "increase dragon-saliency inside a
   target box / move its centroid," measured on the actual output. This is the
   real "gradient-based spatial control" the hypothesis aimed at.
2. **Write up the two-part result as-is** — a clean readout characterization plus
   a well-controlled negative on activation steering. That's a complete story.
3. **Both** — (2) now, (1) as the forward-looking section.

## 7. Honest limitations

- Most probes are single-image / single-seed / single-timestep; Exp 1 and the
  aggregate used a 16-prompt dataset (55 series), but Exp 2/3 cross-image checks
  were ~5 images. More seeds/timesteps would tighten everything.
- The readout is the *simplified* ConceptAttention variant (reuses prompt tokens);
  it cannot probe concepts absent from the prompt without the dedicated
  concept-token augmentation.
- Exp 5's timestep axis is coarse on schnell; a dev run is needed for the
  ConceptAttention-style ~mid-schedule check.
- The control conclusion is about *additive linear AV editing*. It does not rule
  out control via latent guidance (Section 6), a different intervention point
  (residual/hidden states, or the latent), or optimized (gradient-found) edits.

## 8. Reproduce

`capture.py` (instrumented generation) → `exp1_av_localization.py`,
`exp2_rope_frequency.py`, `exp3_entanglement.py`, `exp4_intersection.py`,
`exp5_layer_timestep.py`, `exp6_perturb.py`, `exp6b_concept_metric.py`;
`run_dataset.py` sweeps a prompt set (`prompts.json`); `aggregate.py` collapses
runs into cross-image conclusions. `selftest.py` validates the analysis path on
CPU without the model.

### References
- ConceptAttention — Helbling et al., arXiv:2502.04320 (AV output-space readout).
- Untwisting RoPE — arXiv:2602.05013 (RoPE frequency → spatial scale).
- Diffusion Self-Guidance — Epstein et al., 2023 (attention-gradient latent control).

(Co-written with Claude Opus 4.8)