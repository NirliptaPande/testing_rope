# Probing RoPE geometry and AV semantics in FLUX: readout works, activation steering doesn't

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

(Written by Claude Opus 4.8)