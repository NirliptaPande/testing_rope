"""
capture.py
==========
Run ONE image through FLUX and capture, at a single denoising step, the
post-RoPE Q / K and the attention output (AV = softmax(QK^T/sqrt(d)) @ V)
for every transformer layer.

How the capture works (and why it's robust):
---------------------------------------------
diffusers' FLUX attention ultimately calls
`torch.nn.functional.scaled_dot_product_attention(q, k, v)`. We monkeypatch
that single function. On each call we:
  * pass straight through to the real SDPA (so the generated image is
    bit-identical to an un-instrumented run), and
  * if capture is enabled AND the tensor shape matches FLUX attention
    (num_heads == FLUX heads, head_dim == FLUX head_dim), record q, k, out.

The q,k handed to SDPA are already RMS-normed AND rotary-embedded -- i.e. the
exact operands of the model's attention -- which is precisely the "attention
logits under RoPE" readout we want. The text encoders (T5, CLIP) also call
SDPA, but they have different head_dim, so the shape filter excludes them.

FLUX runs N_double double-stream blocks then N_single single-stream blocks per
denoising step (one SDPA call each), so call_index // n_layers == step index.
We only store the step == --capture-step.

Output (in --outdir):
  capture_store.pt   list of {q,k,av,seq} per layer (fp16, on CPU) + meta
  generated.png      the image that was produced
  meta.json          human-readable run metadata
"""

import os
import json
import argparse

# Disable HF's "xet" downloader: it keeps a dedup chunk cache PLUS the extracted
# files (~2x disk) and can blow a volume quota. Plain HTTPS download = 1x.
# Must be set before huggingface_hub is imported (happens inside load_pipeline).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Global capture state + SDPA monkeypatch                                      #
# --------------------------------------------------------------------------- #
class _Cap:
    enabled = False
    target_step = 0
    n_layers = 57          # 19 double + 38 single for FLUX.1; set from model
    flux_heads = 24        # FLUX.1 num_attention_heads
    flux_head_dim = 128    # FLUX.1 attention_head_dim
    call_count = 0         # counts only FLUX-shaped SDPA calls
    store = []             # captured layers for the target step


CAP = _Cap()
_ORIG_SDPA = F.scaled_dot_product_attention


def _patched_sdpa(*args, **kwargs):
    out = _ORIG_SDPA(*args, **kwargs)
    if not CAP.enabled:
        return out
    try:
        query, key, value = args[0], args[1], args[2]
        if query.dim() == 4:
            _, h, _, d = query.shape
            if h == CAP.flux_heads and d == CAP.flux_head_dim:
                step = CAP.call_count // CAP.n_layers
                if step == CAP.target_step:
                    CAP.store.append(
                        {
                            "q": query.detach().to("cpu", torch.float16),
                            "k": key.detach().to("cpu", torch.float16),
                            "av": out.detach().to("cpu", torch.float16),
                            "seq": int(query.shape[2]),
                        }
                    )
                CAP.call_count += 1
    except Exception as e:  # never let instrumentation break generation
        print(f"[capture] warning: {e}")
    return out


F.scaled_dot_product_attention = _patched_sdpa


# --------------------------------------------------------------------------- #
# Model / generation                                                           #
# --------------------------------------------------------------------------- #
def load_pipeline(model_id, dtype, offload, cache_dir=None):
    from diffusers import FluxPipeline

    # cache_dir=None -> HF default (~/.cache, often ephemeral). Pass a persistent
    # path (e.g. /workspace/hf) so the ~34GB model survives pod restarts and is
    # not re-downloaded.
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
    if offload:
        # Keeps peak VRAM low enough for a 24GB card. Modules stream to GPU
        # only while active. Slower, but fits FLUX (12B) comfortably.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def t5_token_strings(pipe, prompt, max_seq):
    """Decoded T5 token strings aligned to the text portion of the joint seq."""
    tok = pipe.tokenizer_2  # T5 tokenizer (CLIP pooled vec is NOT in joint seq)
    enc = tok(
        prompt,
        padding="max_length",
        max_length=max_seq,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc.input_ids[0].tolist()
    return tok.convert_ids_to_tokens(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    ap.add_argument("--prompt", default="a cat sitting on a wooden table")
    ap.add_argument(
        "--concepts",
        default="cat,table",
        help="comma-separated concept words (must appear in the prompt)",
    )
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4, help="schnell=4, dev~50")
    ap.add_argument(
        "--capture-step",
        type=int,
        default=2,
        help="0-indexed denoising step to capture (schnell: 0..3)",
    )
    ap.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="schnell ignores guidance (0.0); dev uses ~3.5",
    )
    ap.add_argument("--max-seq", type=int, default=256, help="T5 tokens (schnell 256, dev 512)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-offload", action="store_true", help="skip cpu offload (needs ~30GB+)")
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="HF model download dir. Set to a PERSISTENT path (e.g. /workspace/hf) "
        "so FLUX (~34GB) is not re-downloaded on pod restart.",
    )
    ap.add_argument("--outdir", default="runs/run0")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dtype = torch.bfloat16

    print(f"[capture] loading {args.model_id} ...")
    pipe = load_pipeline(args.model_id, dtype, offload=not args.no_offload, cache_dir=args.cache_dir)

    tf = pipe.transformer
    n_double = len(tf.transformer_blocks)
    n_single = len(tf.single_transformer_blocks)
    CAP.n_layers = n_double + n_single
    CAP.flux_heads = tf.config.num_attention_heads
    CAP.flux_head_dim = tf.config.attention_head_dim
    print(
        f"[capture] {CAP.n_layers} layers "
        f"({n_double} double + {n_single} single), "
        f"heads={CAP.flux_heads}, head_dim={CAP.flux_head_dim}"
    )

    # patch grid: FLUX = 8x VAE downsample then 2x patchify => /16, row-major
    h_patches = args.height // 16
    w_patches = args.width // 16
    img_len = h_patches * w_patches

    # arm capture
    CAP.store = []
    CAP.call_count = 0
    CAP.target_step = args.capture_step
    CAP.enabled = True

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    print(f"[capture] generating (capturing step {args.capture_step}) ...")
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        max_sequence_length=args.max_seq,
        generator=gen,
    ).images[0]
    CAP.enabled = False

    if len(CAP.store) != CAP.n_layers:
        print(
            f"[capture] WARNING: captured {len(CAP.store)} layers, "
            f"expected {CAP.n_layers}. Check --capture-step < --steps."
        )

    seq = CAP.store[0]["seq"] if CAP.store else (args.max_seq + img_len)
    txt_len = seq - img_len

    meta = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "concepts": [c.strip() for c in args.concepts.split(",") if c.strip()],
        "height": args.height,
        "width": args.width,
        "h_patches": h_patches,
        "w_patches": w_patches,
        "img_len": img_len,
        "txt_len": txt_len,
        "seq": seq,
        "n_double": n_double,
        "n_single": n_single,
        "n_layers": CAP.n_layers,
        "heads": CAP.flux_heads,
        "head_dim": CAP.flux_head_dim,
        "capture_step": args.capture_step,
        "steps": args.steps,
        "seed": args.seed,
        "t5_tokens": t5_token_strings(pipe, args.prompt, args.max_seq),
        "axes_dims_rope": list(getattr(tf.config, "axes_dims_rope", [16, 56, 56])),
        "rope_theta": 10000.0,
    }

    img_path = os.path.join(args.outdir, "generated.png")
    store_path = os.path.join(args.outdir, "capture_store.pt")
    meta_path = os.path.join(args.outdir, "meta.json")

    image.save(img_path)
    torch.save({"layers": CAP.store, "meta": meta}, store_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[capture] saved:\n  {img_path}\n  {store_path}\n  {meta_path}")
    print(f"[capture] txt_len={txt_len}, img_len={img_len} ({h_patches}x{w_patches} patches)")


if __name__ == "__main__":
    main()