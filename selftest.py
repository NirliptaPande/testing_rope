"""
selftest.py
===========
CPU-only sanity check that runs WITHOUT downloading FLUX. It:
  (1) exercises the SDPA capture monkeypatch with dummy tensors -- confirms it
      records FLUX-shaped calls from the target step only and ignores
      text-encoder-shaped calls;
  (2) fabricates a synthetic capture_store.pt (small, correct shapes) so the
      three experiment scripts can be run end-to-end to verify they produce
      figures.

Run:  python selftest.py
Then: python exp1_av_localization.py --run-dir runs/selftest   (etc.)
"""

import os
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_capture_hook():
    import capture  # patches F.scaled_dot_product_attention on import

    cap = capture.CAP
    cap.n_layers = 10
    cap.flux_heads = 2
    cap.flux_head_dim = 16
    cap.target_step = 1
    cap.call_count = 0
    cap.store = []
    cap.enabled = True

    seq = 80
    # text-encoder-shaped call (different head_dim) -> must be IGNORED
    F.scaled_dot_product_attention(
        torch.randn(1, 8, seq, 64), torch.randn(1, 8, seq, 64), torch.randn(1, 8, seq, 64)
    )
    # two full "steps" of FLUX-shaped calls
    for _ in range(2 * cap.n_layers):
        q = torch.randn(1, cap.flux_heads, seq, cap.flux_head_dim)
        F.scaled_dot_product_attention(q, q.clone(), q.clone())
    cap.enabled = False

    assert len(cap.store) == cap.n_layers, f"got {len(cap.store)} (expected {cap.n_layers})"
    assert cap.store[0]["q"].dtype == torch.float16
    print(f"[selftest] capture hook OK: stored {len(cap.store)} layers from target step only")


def make_synthetic_run(run_dir="runs/selftest"):
    os.makedirs(run_dir, exist_ok=True)
    H = W = 8                      # patch grid
    img_len = H * W               # 64
    txt_len = 16
    seq = txt_len + img_len        # 80
    n_double, n_single = 4, 6
    n_layers = n_double + n_single
    heads, head_dim = 2, 16        # head_dim must == sum(axes_dims_rope)
    axes = [4, 6, 6]

    g = torch.Generator().manual_seed(0)
    layers = []
    for _ in range(n_layers):
        q = torch.randn(1, heads, seq, head_dim, generator=g).half()
        k = torch.randn(1, heads, seq, head_dim, generator=g).half()
        av = torch.randn(1, heads, seq, head_dim, generator=g).half()
        layers.append({"q": q, "k": k, "av": av, "seq": seq})

    tokens = ["<pad>"] * txt_len
    tokens[1] = "▁cat"             # so concept 'cat' matches
    meta = {
        "model_id": "synthetic", "prompt": "a cat", "concepts": ["cat"],
        "height": 128, "width": 128, "h_patches": H, "w_patches": W,
        "img_len": img_len, "txt_len": txt_len, "seq": seq,
        "n_double": n_double, "n_single": n_single, "n_layers": n_layers,
        "heads": heads, "head_dim": head_dim,
        "capture_step": 1, "steps": 4, "seed": 0,
        "t5_tokens": tokens, "axes_dims_rope": axes, "rope_theta": 10000.0,
    }
    torch.save({"layers": layers, "meta": meta}, os.path.join(run_dir, "capture_store.pt"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # dummy base image
    plt.imsave(os.path.join(run_dir, "generated.png"),
               np.random.rand(meta["height"], meta["width"], 3))
    print(f"[selftest] synthetic run written to {run_dir}/")


def test_common_math():
    import common as C
    omega = C.flux_channel_frequencies((4, 6, 6), 10000.0)
    assert omega.numel() == 16
    # high-freq channels (large omega) should be the per-axis early channels
    assert omega.max() == omega[0] == 1.0, "first channel should be omega=1 (highest freq)"
    _, meta = C.load_run("runs/selftest")
    hi, lo, om = C.frequency_band_masks(meta, 0.5)
    assert int(hi.sum()) + int(lo.sum()) == 16 and not bool((hi & lo).any())
    pos, matched = C.concept_positions("cat", meta)
    assert pos == [1], f"concept match failed: {pos} {matched}"
    print("[selftest] common.py math OK (rope freqs, band masks, concept match)")


if __name__ == "__main__":
    test_capture_hook()
    make_synthetic_run()
    test_common_math()
    print("[selftest] ALL OK. Now run the three exp_*.py scripts with "
          "--run-dir runs/selftest")