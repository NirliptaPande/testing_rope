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
    tokens = ["<pad>"] * txt_len
    tokens[1] = "▁cat"             # so concept 'cat' matches
    base_meta = {
        "model_id": "synthetic", "prompt": "a cat", "concepts": ["cat"],
        "height": 128, "width": 128, "h_patches": H, "w_patches": W,
        "img_len": img_len, "txt_len": txt_len, "seq": seq,
        "n_double": n_double, "n_single": n_single, "n_layers": n_layers,
        "heads": heads, "head_dim": head_dim,
        "capture_step": 1, "steps": 4, "seed": 0, "guidance": 0.0, "max_seq": txt_len,
        "t5_tokens": tokens, "axes_dims_rope": axes, "rope_theta": 10000.0,
        "save_v": True, "av_only": False,
    }

    def entry(layer, step, av_only=False):
        av = torch.randn(1, heads, seq, head_dim, generator=g).half()
        e = {"av": av, "seq": seq, "layer": layer, "step": step}
        if not av_only:
            e["q"] = torch.randn(1, heads, seq, head_dim, generator=g).half()
            e["k"] = torch.randn(1, heads, seq, head_dim, generator=g).half()
            e["v"] = torch.randn(1, heads, seq, head_dim, generator=g).half()
        return e

    # single-step store with V -> exercises exp1/2/3/4 (and exp6 dir derivation)
    layers = [entry(l, 1) for l in range(n_layers)]
    meta = dict(base_meta, captured_steps=[1], all_steps=False)
    torch.save({"layers": layers, "meta": meta}, os.path.join(run_dir, "capture_store.pt"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    plt.imsave(os.path.join(run_dir, "generated.png"),
               np.random.rand(meta["height"], meta["width"], 3))

    # multi-step AV-only store -> exercises exp5 layer x timestep sweep
    sweep_dir = run_dir + "_sweep"
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_layers = [entry(l, s, av_only=True) for s in (0, 1, 2) for l in range(n_layers)]
    sweep_meta = dict(base_meta, captured_steps=[0, 1, 2], all_steps=True, av_only=True, save_v=False)
    torch.save({"layers": sweep_layers, "meta": sweep_meta}, os.path.join(sweep_dir, "capture_store.pt"))
    with open(os.path.join(sweep_dir, "meta.json"), "w") as f:
        json.dump(sweep_meta, f, indent=2)
    print(f"[selftest] synthetic runs written to {run_dir}/ and {sweep_dir}/")


def test_common_math():
    import common as C
    omega = C.flux_channel_frequencies((4, 6, 6), 10000.0)
    assert omega.numel() == 16
    # high-freq channels (large omega) should be the per-axis early channels
    assert omega.max() == omega[0] == 1.0, "first channel should be omega=1 (highest freq)"
    _, meta = C.load_run("runs/selftest")
    hi, lo, om = C.frequency_band_masks(meta, 0.5)
    assert int(hi.sum()) + int(lo.sum()) == 16 and not bool((hi & lo).any())
    # spatial-only: drop the text axis (axes[0]=4) -> only 12 H/W channels used
    hi_s, lo_s, _ = C.frequency_band_masks(meta, 0.5, spatial_only=True)
    assert not bool(hi_s[:4].any()) and not bool(lo_s[:4].any()), "text axis not dropped"
    assert int(hi_s.sum()) + int(lo_s.sum()) == 12, "spatial channel count wrong"
    pos, matched = C.concept_positions("cat", meta)
    assert pos == [1], f"concept match failed: {pos} {matched}"
    # real (non-pad) text token count, as exp3 computes it
    n_real = sum(1 for tk in meta["t5_tokens"][: meta["txt_len"]] if tk != "<pad>")
    assert n_real == 1, f"n_real wrong: {n_real}"
    print("[selftest] common.py math OK (rope freqs, spatial bands, concept match, n_real)")


if __name__ == "__main__":
    test_capture_hook()
    make_synthetic_run()
    test_common_math()
    print("[selftest] ALL OK. Smoke-test the analysis scripts (no GPU needed):")
    print("  python exp1_av_localization.py --run-dir runs/selftest")
    print("  python exp2_rope_frequency.py  --run-dir runs/selftest")
    print("  python exp3_entanglement.py    --run-dir runs/selftest")
    print("  python exp4_intersection.py    --run-dir runs/selftest")
    print("  python exp5_layer_timestep.py  --run-dir runs/selftest_sweep")
    print("  (exp6 needs the model: python exp6_perturb.py --run-dir <real run> --concept cat)")