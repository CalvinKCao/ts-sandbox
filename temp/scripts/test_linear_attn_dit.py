"""Unit tests for linear-attn DiT (NaLa / DeltaFlow-P / MixFFN / rational x-attn)."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.diffusion_tsf.dit import (  # noqa: E402
    FactorizedDiT,
    MixFFN,
    NaLaAdaptiveDeltaFlowP,
    RationalKernelCrossAttention,
    _CrossAttention,
    _DiTBlock,
    _DiTCrossAttnBlock,
    _MLP,
    _SelfAttention,
    _gdn_chunked_scan,
    _gdn_fused_scan,
    _gdn_scan_sequential,
    _softmax_self_attn_layer,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_fused_scan_matches_sequential() -> None:
    torch.manual_seed(0)
    B, H, N, D = 2, 2, 8, 4
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    decay = torch.sigmoid(torch.randn(B, H, N))
    beta = torch.sigmoid(torch.randn(B, H, N))
    y_seq = _gdn_scan_sequential(q, k, v, decay, beta)
    y_fused = _gdn_fused_scan(q, k, v, decay, beta)
    y_chunk = _gdn_chunked_scan(q, k, v, decay, beta, chunk_size=8)
    max_fused = (y_seq - y_fused).abs().max().item()
    max_chunk = (y_seq - y_chunk).abs().max().item()
    _assert(y_seq.shape == y_fused.shape, f"shape {tuple(y_seq.shape)} vs {tuple(y_fused.shape)}")
    _assert(max_fused < 1e-4, f"fused vs sequential max abs {max_fused} >= 1e-4")
    _assert(max_chunk < 1e-4, f"chunked vs sequential max abs {max_chunk} >= 1e-4")


def test_flags_off_softmax_modules() -> None:
    dit = FactorizedDiT(
        in_channels=2,
        cond_channels=2,
        out_channels=2,
        image_height=16,
        patch_size=(8, 6),
        embed_dim=32,
        depth=8,
        num_heads=4,
        dropout=0.0,
        use_linear_attn=False,
        use_linear_cross_attn=False,
        use_attn_res=False,
    )
    _assert(dit.bottleneck_idx == 4, f"bottleneck {dit.bottleneck_idx}")
    for i, block in enumerate(dit.blocks):
        if i == dit.bottleneck_idx:
            _assert(isinstance(block, _DiTCrossAttnBlock), f"layer {i} not cross")
            _assert(isinstance(block.self_attn, _SelfAttention), f"layer {i} self")
            _assert(isinstance(block.cross_attn, _CrossAttention), f"layer {i} cross")
            _assert(isinstance(block.mlp, _MLP), f"layer {i} mlp")
        else:
            _assert(isinstance(block, _DiTBlock), f"layer {i} not DiTBlock")
            _assert(isinstance(block.attn, _SelfAttention), f"layer {i} attn")
            _assert(isinstance(block.mlp, _MLP), f"layer {i} mlp")


def test_flags_on_module_types_and_hybrid() -> None:
    depth = 8
    dit = FactorizedDiT(
        in_channels=2,
        cond_channels=2,
        out_channels=2,
        image_height=16,
        patch_size=(8, 6),
        embed_dim=32,
        depth=depth,
        num_heads=4,
        dropout=0.0,
        use_linear_attn=True,
        use_linear_cross_attn=True,
        use_attn_res=True,
    )
    expected_softmax = [3, 7]
    for i, block in enumerate(dit.blocks):
        want_softmax = _softmax_self_attn_layer(i, True)
        _assert(want_softmax == (i in expected_softmax), f"slot {i}")
        if i == dit.bottleneck_idx:
            _assert(isinstance(block, _DiTCrossAttnBlock), f"layer {i}")
            if want_softmax:
                _assert(isinstance(block.self_attn, _SelfAttention), f"bn self {i}")
                _assert(isinstance(block.cross_attn, _CrossAttention), f"bn cross {i}")
            else:
                _assert(isinstance(block.self_attn, NaLaAdaptiveDeltaFlowP), f"bn nala {i}")
                _assert(
                    isinstance(block.cross_attn, RationalKernelCrossAttention),
                    f"bn rational {i}",
                )
            _assert(isinstance(block.mlp, MixFFN), f"bn mix {i}")
        else:
            if want_softmax:
                _assert(isinstance(block.attn, _SelfAttention), f"soft {i}")
            else:
                _assert(isinstance(block.attn, NaLaAdaptiveDeltaFlowP), f"nala {i}")
            _assert(isinstance(block.mlp, MixFFN), f"mix {i}")
    _assert(dit.bottleneck_idx == 4, "depth=8 bottleneck is 4 (linear slot)")
    _assert(isinstance(dit.blocks[4].cross_attn, RationalKernelCrossAttention), "bn rational")


def test_forward_backward_grids_and_mask() -> None:
    torch.manual_seed(1)
    dit = FactorizedDiT(
        in_channels=2,
        cond_channels=2,
        out_channels=2,
        image_height=8,
        patch_size=(4, 4),
        cond_patch_size=(4, 4),
        embed_dim=32,
        depth=8,
        num_heads=4,
        dropout=0.0,
        context_dim=16,
        use_linear_attn=True,
        use_linear_cross_attn=True,
        use_attn_res=True,
        gradient_checkpointing=False,
    )
    B = 2
    # cond 8x16 → 2x4 = 8 tokens; x 8x4 → 2x1 = 2 tokens
    x = torch.randn(B, 2, 8, 4, requires_grad=True)
    cond = torch.randn(B, 2, 8, 16)
    t = torch.randint(0, 1000, (B,))
    ctx = torch.randn(B, 6, 16)
    mask = torch.zeros(B, 6, dtype=torch.bool)
    mask[:, 2:] = True
    out = dit(
        x, t, cond,
        encoder_hidden_states=ctx,
        ctx_key_padding_mask=mask,
    )
    _assert(out.shape == (B, 2, 8, 4), f"out {tuple(out.shape)}")
    out.sum().backward()
    _assert(x.grad is not None and torch.isfinite(x.grad).all(), "x.grad")
    nala_grad = False
    for p in dit.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            nala_grad = True
            break
    _assert(nala_grad, "expected some finite grads")


def test_yaml_flags_reach_model() -> None:
    from models.diffusion_tsf.pipeline.config import load_experiment_config
    from models.diffusion_tsf.pipeline.state import PipelineState

    path = os.path.join(
        REPO,
        "configs/binary_window_norm_patch_refine_canvas128_p32x6_weather_allv_linattn_fixedhp.yaml",
    )
    cfg = load_experiment_config(path)
    st = PipelineState.from_config(cfg)
    _assert(st.use_linear_attn is True, "use_linear_attn")
    _assert(st.use_linear_cross_attn is True, "use_linear_cross_attn")
    _assert(st.use_attn_res is True, "use_attn_res")
    _assert(st.mmpd_patch_size == 12, f"P={st.mmpd_patch_size}")
    _assert(st.guidance_type == "patch_decoder", st.guidance_type)
    _assert(float(st.channel_dropout_drop_frac) == 0.0, "no channel mask")
    dit = FactorizedDiT(
        in_channels=2,
        cond_channels=2,
        out_channels=2,
        image_height=32,
        patch_size=tuple(st.dit_patch_size),
        cond_patch_size=tuple(st.dit_cond_patch_size),
        embed_dim=32,
        depth=int(st.dit_depth),
        num_heads=4,
        use_linear_attn=st.use_linear_attn,
        use_linear_cross_attn=st.use_linear_cross_attn,
        use_attn_res=st.use_attn_res,
    )
    _assert(isinstance(dit.blocks[0].attn, NaLaAdaptiveDeltaFlowP), "layer0 nala")
    _assert(isinstance(dit.blocks[3].attn, _SelfAttention), "layer3 softmax")
    _assert(isinstance(dit.blocks[4].cross_attn, RationalKernelCrossAttention), "rational")
    _assert(dit.blocks[4].use_linear_self, "bn linear self")
    names = [p["phase"] for p in cfg["phases"]]
    _assert("staged_diffusion_pretrain" in names, names)
    pre = next(p for p in cfg["phases"] if p["phase"] == "staged_diffusion_pretrain")
    _assert(pre.get("phase1_config_name") == "binary_dual_scale_staged", "phase1 name")


def test_rational_kernel_fail_fast() -> None:
    attn = RationalKernelCrossAttention(32, 4)
    x = torch.randn(2, 5, 32)
    ctx = torch.randn(2, 7, 32)
    try:
        attn(x, ctx, return_attn_weights=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected fail on return_attn_weights")
    bias = torch.ones(2, 7)
    try:
        attn(x, ctx, attn_bias=bias)
    except ValueError:
        pass
    else:
        raise AssertionError("expected fail on nonzero attn_bias")
    out = attn(x, ctx)
    _assert(out.shape == x.shape, tuple(out.shape))
    _assert(abs(float(attn.scale_factor.detach()) - 0.01) < 1e-8, "scale_factor init")


def main() -> int:
    tests = [
        test_fused_scan_matches_sequential,
        test_flags_off_softmax_modules,
        test_flags_on_module_types_and_hybrid,
        test_forward_backward_grids_and_mask,
        test_yaml_flags_reach_model,
        test_rational_kernel_fail_fast,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            failed.append(fn.__name__)
    if failed:
        print("FAILED:", *failed)
        return 1
    print("all linear-attn unit tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
