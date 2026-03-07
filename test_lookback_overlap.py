"""
Visualize that the lookback overlap (K) is correctly injected at every stage:
  1. Dataset slicing: last K of past == first K of target
  2. Guidance construction: overlap region uses observed values, future uses iTransformer
  3. Forward pass: weighted loss (0.3x overlap, 1.0x forecast)
  4. Generate: output is H timesteps (K trimmed)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models.diffusion_tsf.config import DiffusionTSFConfig
from models.diffusion_tsf.diffusion_model import DiffusionTSF
from models.diffusion_tsf.train_7var_pipeline import (
    TimeSeriesDataset, LOOKBACK_LENGTH, FORECAST_LENGTH, LOOKBACK_OVERLAP,
    PAST_LOSS_WEIGHT, IMAGE_HEIGHT, create_diffusion_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K = LOOKBACK_OVERLAP
H = FORECAST_LENGTH
L = LOOKBACK_LENGTH
print(f"Config: L={L}, H={H}, K={K}  →  target length = K+H = {K+H}")

# ── 1. Dataset slicing verification ──────────────────────────────────────────

n_vars = 3
n_rows = L + H + 200
rng = np.random.RandomState(42)
raw = rng.randn(n_rows, n_vars).astype(np.float32)

ds = TimeSeriesDataset(raw, lookback=L, horizon=H, stride=L, lookback_overlap=K)
past, future = ds[0]
print(f"\n[1] Dataset shapes — past: {tuple(past.shape)}, future: {tuple(future.shape)}")
assert future.shape[-1] == K + H, f"Expected target width {K+H}, got {future.shape[-1]}"

# The overlap region should be identical in both tensors
past_tail = past[:, -K:]       # last K of lookback
future_head = future[:, :K]    # first K of target
max_diff = (past_tail - future_head).abs().max().item()
print(f"  max |past[-K:] - future[:K]| = {max_diff:.2e}  (should be 0)")
assert max_diff < 1e-6, "Overlap region mismatch!"

fig, axes = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw={"height_ratios": [1, 1]})

var_idx = 0
t_past = np.arange(L)
t_target = np.arange(L - K, L + H)

ax = axes[0]
ax.plot(t_past, past[var_idx].numpy(), color="steelblue", lw=0.8, label="past (L)")
ax.plot(t_target, future[var_idx].numpy(), color="tomato", lw=0.8, label="target (K+H)")
ax.axvspan(L - K, L, alpha=0.25, color="gold", label=f"overlap (K={K})")
ax.axvline(L, color="grey", ls="--", lw=0.6)
ax.set_title("1 — Dataset slicing: overlap region is shared", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(L - 60, L + 60)

ax = axes[1]
ax.plot(past_tail[var_idx].numpy(), label="past[-K:]", lw=2, color="steelblue")
ax.plot(future_head[var_idx].numpy(), label="future[:K]", lw=2, ls="--", color="tomato")
ax.set_title("  Close-up: last K of past vs first K of target (identical)", fontsize=10)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("overlap_1_dataset.png", dpi=120)
print("  Saved overlap_1_dataset.png")
plt.close()

# ── 2. Guidance construction ─────────────────────────────────────────────────
# Build a minimal model and inspect _generate_guidance_2d output

n_vars_model = 3
config = DiffusionTSFConfig(
    num_variables=n_vars_model,
    lookback_length=L,
    forecast_length=K + H,
    lookback_overlap=K,
    past_loss_weight=PAST_LOSS_WEIGHT,
    image_height=IMAGE_HEIGHT,
    representation_mode="cdf",
    use_coordinate_channel=True,
    use_guidance_channel=True,
    use_hybrid_condition=False,
    unet_channels=[32, 64],
    attention_levels=[1],
    num_res_blocks=1,
    num_diffusion_steps=20,
    ddim_steps=5,
)
model = DiffusionTSF(config).to(device).eval()

batch_past = past[:n_vars_model].unsqueeze(0).to(device)
batch_future = future[:n_vars_model].unsqueeze(0).to(device)

# Manually call the guidance path
past_norm, future_norm, stats = model._normalize_sequence(batch_past, batch_future)
guidance_2d = model._generate_guidance_2d(batch_past, past_norm, stats, K + H)
print(f"\n[2] Guidance 2D shape: {tuple(guidance_2d.shape)}  (B, C, H_img, K+H)")
assert guidance_2d.shape[-1] == K + H

# Decode the guidance back to 1D for visual comparison
guidance_1d = model.decode_from_2d(guidance_2d)
guidance_denorm = model._denormalize(guidance_1d, stats)
print(f"  Guidance 1D shape: {tuple(guidance_denorm.shape)}")

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
t_target_plot = np.arange(K + H)
var_idx = 0

ax.plot(t_target_plot, batch_future[0, var_idx].cpu().numpy(),
        color="grey", lw=1.5, alpha=0.7, label="ground truth (K+H)")
ax.plot(t_target_plot, guidance_denorm[0, var_idx].cpu().numpy(),
        color="darkgreen", lw=1.5, label="guidance signal (overlap=observed, future=LinReg)")
ax.axvspan(0, K, alpha=0.25, color="gold", label=f"overlap zone (K={K})")
ax.axvline(K, color="grey", ls="--", lw=0.6)
ax.set_title("2 — Guidance: overlap uses observed past, future uses Stage-1 predictor", fontsize=11)
ax.legend(fontsize=8)
ax.set_xlabel("timestep within target window")

plt.tight_layout()
plt.savefig("overlap_2_guidance.png", dpi=120)
print("  Saved overlap_2_guidance.png")
plt.close()

# ── 3. Forward pass: weighted loss ───────────────────────────────────────────
model.train()
out = model.forward(batch_past, batch_future)

noise_pred = out["noise_pred"]
# Recompute the noise to show the split
_, future_norm_fwd, _ = model._normalize_sequence(batch_past, batch_future)
future_2d = model.encode_to_2d(future_norm_fwd)
t_steps = out["t"]
_, noise_actual = model.scheduler.add_noise(future_2d, t_steps)

# Per-column MSE
with torch.no_grad():
    col_mse = ((noise_pred - noise_actual) ** 2).mean(dim=(0, 1, 2))  # (K+H,)
    col_mse_np = col_mse.cpu().numpy()

weights = np.ones(K + H)
weights[:K] = PAST_LOSS_WEIGHT

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.bar(range(K + H), col_mse_np, width=1.0, color="lightcoral", alpha=0.6, label="raw column MSE")
ax.bar(range(K + H), col_mse_np * weights, width=1.0, color="steelblue", alpha=0.6, label="weighted column MSE")
ax.axvspan(0, K, alpha=0.15, color="gold")
ax.axvline(K, color="grey", ls="--", lw=0.6)
ax.set_xlabel("column index (0..K-1 = overlap, K..K+H-1 = forecast)")
ax.set_ylabel("noise MSE")
ax.set_title(f"3 — Loss weighting: overlap columns scaled by {PAST_LOSS_WEIGHT}×", fontsize=11)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("overlap_3_loss.png", dpi=120)
print(f"\n[3] Forward loss: {out['loss'].item():.4f}")
print("  Saved overlap_3_loss.png")
plt.close()

# ── 4. Generate: trimming verification ───────────────────────────────────────
model.eval()
with torch.no_grad():
    gen_out = model.generate(batch_past, use_ddim=True, num_ddim_steps=5)

pred = gen_out["prediction"]
print(f"\n[4] Generate output shape: {tuple(pred.shape)}")
assert pred.shape[-1] == H, f"Expected H={H} after trim, got {pred.shape[-1]}"
print(f"  ✓ Output is {H} timesteps (K={K} trimmed)")

# Also verify future_2d still has full K+H width (pre-trim internal)
assert gen_out["future_2d"].shape[-1] == K + H, "Internal 2D should be K+H wide"
print(f"  ✓ Internal future_2d is {K+H} wide (pre-trim)")

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
var_idx = 0

# Ground truth: only the H-step forecast part
gt_forecast = batch_future[0, var_idx, K:].cpu().numpy()
pred_forecast = pred[0, var_idx].cpu().numpy()

t_fc = np.arange(H)
ax.plot(t_fc, gt_forecast, color="grey", lw=1.5, alpha=0.7, label="ground truth (H steps)")
ax.plot(t_fc, pred_forecast, color="crimson", lw=1.5, label="prediction (K trimmed)")
ax.set_title(f"4 — Generate output: {H} steps returned, overlap discarded", fontsize=11)
ax.set_xlabel("forecast timestep")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("overlap_4_generate.png", dpi=120)
print("  Saved overlap_4_generate.png")
plt.close()

# ── Summary figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# Panel 1: dataset slicing
ax = axes[0]
t_past_all = np.arange(L)
t_tgt_all = np.arange(L - K, L + H)
ax.plot(t_past_all, past[0].numpy(), color="steelblue", lw=0.8, label="past")
ax.plot(t_tgt_all, future[0].numpy(), color="tomato", lw=0.8, label="target")
ax.axvspan(L - K, L, alpha=0.25, color="gold")
ax.axvline(L, color="grey", ls="--", lw=0.6)
ax.set_xlim(L - 50, L + 50)
ax.set_title("Dataset: overlap shared between past tail and target head")
ax.legend(fontsize=8, loc="upper left")

# Panel 2: guidance
ax = axes[1]
t_g = np.arange(K + H)
ax.plot(t_g, batch_future[0, 0].cpu().numpy(), color="grey", lw=1.2, alpha=0.6, label="target")
ax.plot(t_g, guidance_denorm[0, 0].cpu().numpy(), color="darkgreen", lw=1.2, label="guidance")
ax.axvspan(0, K, alpha=0.25, color="gold")
ax.axvline(K, color="grey", ls="--", lw=0.6)
ax.set_title("Guidance: observed past in overlap, predictor in forecast")
ax.legend(fontsize=8)

# Panel 3: loss weighting
ax = axes[2]
ax.bar(range(K + H), col_mse_np, width=1.0, color="lightcoral", alpha=0.5, label="raw MSE")
ax.bar(range(K + H), col_mse_np * weights, width=1.0, color="steelblue", alpha=0.5, label="weighted")
ax.axvspan(0, K, alpha=0.15, color="gold")
ax.axvline(K, color="grey", ls="--", lw=0.6)
ax.set_title(f"Loss: overlap weighted {PAST_LOSS_WEIGHT}×, forecast 1.0×")
ax.legend(fontsize=8)

# Panel 4: generate output
ax = axes[3]
ax.plot(np.arange(H), gt_forecast, color="grey", lw=1.2, alpha=0.6, label="ground truth")
ax.plot(np.arange(H), pred_forecast, color="crimson", lw=1.2, label="prediction (trimmed)")
ax.set_title(f"Generate: returns {H} steps, K={K} overlap discarded")
ax.legend(fontsize=8)
ax.set_xlabel("timestep")

for ax in axes:
    gold_patch = mpatches.Patch(color="gold", alpha=0.35, label=f"overlap (K={K})")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [gold_patch], fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("overlap_summary.png", dpi=150)
print("\nSaved overlap_summary.png")
plt.close()

print("\n── All checks passed ──")
