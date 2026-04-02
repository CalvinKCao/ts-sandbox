# Guided diffusion pipelines (Mermaid)

Notation: **N** = `num_variables`, **A** = `num_aux_channels` (coord ± ramp ± sine ± value), **H** = `image_height`, **W** = width on the time axis (future crop or full L+F canvas), **K** = `lookback_overlap`, **c₀…c₃** = `unet_channels` default `[64,128,256,512]`, **T_emb** = timestep embedding dim **256**. Downsampling uses **stride-2 conv on both height and width** at each U-Net level.

---

## 1. Pixel-space guided diffusion (`DiffusionTSF` + `ConditionalUNet2D`)

Default training path uses `conditioning_mode=visual_concat`, `use_hybrid_condition=True`, and optional `use_guidance_channel=True` (iTransformer ghost images).

```mermaid
flowchart TB
  subgraph S1["Stage 1 — iTransformer (frozen or trained separately)"]
    P1D["Past 1D (B,N,L)"]
    FC["Coarse future (B,N,H_or_L)"]
    P1D --> FC
  end

  subgraph PRE["Pixel 2D encoding (occupancy + blur)"]
    E2D["TimeSeriesTo2D: (B,N,L+H_img) → (B,N,H,W_future)"]
    BLUR["VerticalGaussianBlur → scaled to approx [-1,1]"]
    E2D --> BLUR
  end

  subgraph CH["Channel stack for noisy canvas x (before U-Net)"]
    direction TB
    NV["N channels: noisy future occupancy per variable"]
    AUX["A channels: vertical coord, optional time ramp, time sine, optional value strip"]
    GV["N channels (if guidance): Stage-1 forecast as ghost 2D<br/>overlap K cols from true past, rest from iTrans"]
    NV --> CAT1
    AUX --> CAT1
    GV --> CAT1
    CAT1["concat dim=1 → backbone_in = N + A + (N if guidance)"]
  end

  FC -.->|"encode_to_2d same as data"| GV
  BLUR --> NV

  subgraph HYB["Hybrid 1D context (optional cross-attention)"]
    CTXIN["past_norm slice → (B,L_past,2): value + normalized time index"]
    CTXENC["TimeSeriesContextEncoder → (B,L_past,128)"]
    CTXIN --> CTXENC
  end

  subgraph UNET["ConditionalUNet2D — noise predictor ε̂"]
    direction TB

    subgraph VC["visual_concat conditioning path"]
      VPAST["Past 2D cropped/interp to W: (B,N,H,W)<br/>+ optional value ch → visual_cond_channels = N or N+1"]
    end

    INIT["init_conv: concat(x, cond) → (B,c₀,H,W)<br/>in_ch + visual_cond = (N+A+N_guide) + (N or N+1)"]

    subgraph DOWN["Down path (3 DownBlocks: 64→128, 128→256, 256→512)"]
      D0["Block i=0: Res×2 (64→128 ch) — no attn if attention_levels=[1,2] (default)"]
      DS0["stride-2 conv → H/2 × W/2 @ 128 ch"]
      D1["Block i=1: Res×2 (128→256) + SpatialTransformer if i∈attention_levels"]
      DS1["stride-2 → H/4 × W/4 @ 256 ch"]
      D2["Block i=2: Res×2 (256→512) + SpatialTransformer if i∈attention_levels"]
      DS2["stride-2 → H/8 × W/8 @ 512 ch"]
      D0 --> DS0 --> D1 --> DS1 --> D2 --> DS2
    end

    MID["MiddleBlock or DilatedMiddleBlock @ c₃=512<br/>H/8 × W/8 — optional cross-attn to ctx"]

    subgraph UP["Up path (mirrored)"]
      U0["UpBlock: transposed conv ×2, concat skip, Res×2 → c₂"]
      U1["→ c₁ @ H/4 × W/4"]
      U2["→ c₀ @ H/2 × W/2"]
      U3["→ c₀ @ H × W"]
      U0 --> U1 --> U2 --> U3
    end

    FIN["GroupNorm + SiLU + final_conv (B,c₀,H,W) → (B,N,H,W)"]

    INIT --> D0
    DS2 --> MID --> U0
    U3 --> FIN
  end

  subgraph TIME["Timestep conditioning"]
    TE["t ∈ 0..T-1 → sinusoidal emb (dim 256) → MLP → t_emb injected in every ResBlock"]
  end

  CAT1 --> INIT
  VPAST --> INIT
  CTXENC -.->|"encoder_hidden_states"| D1
  CTXENC -.->|"encoder_hidden_states"| D2
  CTXENC -.->|"encoder_hidden_states"| MID
  TE -.->|"t_emb in each ResBlock + middle"| D0

  FIN --> OUT["Predicted noise ε̂ matching noisy channels (B,N,H,W)"]
```

**Shapes (example: N=7, A=3 with coord+ramp+sine, guidance on, H=128, W=200):**

| Tensor | Shape |
|--------|--------|
| Noisy + aux + guidance (x) | `(B, 7+3+7, 128, 200)` = `(B, 17, 128, 200)` |
| Visual cond (past stripe) | `(B, 7, 128, 200)` |
| After `init_conv` concat | `(B, 17+7, 128, 200)` → **`(B, 64, 128, 200)`** |
| Deepest features | `(B, 512, 16, 25)` |
| Output ε̂ | `(B, 7, 128, 200)` |

**Attention / hybrid:** `attention_levels` lists **indices i** of `DownBlock` / `UpBlock` in order (`i ∈ {0,1,2}` for three blocks). `use_attn = (i in attention_levels)`. Default `[1,2]` turns on **SpatialTransformerBlock** (self-attn + cross-attn to 1D context) only on **i=1 and i=2** (the **128→256** and **256→512** stages, before their stride-2), not on the first down block. Symmetric levels on the up path.

**CFG (training/inference):** Classifier-free guidance doubles the conditional path: null cond = zeros for visual concat; null context / null guidance when `cfg_scale > 1`.

---

## 2. Latent-space guided diffusion (`LatentDiffusionTSF` + frozen `TimeSeriesVAE`)

Diffusion runs on **VAE latents** with **4×** spatial compression: `H_lat = H/4`, `W_lat = W_pixel/4`, `C_z = latent_channels` (default **4**). Guidance ghosts are encoded in pixel space, then **the same VAE encoder** produces `cz`-channel latent guidance.

```mermaid
flowchart TB
  subgraph VAE["Frozen TimeSeriesVAE (not trained in LDM step)"]
    direction TB
    PIX["Pixel sheet (B,1,H,W) per variate occupancy"]
    E1["enc: stride-2 → stride-2 + mid conv"]
    Z0["μ, logσ (B,C_z,H/4,W/4) → z_scaled = z / scale_factor"]
    PIX --> E1 --> Z0
    DEC["decode: conv + 2× upsample → (B,1,H,W) tanh"]
  end

  subgraph GUIDE["Guidance in latent space"]
    G2D["iTrans forecast → encode_to_2d → ghost (B,1,H,W_future)"]
    GZ["encode_to_scaled_latent(ghost) → (B,C_z,H_lat,W_future_lat)"]
    G2D --> GZ
  end

  subgraph CHZ["Latent canvas channels (unified axis example)"]
    direction TB
    ZN["C_z: noisy latent future"]
    AUXZ["A: coord / time (injected on latent grid, same flags as pixel)"]
    GZC["C_z: guidance latent (if use_guidance_channel)"]
    ZN --> CZ
    AUXZ --> CZ
    GZC --> CZ
    CZ["in_ch = C_z + A + (C_z if guidance)<br/>e.g. 4+1+4 = 9 with coord + guidance"]
  end

  subgraph CONDZ["Visual concat cond (past latent)"]
    PZ["z_past padded/cropped to W_lat → (B,C_z,H_lat,W_lat)<br/>+ optional value ch → cond ch = C_z or C_z+1"]
  end

  subgraph UNETZ["Same ConditionalUNet2D, smaller grid"]
    direction TB
    IZ["init_conv: (in_ch + cond_ch) → c₀<br/>image_height = H_lat (e.g. 32 if H=128)"]
    DZ["DownBlocks: spatial halving each stage<br/>(H_lat,W_lat) → … → (H_lat/8, W_lat/8)"]
    MZ["Middle @ c₃"]
    UZ["UpBlocks + skips"]
    FZ["final_conv → (B,C_z,H_lat,W_lat) noise pred"]
    IZ --> DZ --> MZ --> UZ --> FZ
  end

  subgraph HYBZ["Hybrid context (same as pixel)"]
    CTXZ["TimeSeriesContextEncoder: (B,L_past,2) → (B,L_past,128)"]
  end

  Z0 -.->|"encode past & future sheets"| ZN
  GZ --> GZC
  CZ --> IZ
  PZ --> IZ
  CTXZ -.->|"cross-attn"| DZ
  CTXZ -.-> MZ

  FZ --> SCHED["Training: noise / x0 loss on latents; K_lat = K/4 overlap cols"]
```

At inference, DDIM produces `z` then `decode_from_scaled_latent` → pixels (same `DEC` as in the VAE subgraph).

**Shapes (univariate, H=128 → H_lat=32, C_z=4, coord only A=1, guidance on, W_lat = W_pixel/4):**

| Stage | Shape |
|-------|--------|
| Pixel ghost / data | `(B, 1, 128, W_px)` |
| After VAE encode | `(B, 4, 32, W_px/4)` |
| Noisy + coord + guidance | `(B, 4+1+4, 32, W_lat)` = `(B, 9, 32, W_lat)` |
| Past cond | `(B, 4, 32, W_lat)` |
| After `init_conv` | `(B, 9+4, 32, W_lat)` → **`(B, 64, 32, W_lat)`** |
| ε̂ output | `(B, 4, 32, W_lat)` |

**CI multivariate (ETTh1 7-var):** the same per-variate shapes apply; runs use a **shared** univariate VAE + U-Net per variate (batch over variates), so channel counts stay **C_z**-based, not `N×C_z` in a single forward.

**Code refs:** `DiffusionTSF` / `ConditionalUNet2D` — `diffusion_model.py`, `unet.py`; latent — `latent_diffusion_model.py`, `vae.py`, `config.py` (`LatentDiffusionConfig.latent_image_height`, `latent_spatial_downsample=4`).
