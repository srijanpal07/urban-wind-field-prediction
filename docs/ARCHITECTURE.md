# ARCHITECTURE.md — Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION                              │
│                                                                  │
│  STL file ──→ src/data/geometry.py ──→ obstacle_mask [H,W] bool │
│   (Y-up)       triangle rasterization   Z→col, X_inv→row        │
│                (skimage.draw.polygon)   512×512 current default  │
│                    │                                             │
│                    ▼                                             │
│              src/data/lbm_solver.py                              │
│              D2Q9 LBM on GPU (PyTorch)                          │
│              4-side Zou-He BCs (auto by sign of ux_in/uy_in)   │
│              u_arr, v_arr [T, H, W]  — 500 snapshots/condition  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     SYNTHETIC DRONE SAMPLING                     │
│                                                                  │
│  src/data/drone_sampler.py                                       │
│  ├─ make_traverse_path(): A* left-to-right street traverse       │
│  │    clearance=2 cells, jitter=2% grid for training diversity  │
│  ├─ interpolate_path(): linspace waypoints → 2400 positions      │
│  ├─ sample_field(): bilinear interp at (x,y,t); NaN on solid   │
│  ├─ Polar noise: ±0.008 LBU speed, ±10° direction              │
│  └─ obs_to_grid(): vectorized bilinear scatter +                 │
│       scipy.ndimage.gaussian_filter  (σ=3 cells, ~1ms/2400obs) │
│       → obs_u[H,W], obs_v[H,W], confidence[H,W] ∈ [0,1]       │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐  ┌─────────────────────────────────────┐
│  PHASE 1: U-FNO         │  │  PHASE 2: FLOW MATCHING             │
│  src/models/ufno.py     │  │  src/models/flow_matching.py        │
│                         │  │                                     │
│  Input [B, 6, H, W]:   │  │  Input:                             │
│    geom_mask            │  │    x_t    [B, 2, T, H, W] noisy    │
│    obs_u, obs_v         │  │    s      [B,] ODE timestep         │
│    obs_confidence       │  │    obs    [B, 6, H, W] (same 6ch)  │
│    x_grid, y_grid       │  │    geo    [B, 2, H, W] mask+SDF     │
│                         │  │                                     │
│  Architecture:          │  │  Architecture:                      │
│    Lift Conv2d(6→48)    │  │    GeometryEncoder(2→8 ch CNN)      │
│    4× FNOBlock          │  │    SpatiotemporalUNet:              │
│      SpectralConv2d     │  │      Lift per-frame Conv2d          │
│      + pointwise        │  │      n_levels ResBlock2D+AttnBlk   │
│      + skip connection  │  │      Bottleneck: spatial+temporal   │
│    Heteroscedastic head │  │      Decoder: up+skip+ResBlock      │
│                         │  │    FiLM timestep conditioning       │
│  Output [B, 4, H, W]:  │  │    Attn only at ≤64×64 feature maps │
│    u_pred, v_pred       │  │                                     │
│    log_var_u, log_var_v │  │  Output [B, 2, T, H, W]:           │
│                         │  │    velocity field v_θ (flow target) │
│  ~7.4M params           │  │    ~14M params (hidden=32,4 levels) │
│  modes=max(20,grid//8)  │  │                                     │
└─────────────────────────┘  │  Inference (DPS ensemble):          │
                             │    N=20 samples, 20 ODE steps       │
┌─────────────────────────┐  │    Guided: x_{s+ds} = x_s + ds·v̂  │
│  TRAINING: U-FNO        │  │             − ρ·∇‖obs−H(x̂_1)‖²   │
│  src/training/train_ufno│  │    Post-process per sample:         │
│                         │  │      Leray projection (2D FFT ∇·u=0)│
│  Loss:                  │  │      Obstacle mask zero-out         │
│    NLL (Gaussian) +     │  └─────────────────────────────────────┘
│    λ=0.01 × |∇·u|²     │
│  AdamW lr=1e-3          │  ┌─────────────────────────────────────┐
│  CosineAnnealingLR      │  │  TRAINING: FLOW MATCHING            │
│  Grad clip 1.0          │  │  src/training/train_fm.py           │
│                         │  │                                     │
│  Dataset:               │  │  Loss (straight-line ODE):          │
│    2400 drone samples   │  │    x_t=(1-s)·x_noise + s·x_target  │
│    per training example │  │    L=E[‖v_θ(x_t,s,c)−(x_t-x_n)‖²]│
│    obs_window=30 snaps  │  │    x_noise = physics_prior(), not   │
│                         │  │    plain Gaussian (default on)      │
│                         │  │    Excludes solid cells             │
└─────────────────────────┘  │                                     │
                             │  AMP (float16) + grad checkpointing │
                             │  recommended for 512×512            │
                             │  batch=4, hidden=32 → 23.7 GB peak │
                             └─────────────────────────────────────┘
```

## LBM Solver Details

Scheme: D2Q9 (2D, 9 velocity directions)
Collision: BGK (single relaxation time τ)
- τ = 0.7  →  ν = (τ - 0.5)/3 = 0.0667 (kinematic viscosity in LB units)

Boundary conditions (4-side, auto-selected each step):
- Left  (col=0):   Zou-He inlet if ux_in > 0; zero-gradient outlet otherwise
- Right (col=W-1): Zou-He inlet if ux_in < 0; zero-gradient outlet otherwise
- Top   (row=0):   Zou-He inlet if uy_in > 0; zero-gradient outlet otherwise
- Bot   (row=H-1): Zou-He inlet if uy_in < 0; zero-gradient outlet otherwise
- ux_in == 0: both L/R sides use zero-gradient (no override)
- uy_in == 0: top/bottom remain periodic from torch.roll (correct for pure x-flow)
- Buildings: Bounce-back (no-slip)
- Corners: last-applied BC wins (y-BCs overwrite x-BCs at corners — harmless)

Transient mode (ON by default):
- Inlet speed varies sinusoidally: factor = 1 + A*(0.7*sin(2πt/T) + 0.3*sin(2πt/1.7T))
- Amplitude A=0.25, period T=40 steps, clamped to [0.2, 1.8]

## Geometry Pipeline

```
city_model.STL  (Y-up, X=10000mm, Z=11250mm footprint)
   │
   ├─ load_stl()           detect binary vs ASCII, parse triangles [N, 3, 3]
   │
   ├─ auto-detect up axis  argmin(spans) → Y (span 2435mm << X,Z spans)
   │
   ├─ slice at Y=730mm     30% of Y range → ~20m real altitude at 1:40 scale
   │
   ├─ axis mapping         h0=Z → columns (horizontal display axis)
   │                       h1=X → rows, inverted: X_max→row 0 (bottom)
   │
   ├─ triangle rasterize   skimage.draw.polygon (roof triangles → correct shapes)
   │                       skimage.draw.line    (wall edges → outlines)
   │                       Only skip triangles entirely below slice height
   │
   ├─ binary_fill_holes    fill enclosed building interiors
   │
   └─ strip 2-pixel border removes ground-plane edge artefacts
```

Output: bool array [H, W], True = solid building cell

## Drone Sampler Details

Path generation:
1. `make_traverse_path(margin=0.1, seed)`: left-to-right A* traverse
2. A* uses 8-connectivity with 2-cell building clearance
3. `interpolate_path(waypoints, n=2400)`: linspace positions along path

Sampling:
- `sample_field(u_arr, v_arr, x_path, y_path, t_indices)`: bilinear interp at
  each (x,y,t); returns NaN for solid cells
- Polar noise: perturb speed by ±noise_speed_std=0.008 LBU (~0.5 m/s) and
  direction by ±noise_angle_std=10° independently

Grid splatting `obs_to_grid(obs, grid_size, sigma=3.0)`:
- Vectorized bilinear scatter: 4 `np.add.at()` calls distribute each obs to
  its 4 surrounding cells with bilinear weights
- `scipy.ndimage.gaussian_filter(sigma=3.0)` applied to accumulated u, v, weight grids
- Weighted average at each cell: u_grid = u_blur / w_blur (where w_blur > 1e-6)
- Returns: u_grid [H,W], v_grid [H,W], w_norm [H,W] ∈ [0,1]
- Performance: ~1ms for 2400 samples (vs ~50ms for old Python loop)

## Phase 2 Flow Matching Architecture Details

### SinusoidalEmbedding
- Input: s ∈ [0,1] (ODE timestep), [B,]
- Sinusoidal frequencies (dim//2) × 2-layer MLP → [B, t_emb_dim]

### GeometryEncoder
- Input: [B, 2, H, W] — binary obstacle mask + signed distance field (SDF),
  from `src/data/geometry.build_geo_channels()`. SDF gives a smooth
  proximity-to-wall gradient instead of the mask's hard step discontinuity.
- Conv2d(2→16, GN, GELU) → Conv2d(16→8, GN, GELU)
- Output: [B, 8, H, W] geometry features injected at every U-Net level

### Physics-Informed Prior (`FlowMatchingModel.physics_prior`)
- Replaces `torch.randn_like` as the flow-matching source distribution
  `x_noise`, used identically in `flow_match_loss()` (training) and `sample()`
  (inference) — controlled by `use_physics_prior` (default `True`, saved in
  the checkpoint so train/sample never diverge).
- Computation: confidence-weighted mean of `obs_u, obs_v` over the drone
  observation window → uniform ambient (u, v) field → zeroed inside obstacles
  → Leray-projected to divergence-free (reuses the same projection used for
  post-processing) → broadcast over `T_out` → Gaussian noise added on top for
  ensemble diversity.
- Rationale: the network only needs to learn the obstacle-wake correction to
  an already-plausible ambient flow, rather than the entire field from a
  structureless noise field — without adding a PDE solve or new model.

### ResBlock2D
- Pre-norm: GroupNorm → GELU → Conv2d(3×3) → GroupNorm → FiLM scale → GELU → Conv2d(3×3) + skip
- FiLM scale: `1 + Linear(t_emb_dim→out_ch)` applied after norm2 (matches out_ch shape)
- Skip: 1×1 Conv if in_ch ≠ out_ch, else Identity

### SpatialAttnBlock
- GroupNorm → reshape [B,C,H,W]→[B,H*W,C] → MultiheadAttention → reshape → residual
- Only instantiated at levels where feature map ≤64×64 cells

### TemporalAttnBlock
- GroupNorm → reshape [B,C,T,H,W]→[B*H*W,T,C] → MultiheadAttention → reshape → residual
- Only at bottleneck (smallest spatial resolution)

### SpatiotemporalUNet
- `_per_frame(x, fn)`: folds T into batch dim [B,C,T,H,W]→[B*T,C,H,W], applies fn, unfolds
- Encoder: hidden doubles at each level (hidden, 2h, 4h, 8h); channels at 512×512 grid:
  Level 0: 64ch → 512×512 spatial (no attn at grid=512)
  Level 1: 128ch → 256×256 (no attn)
  Level 2: 256ch → 128×128 (no attn)
  Level 3: 512ch → 64×64 (attn ✓)
  Bottleneck: 1024ch → 32×32 (spatial attn + temporal attn ✓)
- Gradient checkpointing wraps each encoder/decoder level for memory savings

### Leray Projection
```python
U = rfft2(u); V = rfft2(v)
kx, ky = rfftfreq(W)*W, fftfreq(H)*H  # wavenumber grids
K2 = kx² + ky² (with K2[0,0]=1 to avoid div-by-zero)
div = kx*U + ky*V
U_proj = U - kx*div/K2
V_proj = V - ky*div/K2
u_div_free = irfft2(U_proj)
```

## Ensemble Calibration (`src/evaluation/calibration.py`)

- `spread_skill(ensemble, truth, fluid_mask)`: correlation between pixel-wise
  ensemble std (spread) and `|mean − truth|` (error). Positive correlation
  means spread is informative about where the model is wrong — required for
  spread to be usable as a risk signal in Phase 3 path planning.
- `coverage(ensemble, truth, fluid_mask, interval=0.9)`: fraction of
  fluid-cell locations where truth falls inside the ensemble's central
  90% quantile band. Want ≈0.90; ≪0.90 means overconfident (spread too
  narrow), ≫0.90 means underconfident (spread too wide).
- `reliability_curve(...)`: (nominal, empirical) coverage pairs across
  multiple interval widths, for a reliability diagram.
- Wired into `infer_fm.py`'s ensemble statistics print block — computed on
  every inference run, not a separate evaluation step.

## Physical Unit Conversion

```
lbm_to_ms = ref_speed / 0.08 = 5.0 / 0.08 = 62.5

physical_speed (m/s) = lbm_speed * 62.5
```

Physical timestep derivation:
```
dx_physical = domain_real / grid_size ≈ 425m / 512 ≈ 0.83 m/cell
dt_physical = dx_physical / lbm_to_ms ≈ 0.83 / 62.5 ≈ 0.013 s/step
dt_snapshot = collect_every × dt_physical = 3 × 0.013 ≈ 0.04 s/snapshot
```

With current settings (collect_every=3, steps=500):
- Total simulated time: 500 × 0.04s ≈ 20 seconds
- T_out=10 forecast: 10 × 0.04s ≈ 0.4 seconds of LBM dynamics
- The "5-minute forecast" refers to real drone flight time under the 15-minute
  atmospheric stability assumption — the LBM field is quasi-steady within this window

## Key Constants / Defaults

| Parameter       | Value   | Notes                                              |
|-----------------|---------|----------------------------------------------------|
| grid_size       | 512     | 512×512 cells (current training config)            |
| LBM tau         | 0.7     | Relaxation time → ν=0.0667 LBU                    |
| inlet_speed     | 0.08    | Reference LB speed = 5.0 m/s physical             |
| ref_speed       | 5.0     | m/s at LBM speed 0.08 (lbm_to_ms = 62.5)         |
| lbm_to_ms       | 62.5    | Fixed conversion constant                          |
| collect_every   | 3       | LBM steps between saved snapshots                 |
| n_steps         | 500     | Snapshots per condition (current run)              |
| warmup          | 2000    | LBM steps before collecting                        |
| obs_window      | 30      | LBM snapshots in drone observation window          |
| total_steps     | 2400    | Drone positions per training sample (10Hz × 240s) |
| T_out           | 10      | Forecast sequence length (flow matching)           |
| FM hidden       | 32      | U-Net base channels at 512² (batch=4, 23.7 GB)   |
| FM n_levels     | 4       | U-Net depth                                        |
| FM t_emb_dim    | 256     | Timestep embedding dimension                       |
| FM n_samples    | 20      | Ensemble size at inference                         |
| FM n_steps_ode  | 20      | ODE integration steps                              |
| FM rho          | 0.5     | DPS guidance strength                              |
| U-FNO modes     | 32@512² | max(20, grid_size//8)                              |
| U-FNO hidden    | 48      | Channel width                                      |
| splat_sigma     | 3.0     | Gaussian splat radius (cells)                      |
| noise_speed_std | 0.008   | Drone speed noise (LBU, ~0.5 m/s)                 |
| noise_angle_std | 10°     | Drone direction noise                              |
