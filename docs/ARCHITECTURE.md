# ARCHITECTURE.md — Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION                              │
│                                                                  │
│  STL file ──→ geometry.py ──→ obstacle_mask [H, W] bool         │
│   (Y-up)       triangle         Z→col, X_inv→row                │
│                rasterization    256×256 default                  │
│                    │                                             │
│                    ▼                                             │
│              lbm_solver.py                                       │
│              D2Q9 LBM on GPU (PyTorch)                          │
│              Inlet side: left (ux≥0) or right (ux<0)            │
│              u_arr, v_arr [T, H, W]                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     SYNTHETIC SAMPLING                           │
│                                                                  │
│  drone_sampler.py                                                │
│  ├─ A* street-following path (avoids buildings + 3-cell buffer) │
│  ├─ Skips solid cells → returns NaN for building positions      │
│  ├─ Bilinear interpolation at drone (x,y,t)                     │
│  ├─ + Gaussian noise (noise_std=0.02 LB units)                  │
│  └─ Gaussian splat onto grid → obs_u[H,W], obs_v[H,W], conf[H,W]│
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     MODEL: WindFNO (model.py)                    │
│                                                                  │
│  Input [B, 6, H, W]:                                            │
│    ch0  geom_mask      (1=building, 0=fluid)                    │
│    ch1  obs_u          (drone-sampled u, splatted to grid)      │
│    ch2  obs_v          (drone-sampled v, splatted to grid)      │
│    ch3  obs_confidence (Gaussian weight, 0=unobserved)          │
│    ch4  x_grid         (normalized 0→1 coordinate)             │
│    ch5  y_grid         (normalized 0→1 coordinate)             │
│                                                                  │
│  Architecture:                                                   │
│    Lift (Conv2d 6→48)                                           │
│    ├─ FNO Layer 1  (SpectralConv2d + pointwise, modes=32@256²)  │
│    ├─ FNO Layer 2  ← save skip connection here                  │
│    ├─ FNO Layer 3                                               │
│    └─ FNO Layer 4                                               │
│    Cat([layer4_out, skip]) → Conv2d → GELU → Conv2d             │
│                                                                  │
│  Output [B, 4, H, W]:                                           │
│    ch0  u_pred     predicted u velocity                         │
│    ch1  v_pred     predicted v velocity                         │
│    ch2  log_var_u  log variance for u (heteroscedastic)         │
│    ch3  log_var_v  log variance for v                           │
│    sigma = exp(0.5 * log_var)  [derived, not raw output]        │
│                                                                  │
│  Modes: max(20, grid_size // 8)  — 32 at 256², 20 at 128²      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     TRAINING (train.py)                          │
│                                                                  │
│  Loss = NLL (Gaussian) + λ * divergence_residual                │
│                                                                  │
│  NLL per fluid cell:                                             │
│    L = log(σ) + 0.5 * ((y_true - y_pred) / σ)²                 │
│                                                                  │
│  Divergence regularizer (incompressible flow):                   │
│    L_div = (∂u/∂x + ∂v/∂y)²  λ=0.01                           │
│                                                                  │
│  Optimizer: AdamW, lr=1e-3, weight_decay=1e-4                   │
│  Scheduler: CosineAnnealingLR                                    │
│  Grad clip: 1.0                                                  │
│                                                                  │
│  Dataset construction:                                           │
│    For each sample i:                                            │
│      - Random start time t0                                      │
│      - A* street path over [t0, t0+obs_window]                  │
│      - Small random jitter on path for diversity                 │
│      - Target: dense field at t0 + obs_window + horizon         │
│  Checkpoint: saves model_state, grid_size, horizon, modes       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     VISUALIZATION (visualize.py)                 │
│                                                                  │
│  6-panel Matplotlib dashboard, dark theme (#0d1117 background)  │
│  All spatial panels: invert_xaxis() so Z_min appears on right   │
│                                                                  │
│  Top row:                                                        │
│    [0] LBM ground truth    — speed colormap + quiver arrows     │
│    [1] U-FNO prediction    — same colormap, same normalization  │
│    [2] Uncertainty σ       — YlOrRd, auto-scaled per frame      │
│                                                                  │
│  Bottom row:                                                     │
│    [3] Absolute error      — |GT speed - pred speed|, hot cmap  │
│    [4] Drone trajectory    — A* path + confidence heatmap       │
│        - Green dot = current drone                              │
│        - Orange diamonds = A* waypoints                         │
│        - Colored scatter = recent wind obs (colored by u)       │
│    [5] Live metrics        — rolling RMSE and MAE               │
│                                                                  │
│  Colorbars: FuncFormatter converts LBM → m/s using lbm_to_ms   │
│             Dark-theme tick labels (SUBTEXT color, size 6)      │
│  Buildings: 2× upsample + Gaussian blur + contourf (smooth)    │
│  Animation: FuncAnimation, 80ms interval, frame = timestep      │
│  Quiver U:  negated (-u) to correct for invert_xaxis() —        │
│             matplotlib quiver ignores axis inversion             │
└─────────────────────────────────────────────────────────────────┘
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

Transient mode (ON by default, disable with `--no-transient`):
- Inlet speed varies sinusoidally: factor = 1 + A*(0.7*sin(2πt/T) + 0.3*sin(2πt/1.7T))
- Amplitude A=0.25, period T=40 steps, clamped to [0.2, 1.8]

Wind direction rotation (`--angle-end`):
- Inlet angle linearly interpolates from `--angle` to `--angle-end` over the
  collection window. Warmup always runs at the starting angle.
- All 4-side BCs update each step based on current (ux_in, uy_in).

Stability: τ must be in (0.5, 2.0). Safe range: 0.6–0.9.
Inlet speed: keep below 0.15 in LB units to stay subsonic (Ma < 0.3).

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
- 256×256 default: ~27.5% solid (4510 cells for city_model.STL)

## Drone Sampler Details

Path generation:
1. `make_waypoints()`: random free-space targets (non-solid cells)
2. `_astar()`: A* on grid with obstacle inflation (3-cell clearance), 8-connectivity
3. `make_street_path()`: connects waypoints via A*, stores `_last_targets` for viz

Sampling: bilinear interpolation of u/v field at (x,y) position at time t
- Solid cells: returns NaN (not sampled)
- NaN observations: skipped in `obs_to_grid`

Grid splatting (obs_to_grid):
- Each valid observation splatted with Gaussian kernel (σ=3 grid cells)
- Weighted average at each grid cell
- Output: obs_u[H,W], obs_v[H,W], confidence[H,W] ∈ [0,1]

## Physical Unit Conversion

LBM speeds are dimensionless. Conversion to m/s:
```
lbm_to_ms = ref_speed_ms / inlet_speed_lbm
           = 10.0 / 0.08  = 125.0  (default)

physical_speed = lbm_speed * 125.0
```

With default settings:
- Free-stream (0.08 LBM) → 10 m/s
- Peak channel jets (~0.20 LBM) → ~25 m/s
- Colorbars apply FuncFormatter; underlying arrays always in LBM units

## Key Constants / Defaults

| Parameter      | Value  | Notes                                         |
|----------------|--------|-----------------------------------------------|
| grid_size      | 256    | 256×256 cells (128 also valid, faster)        |
| LBM tau        | 0.7    | Relaxation time                               |
| inlet_speed    | 0.08   | LB units                                      |
| inlet_angle    | 0.0°   | Right-to-left visually (col=0 inlet on right) |
| inlet_angle_end| None   | If set, angle rotates linearly to this value  |
| transient      | ON     | Gusty inlet by default; --no-transient to off |
| ref_speed      | 10.0   | m/s reference at inlet (colorbar conversion)  |
| n_warmup       | 400    | LBM steps before collecting                   |
| n_collect      | 150    | Timesteps in dataset                          |
| obs_window     | 20     | Timesteps of drone obs used as input          |
| horizon        | 10     | Timesteps ahead to predict                    |
| FNO modes      | 32     | Fourier modes at 256² (max(20, grid//8))      |
| FNO hidden     | 48     | Channel width                                 |
| FNO layers     | 4      | Number of FNO blocks                          |
| noise_std      | 0.02   | Drone measurement noise (LB units)            |
| splat_sigma    | 3.0    | Gaussian splat radius (cells)                 |
| batch_size     | 8      | Training batch                                |
| lr             | 1e-3   | AdamW learning rate                           |
| epochs         | 10     | Training epochs (default, increase for quality)|
