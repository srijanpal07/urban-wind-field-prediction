# CLAUDE.md — Project Memory for Claude Code
# Urban Wind Field Prediction via Drone Sampling
# Read this first. Then read docs/ARCHITECTURE.md and docs/ROADMAP.md.


## What This Project Is

A machine learning pipeline that predicts urban wind fields from sparse drone
measurements. A drone flies through a city at a fixed altitude (20m real-world,
~0.5m in 1:40 scaled model), sampling wind velocity along its path. A neural
network reconstructs and forecasts the full 2D wind field on that horizontal
slice — telling us what the wind looks like everywhere, not just where the drone
flew.

This is fundamentally a **data assimilation + short-horizon forecasting** problem
conditioned on known urban geometry.

## Physical Setup

- City model: physical scaled model at 1:40 scale (~10m × 11.25m footprint = ~400m × 450m real)
- STL file: `data/city_model.STL` — Y-up coordinate system, X=10000mm, Z=11250mm footprint
- Drone altitude slice: horizontal plane sliced at 30% of Y height (≈730mm model scale = ~29m real)
- Wind measurements: u (Z-direction) and v (X-direction) at drone position + timestamp
- No real drone yet: wind samples are synthetically drawn from the solver with Gaussian noise
- No real CFD yet: 2D LBM solver generates physics-consistent ground truth

## The Prediction Task

```
Inputs  → geometry mask (buildings as obstacles)
        → sparse drone wind samples over last ~4 min (observation window)
        → drone trajectory (x, y, t) positions

Output (Phase 1 U-FNO):
        → dense 2D wind field u(x,y), v(x,y) on the horizontal slice
        → uncertainty σ(x,y) per grid cell
        → single snapshot prediction at t+horizon

Output (Phase 2 Flow Matching):
        → N=20 ensemble samples of future wind field sequence u(x,y,t), v(x,y,t)
        → T_out=10 future frames per sample
        → ensemble spread = uncertainty map
        → exceedance probability for risk-aware path planning
```

Timing contract:
  t=0 to t=4min   drone samples wind along A→B leg (2400 samples at 10 Hz)
  t=4min          model runs inference (~seconds on GPU)
  t=5min          drone arrives at B; prediction window begins
  t=5 to t=10min  predicted wind field is used for planning next leg

## Current Status

- **Phase 1 (U-FNO)**: ✅ Complete. Archived to `outputs/ufno/`
- **Phase 2 (Flow Matching)**: 🔄 In progress. Model implemented; training running.

## Repository Structure

```
scripts/
  generate_data.py    ← Step 1: run LBM for 128 train + 16 test conditions
  train_ufno.py       ← Step 2a: train U-FNO (Phase 1 baseline)
  train_fm.py         ← Step 2b: train FlowMatchingModel (Phase 2)
  infer_ufno.py       ← Step 3a: U-FNO inference at random/specified condition
  infer_fm.py         ← Step 3b: Flow-matching ensemble inference
  evaluate_ufno.py    ← Step 4: benchmark U-FNO on random or held-out conditions
  run_pipeline.py     ← Legacy: single-condition end-to-end convenience script

src/
  data/
    lbm_solver.py     ← 2D LBM wind field simulator (D2Q9, GPU-native PyTorch)
    geometry.py       ← STL → 2D obstacle mask + SDF (build_geo_channels)
    drone_sampler.py  ← A* traverse path + vectorized obs_to_grid
  models/
    ufno.py           ← U-FNO neural network (~7.4M params), heteroscedastic
    flow_matching.py  ← FlowMatchingModel: spatiotemporal U-Net + DPS + Leray + physics prior
  training/
    train_ufno.py     ← NLL + divergence loss training loop for U-FNO
    train_fm.py       ← Flow-matching training loop + WindSequenceDataset
  evaluation/
    calibration.py    ← Ensemble calibration: spread-skill correlation, interval coverage
  viz/
    visualize.py      ← 6-panel interactive dashboard with m/s colorbars

data/
  city_model.STL      ← Real 1:40 scale city model (Y-up)
  obstacle_mask.npy   ← Cached rasterized mask
  cache/              ← Per-condition LBM cache (lbm_{mode}_a{angle}_s{speed}.npz)
  lbm_test.npz        ← Held-out test dataset (16 unseen conditions)

outputs/
  ufno/               ← Archived Phase 1 checkpoints + visualizations
    wind_fno.pth      ← Best U-FNO checkpoint (modes + grid_size in dict)
    eval_results.json ← Phase 1 benchmark results
    wind_fno_history.png, wind_dashboard.gif, etc.
  flow_matching/      ← Phase 2 outputs (populated during training)
    fm_model.pth      ← Best flow-matching checkpoint
    fm_model_history.png
```

## Multi-Condition Training Philosophy

The model input is always **6 channels** (geometry + sparse drone obs + coords).
Angle and speed are NOT passed as inputs — the model learns to infer the full
wind field from the drone observations alone. This is the physically meaningful
setup: in real deployment the inlet conditions are unknown; only the drone
readings are available.

## Data Generation

`generate_data.py` generates two datasets in one pass:

| Dataset | Angles | Speeds | Mode | File |
|---------|--------|--------|------|------|
| Training | 32 × every 11.25° → [0, 11.25, …, 348.75] | [0.02, 0.04, 0.08, 0.10] | transient | `data/cache/lbm_transient_*.npz` |
| Test | 8 × midpoints → [5.625, 50.625, …, 320.625] | [0.03, 0.06] | steady-state | `data/lbm_test.npz` |

Current run: 500 snapshots/condition (`--steps 500 --warmup 2000`), 512×512 grid.

## Drone Sampling: Realistic 2400-Sample Protocol

The drone samples at **2400 positions per training example** — matching real-world
10 Hz GPS/IMU logging over a 4-minute leg. This is set via `total_steps = 2400`
in `src/training/train_ufno.py` and `src/training/train_fm.py`.

`obs_to_grid()` in `drone_sampler.py` uses vectorized bilinear scatter +
`scipy.ndimage.gaussian_filter` (O(N + H×W), ~1ms for 2400 samples) instead of
the old per-observation Python loop (O(N × patch²), would be ~50ms).

## Physical Speed Scale

LBM speed 0.08 is the reference and maps to `--ref-speed` m/s (default 5.0).
Other LBM speeds scale proportionally: 0.02→1.25, 0.04→2.5, 0.08→5.0, 0.10→6.25 m/s.
`lbm_to_ms = ref_speed / 0.08 = 62.5` is used as a **fixed constant** throughout.

Physical timestep (important for interpreting T_out):
  dx_physical ≈ 425m / 512 cells ≈ 0.83 m/cell  (real-world city scale)
  dt_physical = dx_physical / lbm_to_ms ≈ 0.013 s/LBM step
  dt_snapshot = collect_every × dt_physical = 3 × 0.013 ≈ 0.04 s/snapshot
  → T_out=10 frames ≈ 0.4 s of simulated wind evolution

The "5-minute forecast" in the timing contract refers to real drone flight time,
not LBM simulation time. Under the 15-minute atmospheric stability assumption,
the LBM wind field is quasi-steady — the ensemble captures spatial uncertainty,
not rapid temporal dynamics.

## Phase 1: U-FNO (Baseline, Archived)

### Model (src/models/ufno.py)
- Input: [B, 6, H, W] (geom_mask, obs_u, obs_v, confidence, x_grid, y_grid)
- Output: [B, 4, H, W] (u_pred, v_pred, log_var_u, log_var_v)
- Architecture: Lift → 4× FNOBlock (SpectralConv2d + skip) → heteroscedastic head
- Modes: `max(20, grid_size // 8)` — 32 at 256², stored in checkpoint
- Parameters: ~7.4M

### Training (scripts/train_ufno.py)
- Loss: NLL (Gaussian) + λ=0.01 × divergence residual
- Optimizer: AdamW lr=1e-3, weight_decay=1e-4, CosineAnnealingLR
- Checkpoint saves: model_state, modes, grid_size, horizon

### Phase 1 Benchmark Results
Held-out test set: 16 conditions (8 angles × 2 speeds, all unseen during training).
Grid: 512×512, 128 training conditions, 200 epochs, batch=32.

| Run | Epochs | Vec RMSE (m/s) | Speed MAE (m/s) | Dir Error |
|-----|--------|----------------|-----------------|-----------|
| Run 1 (64 cond, 80 samples) | 50 | 3.39 | 2.20 | 72.3° |
| Run 2 (128 cond, 80 samples) | 200 | **2.75** | **1.87** | **43.4°** |

Phase 1 ceiling: val loss plateaued ~epoch 150. Known bottleneck was only 80
drone observations per sample (8-second traverse). Raised to 2400 (4-minute leg)
for all subsequent training.

## Phase 2: Flow Matching (Current)

### Model (src/models/flow_matching.py)
- **FlowMatchingModel**: GeometryEncoder + SpatiotemporalUNet
- **GeometryEncoder**: Conv2d(2→16→8) — input is binary mask + SDF
  (`src/data/geometry.build_geo_channels`), produces 8-channel geometry features
- **SpatiotemporalUNet**: 5D U-Net operating on [B, C, T, H, W]
  - Lift: per-frame Conv2d(16→hidden)
  - Encoder: n_levels ResBlock2D + SpatialAttnBlock + stride-2 down
  - Bottleneck: ResBlock2D + SpatialAttnBlock + TemporalAttnBlock
  - Decoder: ConvTranspose2d + skip concat + ResBlock2D + SpatialAttnBlock
  - Output: per-frame Conv2d(hidden→2)
- **Attention memory**: Capped at ≤64×64 feature maps via resolution-aware
  `attn_start = max(0, ceil(log2(grid_size / 64)))` — avoids O(H²W²) OOM
- **ResBlock2D**: FiLM-style timestep conditioning (scale applied after norm2)
- **Physics-informed prior** (`FlowMatchingModel.physics_prior`): the
  flow-matching source distribution is a confidence-weighted ambient (u,v)
  field, obstacle-zeroed and Leray-projected to divergence-free, plus Gaussian
  noise — not plain `torch.randn_like`. Controlled by `use_physics_prior`
  (default `True`); must match between training and `sample()` or the ODE
  integration starts from the wrong distribution.
- Parameters: ~14M at hidden=32, ~31M at hidden=48

### Training (scripts/train_fm.py / src/training/train_fm.py)
- Loss: flow-matching MSE on velocity field (straight-line interpolation)
  `x_t = (1-s)·x_noise + s·x_target`, `L = E[‖v_θ(x_t,s,c) − (x_target−x_noise)‖²]`
- Excludes solid cells from loss (obstacle mask)
- **AMP (float16)**: enabled by default, ~2× memory saving
- **Gradient checkpointing**: enabled by default, ~4× activation memory saving
- **Physics-informed prior**: enabled by default (`--no-physics-prior` to disable)
- Recommended: `--hidden 32 --batch 4` for 512×512 (23.7 GB peak VRAM)
- Checkpoint saves: model_state, T_out, grid_size, hidden, n_levels, t_emb_dim,
  use_physics_prior, history

### Inference (scripts/infer_fm.py)
- DPS-guided ensemble: N=20 samples, 20 ODE steps, rho=0.5
- Reads `use_physics_prior` from the checkpoint so the sampling source
  distribution matches training (override with `--no-physics-prior` only for
  ablation against a model actually trained without it)
- Post-processing per sample: Leray projection (2D FFT ∇·u=0) + obstacle mask zeroing
- Outputs: ensemble mean, spread (σ), per-member RMSE vs ground truth,
  spread-error correlation, 90% interval coverage (`src/evaluation/calibration.py`)

### Recommended Workflow (Phase 2)
```bash
# 1. Generate data (500 snapshots, 512×512 — ~2.5 hours on RTX 5000 Ada):
python scripts/generate_data.py --stl data/city_model.STL --grid 512 --steps 500 --warmup 2000

# 2. Train flow-matching model (200 epochs, ~2.8 days on RTX 5000 Ada):
python scripts/train_fm.py --epochs 200 --batch 4 --T-out 10 --hidden 32 --n-levels 4

# 2b. Resume training:
python scripts/train_fm.py --resume --epochs 400

# 3. Ensemble inference:
python scripts/infer_fm.py --stl data/city_model.STL
python scripts/infer_fm.py --stl data/city_model.STL --angle 45 --speed 0.08 --n-samples 20

# 4. U-FNO baseline (still works, checkpoint in outputs/ufno/wind_fno.pth):
python scripts/infer_ufno.py --stl data/city_model.STL --model outputs/ufno/wind_fno.pth
python scripts/evaluate_ufno.py --stl data/city_model.STL --test-data data/lbm_test.npz \
  --model outputs/ufno/wind_fno.pth
```

### Key Defaults (Phase 2)
| Script        | Argument         | Default                             | Notes                              |
|---------------|------------------|-------------------------------------|------------------------------------|
| train_fm.py   | `--epochs`       | 200                                 | Total epochs                       |
| train_fm.py   | `--batch`        | 4                                   | Recommended for 512×512 (23.7 GB) |
| train_fm.py   | `--T-out`        | 20 (default), use 10 for 512²      | Forecast sequence length           |
| train_fm.py   | `--hidden`       | 64 (default), use 32 for 512²      | U-Net base channels                |
| train_fm.py   | `--n-levels`     | 4                                   | U-Net depth                        |
| train_fm.py   | `--obs-window`   | 30                                  | LBM snapshots in obs window        |
| train_fm.py   | `--no-amp`       | False                               | Disable mixed precision            |
| train_fm.py   | `--no-checkpoint`| False                               | Disable grad checkpointing         |
| infer_fm.py   | `--n-samples`    | 20                                  | Ensemble size                      |
| infer_fm.py   | `--n-steps`      | 20                                  | ODE integration steps              |
| infer_fm.py   | `--rho`          | 0.5                                 | DPS guidance strength              |

## Hardware

- GPU: NVIDIA RTX 5000 Ada Generation, 32GB VRAM
- CUDA: 12.8, Driver: 570.211.01
- Use --device cuda (default). Falls back to CPU automatically if CUDA unavailable.
- Conda env: `urban-wind` (Python 3.11, PyTorch 2.11.0+cu128)

## What Claude Code Should Know

1. **Coordinate system**: The STL is Y-up (auto-detected). In the 2D mask:
   - Columns = Z axis (Z_min at col 0 = right side of display)
   - Rows = X axis inverted (X_max at row 0 = bottom of display)
   - `invert_xaxis()` is applied to all visualization panels so col 0 appears on
     the right. Wind with angle=0° (ux_in>0) flows right→left visually.

2. **LBM inlet side**: All 4 sides handled, auto-selected by sign of ux_in/uy_in.
   - ux_in > 0: inlet at LEFT col (col=0), outlet at RIGHT (col=W-1)
   - ux_in < 0: inlet at RIGHT col (col=W-1), outlet at LEFT (col=0)
   - uy_in > 0: inlet at TOP row (row=0, bottom of display), outlet at BOTTOM
   - uy_in < 0: inlet at BOTTOM row (row=H-1), outlet at TOP (row=0)
   - Multiple sides active simultaneously for diagonal angles (e.g. 45°)

3. **Geometry rasterization**: Uses `skimage.draw.polygon` + `skimage.draw.line`
   per triangle. Do NOT revert to bounding boxes (produces rectangular artifacts).

4. **Obstacle mask**: True = solid building. Drone sampler skips solid cells
   (returns NaN). LBM applies bounce-back on solid nodes.

5. **U-FNO checkpoint**: Saves `modes`, `grid_size`, `horizon` in the dict.
   Always use `ckpt.get('modes', 20)` when loading — do not hardcode modes=20.

6. **FM checkpoint**: Saves `T_out`, `grid_size`, `hidden`, `n_levels`, `t_emb_dim`,
   `use_physics_prior`. Always pass `grid_size` to `FlowMatchingModel()` so
   attn_start is computed correctly for the resolution. Always read
   `use_physics_prior` from the checkpoint at inference — sampling must use the
   same source distribution the model was trained with.

7. **m/s colorbars**: `lbm_to_ms = ref_speed / 0.08 = 62.5` (fixed constant).
   Applied via `FuncFormatter` — underlying data always stays in LBM units.

8. **LBM cache** (generate_data.py): Keyed as
   `lbm_{mode_str}_a{angle:07.3f}_s{speed:.4f}.npz` with embedded `mask_hash`.
   Training mode_str = `transient`, test mode_str = `steady`.
   Cache is wiped by default on each run (use `--keep-cache` to skip).

9. **Quiver arrow direction**: The U component is negated (`-u`) in `set_UVC()`
   calls in `visualize.py`. Compensates for `invert_xaxis()`.

10. **Drone path**: Left-to-right traverse via `make_traverse_path(margin=0.1)`
    in `drone_sampler.py`. Uses A* with clearance=2. Training adds 2% Gaussian
    jitter per sample for diversity.

11. **Noise model**: Polar noise — perturb speed (±0.008 LBU ≈ ±0.5 m/s) and
    direction (±10°) independently. More realistic than isotropic Cartesian noise.

12. **Grid size**: Always derived from data shape — never hardcode. FM model
    must receive `grid_size` at construction time to compute `attn_start`.

13. **obs_to_grid**: Now vectorized (bilinear scatter + scipy Gaussian filter).
    Signature: `obs_to_grid(obs, grid_size, sigma=3.0)` → (u_grid, v_grid, w_norm).

14. **Spatial attention memory**: The FM model only applies SpatialAttnBlock at
    levels where feature map ≤64×64. Formula:
    `attn_start = max(0, ceil(log2(grid_size / 64)))`. At grid=512: attn_start=3
    (attention only at 64×64 and 32×32 bottleneck). Do NOT change this — applying
    attention at 128×128 uses 1.1 GB/head and will OOM.

15. **AMP + grad checkpointing**: Required for training at 512×512. Peak VRAM:
    - hidden=32, batch=4, T_out=10, AMP+ckpt → 23.7 GB ✓
    - hidden=48, batch=2, T_out=10, AMP+ckpt → 18.7 GB ✓
    - hidden=48, batch=4, T_out=10, AMP+ckpt → OOM ✗

16. **U-FNO baseline preserved**: `outputs/ufno/wind_fno.pth` contains the best
    Phase 1 checkpoint. Use `scripts/infer_ufno.py --model outputs/ufno/wind_fno.pth`
    to reproduce Phase 1 results. Do not delete this checkpoint.

17. **Phase 3 goal**: Trajectory optimizer + active sensing (risk-aware path planning
    using Phase 2 ensemble output).
    **Phase 4 goal**: Replace LBM with OpenFOAM CFD.
    See docs/ROADMAP.md for full details.

18. **Geo conditioning is 2 channels, not 1**: `build_geo_channels()` in
    `geometry.py` returns `[mask, sdf]` — `GeometryEncoder` and
    `FlowMatchingModel` take `geo_in_channels=2` (default). The old 1-channel
    binary mask path is gone; any pre-existing FM checkpoint trained before
    this change is incompatible and cannot be resumed with `--resume`.

19. **Obs channel layout (source of a real, now-fixed bug)**: `obs` is always
    `[mask(0), obs_u(1), obs_v(2), confidence(3), x(4), y(5)]`. `sample()`'s DPS
    guidance previously sliced `obs[:, :2]`/`obs[:, 2:3]` (i.e. `[mask, obs_v]`
    as "observed u,v" weighted by `obs_v` as "confidence") — wrong on both
    counts. Now correctly `obs[:, 1:3]` / `obs[:, 3:4]`. Any code that touches
    obs-channel slicing must respect this layout exactly.

20. **Physics prior is the flow-matching source distribution, not a target**:
    `FlowMatchingModel.physics_prior()` replaces `torch.randn_like` as `x_noise`
    in both `flow_match_loss()` and `sample()`. It is not a residual-correction
    target and does not change the loss function — only what the ODE starts
    from. Train and sample must use the same `use_physics_prior` value.

21. **Calibration is computed every inference run**: `infer_fm.py` prints
    spread-error correlation (want > 0) and 90% interval coverage (want ~0.90)
    via `src/evaluation/calibration.py`. Use these numbers, not just RMSE, to
    judge whether the ensemble spread is trustworthy for risk-aware planning.
