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

- City model: physical scaled model at 1:40 scale (~5m × 5m footprint = ~200m × 200m real)
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

Output  → dense 2D wind field u(x,y), v(x,y) on the horizontal slice
        → uncertainty σ(x,y) per grid cell
        → prediction valid for t+5min ahead (forecast horizon)
```

Timing contract:
  t=0 to t=4min   drone samples wind along A→B leg
  t=4min          model runs inference (~seconds on GPU)
  t=5min          drone arrives at B; prediction window begins
  t=5 to t=10min  predicted wind field is used for planning next leg

## Current Implementation (Phase 1 — Working Prototype)

Phase 1 trains across 64 wind conditions (16 angles × 4 LBM speeds) so the
model generalises over the full 360° direction range and ~1–6 m/s speed range.
The pipeline is split into four standalone scripts.

### Multi-condition training philosophy
The model input is always **6 channels** (geometry + sparse drone obs + coords).
Angle and speed are NOT passed as inputs — the model learns to infer the full
wind field from the drone observations alone, without a "cheat code" label.
This is the physically meaningful setup: in real deployment the inlet conditions
are unknown; only the drone readings are available.

### Test split (held-out evaluation)
`generate_data.py` generates two datasets in one pass:

| Dataset | Angles | Speeds | Mode | File |
|---------|--------|--------|------|------|
| Training | 16 × every 22.5° → [0, 22.5, …, 337.5] | [0.02, 0.04, 0.08, 0.10] | transient | `data/lbm_multicond.npz` |
| Test | 8 × midpoints → [11.25, 56.25, …, 326.25] | [0.03, 0.06] | steady-state | `data/lbm_test.npz` |

Test angles are exactly midway between every other pair of training angles; test
speeds sit between training speeds. Neither appears during training — a clean
interpolation benchmark.

Use `--test-data data/lbm_test.npz` with `evaluate.py` to run the held-out
evaluation instead of random on-the-fly LBM conditions.

### Physical speed scale
LBM speed 0.08 is the reference and maps to `--ref-speed` m/s (default 5.0).
Other LBM speeds scale proportionally: 0.02→1.25, 0.04→2.5, 0.08→5.0, 0.10→6.25 m/s.
`lbm_to_ms = ref_speed / 0.08` is used as a **fixed constant** in evaluate.py
and infer.py for consistent physical units. (run_pipeline.py still uses the
per-condition formula `ref_speed / abs(speed)` for its single-condition display.)

### File Map
```
generate_data.py       ← Step 1: run LBM for 64 train + 16 test conditions
train_model.py         ← Step 2: train WindFNO on multi-condition dataset
infer.py               ← Step 3: infer at random or specified wind condition
evaluate.py            ← Step 4: benchmark on random or held-out conditions
run_pipeline.py        ← Legacy: single-condition end-to-end convenience script
src/
  lbm_solver.py        ← 2D LBM wind field simulator (D2Q9, GPU-native PyTorch)
  geometry.py          ← STL → 2D binary obstacle mask (triangle rasterization)
  drone_sampler.py     ← A* traverse path + polar noise wind sampling
  model.py             ← U-FNO neural network (Fourier Neural Operator), 6 in_channels
  train.py             ← Training loop; WindDataset samples across all N conditions
  visualize.py         ← 6-panel interactive dashboard with m/s colorbars
docs/
  ARCHITECTURE.md      ← Full technical architecture details
  ROADMAP.md           ← Phased development plan
  DECISIONS.md         ← Why we made key design choices
data/
  city_model.STL       ← Real 1:40 scale city model (Y-up)
  obstacle_mask.npy    ← Cached rasterized mask
  cache/               ← Per-condition LBM cache (lbm_{mode}_a{angle}_s{speed}.npz)
  lbm_multicond.npz    ← Training dataset [N,T,H,W] (64 conditions)
  lbm_test.npz         ← Held-out test dataset [N,T,H,W] (16 unseen conditions)
outputs/
  wind_fno.pth         ← Best trained checkpoint (includes modes + grid_size)
  wind_fno_history.png ← Training curve from train_model.py
  wind_dashboard.gif   ← Last saved animation
```

### Recommended Workflow
```bash
pip install -r requirements.txt

# 1. Generate train + test data (64 + 16 conditions; ~45-90 min on GPU at 512×512):
python generate_data.py --stl data/city_model.STL --grid 512 --warmup 2000

# 2. Train (50 epochs default):
python train_model.py

# 3. Infer at a random wind condition:
python infer.py --stl data/city_model.STL

# 3b. Infer at a specific condition:
python infer.py --stl data/city_model.STL --angle 135 --speed 0.08

# 3c. Save to GIF:
python infer.py --stl data/city_model.STL --save outputs/infer_result.gif

# 4a. Quick sanity check (10 random conditions, may overlap training angles):
python evaluate.py --stl data/city_model.STL

# 4b. Rigorous held-out evaluation (16 unseen angles/speeds, no LBM re-run needed):
python evaluate.py --stl data/city_model.STL --test-data data/lbm_test.npz

# Skip test data generation (training data only):
python generate_data.py --stl data/city_model.STL --skip-test

# Legacy single-condition pipeline (still works):
python run_pipeline.py --stl data/city_model.STL
```

### Key Defaults
| Script              | Argument          | Default                                     | Notes                                         |
|---------------------|-------------------|---------------------------------------------|-----------------------------------------------|
| generate_data.py    | `--angles`        | 0 22.5 45 … 337.5 (16 values)              | Training angles, every 22.5°                  |
| generate_data.py    | `--speeds`        | 0.02 0.04 0.08 0.10                         | 4 LBM speeds → 1.25–6.25 m/s                 |
| generate_data.py    | `--test-angles`   | 11.25 56.25 … 326.25 (8 values)            | Held-out angles (midpoints, unseen in train)  |
| generate_data.py    | `--test-speeds`   | 0.03 0.06                                   | Held-out speeds (between train speeds)        |
| generate_data.py    | `--warmup`        | 1000                                        | LBM warmup steps per condition                |
| generate_data.py    | `--steps`         | 150                                         | Snapshots collected per condition             |
| generate_data.py    | `--test-output`   | data/lbm_test.npz                           | Path for held-out test dataset                |
| generate_data.py    | `--skip-test`     | False                                       | Skip test set generation                      |
| train_model.py      | `--epochs`        | 50                                          | Training epochs                               |
| train_model.py      | `--batch`         | 32                                          | Batch size                                    |
| infer.py            | `--angle`         | random [0, 360)                             | Printed to stdout for reproducibility         |
| infer.py            | `--speed`         | random [0.02, 0.10]                         | Printed to stdout                             |
| infer.py            | `--ref-speed`     | 5.0                                         | m/s corresponding to LBM speed 0.08          |
| evaluate.py         | `--n`             | 10                                          | Random conditions (ignored if --test-data)    |
| evaluate.py         | `--test-data`     | None                                        | Path to lbm_test.npz for held-out eval       |
| evaluate.py         | `--seed`          | 42                                          | RNG seed for drone paths + random conditions  |

## Hardware
- GPU: NVIDIA RTX 5000 Ada Generation, 32GB VRAM
- CUDA: 12.8
- Driver: 570.211.01
- Use --device cuda (default). Falls back to CPU automatically if CUDA unavailable.

## What Claude Code Should Know

1. **Coordinate system**: The STL is Y-up (auto-detected). In the 2D mask:
   - Columns = Z axis (Z_min at col 0 = right side of display)
   - Rows = X axis inverted (X_max at row 0 = bottom of display)
   - `invert_xaxis()` is applied to all visualization panels so col 0 appears on
     the right. Wind with angle=0° (ux_in>0) flows right→left visually.

2. **LBM inlet side**: All 4 sides handled, auto-selected by sign of ux_in/uy_in.
   - ux_in > 0: inlet at LEFT col (col=0), outlet at RIGHT (col=W-1)
   - ux_in < 0: inlet at RIGHT col (col=W-1), outlet at LEFT (col=0)
   - uy_in > 0: inlet at TOP row (row=0, bottom of display), outlet at BOTTOM (row=H-1, top of display)
   - uy_in < 0: inlet at BOTTOM row (row=H-1), outlet at TOP (row=0)
   - Multiple sides active simultaneously when angle is diagonal (e.g. 45°)
   - With invert_xaxis: col=0 appears on the right, so ux_in>0 gives right-to-left
     visual wind direction. Do NOT change this without updating the BC logic.

3. **Geometry rasterization**: Uses `skimage.draw.polygon` + `skimage.draw.line`
   per triangle for correct diagonal/angular building shapes. The old bounding-box
   approach produced rectangular artifacts. Do NOT revert to bounding boxes.

4. **Obstacle mask**: True = solid building. Drone sampler skips solid cells
   (returns NaN). LBM applies bounce-back on solid nodes.

5. **Model checkpoint**: Saves `modes`, `grid_size`, `horizon` in the dict.
   Always use `ckpt.get('modes', 20)` when loading — do not hardcode modes=20.

6. **Fourier modes**: Scaled as `max(20, grid_size // 8)`.
   - grid=128 → modes=20
   - grid=256 → modes=32

7. **m/s colorbars**: `lbm_to_ms = ref_speed / 0.08` (fixed, default 62.5×).
   Applied via `FuncFormatter` — underlying data always stays in LBM units.

8. **LBM cache** (generate_data.py): Keyed per condition as
   `lbm_{mode_str}_a{angle:07.3f}_s{speed:.4f}.npz` with embedded `mask_hash`.
   Training mode_str = `transient`, test mode_str = `steady`.
   Cache is wiped by default on each generate_data.py run (use `--keep-cache` to skip).

9. **Quiver arrow direction**: The U component is negated (`-u`) in `set_UVC()`
   calls in `visualize.py`. This compensates for `invert_xaxis()` — matplotlib
   quiver draws positive U rightward on screen regardless of axis inversion, so
   negating makes arrows correctly point in the visual flow direction.

10. **Wind direction rotation**: `--angle-end` linearly rotates the inlet angle
    from `--angle` to `--angle-end` degrees over the collection window. Warmup
    always runs at the starting angle so flow is fully developed before rotation.

11. **Multi-condition dataset**: `src/train.py:WindDataset` accepts `u_all[N,T,H,W]`,
    picks a random condition index per sample in `__getitem__`, then a random
    time window within that condition. `n_samples = max(600, 30*N)` so coverage
    scales with the number of conditions. `run_pipeline.py` wraps its single
    `u_arr[T,H,W]` as `u_arr[np.newaxis]` before calling `train()`.

13. **Drone path**: Left-to-right traverse via `make_traverse_path(margin=0.1)`
    in `drone_sampler.py`. Uses A* with clearance=2 to avoid buildings. Training
    adds small Gaussian jitter (2% of grid) for diversity. Evaluate uses random
    seeds per condition.

14. **Noise model**: Polar noise — perturb speed (±`noise_speed_std`=0.008 LBU,
    ~0.5 m/s) and direction (±`noise_angle_std`=10°) independently. More
    physically realistic than isotropic Cartesian Gaussian noise.

15. **Grid size**: Always derived from data shape in `train()` — `grid_size = H`
    from `u_all.shape`. Never hardcode it. evaluate.py and infer.py read
    `grid_size` from `ckpt.get('grid_size', 256)` after loading the checkpoint
    (model must load before geometry to get grid_size).

16. **obs_window**: Training uses `total_steps=80` observations per sample
    (≈10 Hz × 8 s traverse). evaluate.py uses `n_obs_per_pred=80` to match.
    Mismatch causes degraded metrics.

12. **Phase 2 goal**: Replace U-FNO with a Latent Diffusion Model.
    **Phase 3 goal**: World Model (learned latent dynamics of urban wind).
    See docs/ROADMAP.md for details.
