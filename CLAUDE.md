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

Phase 1 is complete and running on real GPU with real STL. The pipeline
generates LBM data, trains the U-FNO, and produces a live visualization.

### File Map
```
run_pipeline.py        ← ENTRY POINT: runs full pipeline end-to-end
src/
  lbm_solver.py        ← 2D LBM wind field simulator (D2Q9, GPU-native PyTorch)
  geometry.py          ← STL → 2D binary obstacle mask (triangle rasterization)
  drone_sampler.py     ← A* street-following drone path + noisy wind sampling
  model.py             ← U-FNO neural network (Fourier Neural Operator)
  train.py             ← Training loop (NLL loss + divergence-free regularization)
  visualize.py         ← 6-panel interactive dashboard with m/s colorbars
docs/
  ARCHITECTURE.md      ← Full technical architecture details
  ROADMAP.md           ← Phased development plan
  DECISIONS.md         ← Why we made key design choices
data/
  city_model.STL       ← Real 1:40 scale city model (Y-up)
  obstacle_mask.npy    ← Cached rasterized mask (regenerated if STL changes)
  lbm_data.npz         ← Cached LBM simulation (regenerated if mask/mode changes)
outputs/
  wind_fno.pth         ← Best trained checkpoint (includes modes + grid_size)
  wind_dashboard.gif   ← Last saved animation
```

### How to Run
```bash
pip install -r requirements.txt

# Full pipeline with real STL (recommended):
python run_pipeline.py --stl data/city_model.STL

# Skip training, just visualize with existing model:
python run_pipeline.py --stl data/city_model.STL --no-train

# Save animation (headless / after full pipeline run):
python run_pipeline.py --stl data/city_model.STL --save outputs/wind_dashboard.gif

# Disable gusty wind (steady inlet):
python run_pipeline.py --stl data/city_model.STL --no-transient

# Override physical reference speed for colorbars:
python run_pipeline.py --stl data/city_model.STL --ref-speed 8.0

# Gradually rotate wind direction 90° during collection:
python run_pipeline.py --stl data/city_model.STL --angle-end 90
```

### Key Defaults (run_pipeline.py)
| Argument          | Default  | Notes                                           |
|-------------------|----------|-------------------------------------------------|
| `--grid`          | 256      | Grid resolution (256×256)                       |
| `--warmup`        | 400      | LBM warmup steps                                |
| `--steps`         | 150      | LBM snapshot collection steps                   |
| `--speed`         | 0.08     | Inlet speed in LBM units                        |
| `--angle`         | 0.0      | Inlet angle (0° = right-to-left visually)       |
| `--angle-end`     | None     | Rotate inlet angle to this value during collect |
| `--ref-speed`     | 10.0     | Physical inlet speed in m/s for colorbars       |
| `--epochs`        | 10       | Training epochs                                 |
| `--no-transient`  | off      | Disable gusty wind (transient is ON by default) |

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

7. **m/s colorbars**: `lbm_to_ms = ref_speed / abs(inlet_speed)` (default 125×).
   Applied via `FuncFormatter` — underlying data always stays in LBM units.

8. **LBM cache**: Keyed by MD5 of obstacle mask + sim_mode string.
   sim_mode encodes transient flag + start/end angles, e.g. `transient_a0_90`.
   Delete `data/lbm_data.npz` to force regeneration.

9. **Quiver arrow direction**: The U component is negated (`-u`) in `set_UVC()`
   calls in `visualize.py`. This compensates for `invert_xaxis()` — matplotlib
   quiver draws positive U rightward on screen regardless of axis inversion, so
   negating makes arrows correctly point in the visual flow direction.

10. **Wind direction rotation**: `--angle-end` linearly rotates the inlet angle
    from `--angle` to `--angle-end` degrees over the collection window. Warmup
    always runs at the starting angle so flow is fully developed before rotation.

9. **Phase 2 goal**: Replace U-FNO with a Latent Diffusion Model.
   **Phase 3 goal**: World Model (learned latent dynamics of urban wind).
   See docs/ROADMAP.md for details.
