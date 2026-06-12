---
name: project-overview
description: Core goal and architecture of the urban wind field prediction project
metadata:
  type: project
---

# Urban Wind Field Prediction via Drone Sampling

**Goal:** ML pipeline that predicts 2D urban wind fields from sparse drone measurements
at a fixed altitude (20m real / 0.5m in 1:40 scale model). The drone flies an A*
street-following path, samples wind, and a neural net reconstructs the full wind
field — including a short-horizon forecast and per-cell uncertainty map.

**Why:** Data assimilation + short-horizon forecasting for drone path planning in urban
environments. The predicted uncertainty map drives the *next* flight leg's trajectory.

## Physical Setup

- 1:40 scale physical city model (~5m×5m = 200m×200m real)
- STL: `data/city_model.STL`, Y-up axes, X=10000mm × Z=11250mm footprint
- 2D horizontal slice at Y=730mm (30% of max height ≈ 20m real)
- Inputs: building geometry mask + sparse (u,v) samples from drone over last ~4 min
- Output: dense u(x,y), v(x,y) + per-cell uncertainty σ(x,y), valid at t+5min

## Current Phase: Phase 1 ✅ COMPLETE

Pipeline running on RTX 5000 Ada (32GB VRAM) with real STL at 256×256 resolution.

| File | Role |
|------|------|
| `run_pipeline.py` | Entry point (STL → LBM → train → visualize) |
| `src/lbm_solver.py` | D2Q9 LBM wind simulator, GPU PyTorch, steady + transient |
| `src/geometry.py` | STL → 2D mask via triangle rasterization (skimage.draw) |
| `src/drone_sampler.py` | A* street-following path + bilinear interpolation + noise |
| `src/model.py` | U-FNO (~7.4M params), heteroscedastic: u,v,log_var_u,log_var_v |
| `src/train.py` | NLL + divergence-free loss, AdamW, tqdm progress bars |
| `src/visualize.py` | 6-panel dashboard, m/s colorbars, smooth building rendering |

## Key Implementation Details

- **Coordinate system**: Z→columns (horizontal), X→rows inverted; `invert_xaxis()` on all panels → wind flows right→left visually
- **LBM inlet**: auto-selects left/right side from sign of ux_in (Zou-He BC)
- **FNO modes**: `max(20, grid_size // 8)` — 32 at 256², stored in checkpoint
- **m/s colorbars**: `lbm_to_ms = ref_speed / inlet_speed = 125×` (10 m/s default)
- **Building rendering**: 2× upsample + Gaussian blur + contourf (smooth edges)
- **Cache**: `data/lbm_data.npz` keyed by MD5(mask) + sim_mode

## Roadmap

- **Phase 2**: Replace U-FNO with Latent Diffusion Model (full posterior sampling)
- **Phase 3**: World Model / RSSM (latent dynamics + observation update → adaptive path)
- **Phase 4**: Replace LBM with real OpenFOAM CFD runs

## How to Apply

Always think of this as a physics-ML problem:
- LBM units must not be rescaled in computation (only display-side via FuncFormatter)
- The model predicts heteroscedastic uncertainty — use σ for downstream planning
- Adaptive drone trajectory (informative path planning) is the Phase 3 goal
- Do not change `skimage.draw.polygon` rasterization back to bounding boxes
