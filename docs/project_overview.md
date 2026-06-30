---
name: project-overview
description: Core goal, architecture, and current status of the urban wind field prediction project
metadata:
  type: project
---

# Urban Wind Field Prediction via Drone Sampling

**Goal:** ML pipeline that predicts 2D urban wind fields from sparse drone measurements
at a fixed altitude (20m real / 0.5m in 1:40 scale model). The drone flies an A*
street-following path, samples wind, and a neural net reconstructs the full wind
field — including a short-horizon forecast and per-cell uncertainty map.

**Why:** Data assimilation + short-horizon forecasting for drone path planning in urban
environments. The predicted uncertainty/risk map drives the *next* flight leg's trajectory.

## Physical Setup

- 1:40 scale physical city model (~10m×11.25m = ~400m×450m real)
- STL: `data/city_model.STL`, Y-up axes, X=10000mm × Z=11250mm footprint
- 2D horizontal slice at Y=730mm (30% of max height ≈ 20m real)
- Inputs: building geometry mask + sparse (u,v) samples from drone over last ~4 min
  (2400 samples at 10 Hz over a 4-minute leg)
- Phase 1 output: dense u(x,y), v(x,y) + per-cell uncertainty σ(x,y), single snapshot
- Phase 2 output: N=20 ensemble sequences u(x,y,t), v(x,y,t) for risk-aware planning

## Phase Status

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | U-FNO prototype: LBM → drone obs → single-snapshot prediction | ✅ Complete |
| Phase 2 | Flow-matching generative model: probabilistic ensemble forecast | 🔄 Training |
| Phase 3 | Trajectory optimizer: risk-aware path planning from ensemble | ⏳ Planned |
| Phase 4 | Real CFD (OpenFOAM) replaces LBM data | ⏳ Planned |

## Repository Structure (post-reorg)

| Path | Role |
|------|------|
| `scripts/generate_data.py` | LBM data generation (128 train + 16 test conditions) |
| `scripts/train_fm.py` | Phase 2 flow-matching training entry point |
| `scripts/train_ufno.py` | Phase 1 U-FNO training (baseline) |
| `scripts/infer_fm.py` | Phase 2 ensemble inference + visualization |
| `scripts/infer_ufno.py` | Phase 1 U-FNO inference |
| `scripts/evaluate_ufno.py` | Phase 1 held-out evaluation |
| `src/data/lbm_solver.py` | D2Q9 LBM wind simulator, GPU PyTorch |
| `src/data/geometry.py` | STL → 2D mask (triangle rasterization) + SDF (build_geo_channels) |
| `src/data/drone_sampler.py` | A* path + vectorized obs_to_grid (2400 samples) |
| `src/models/ufno.py` | U-FNO (~7.4M params), heteroscedastic output |
| `src/models/flow_matching.py` | FlowMatchingModel (~14M params at hidden=32) + physics prior |
| `src/training/train_ufno.py` | NLL + divergence loss, WindDataset |
| `src/training/train_fm.py` | Flow-matching loss, WindSequenceDataset |
| `src/evaluation/calibration.py` | Ensemble spread-skill correlation + interval coverage |
| `src/viz/visualize.py` | 6-panel dashboard, m/s colorbars |
| `outputs/ufno/` | Archived Phase 1 checkpoint + visualizations |
| `outputs/flow_matching/` | Phase 2 active output directory |

## Phase 1 Benchmark (U-FNO, Archived)

Held-out test: 16 conditions (8 angles × 2 speeds), 512×512, 128 train conditions.

| Run | Epochs | Vec RMSE | Speed MAE | Dir Error |
|-----|--------|----------|-----------|-----------|
| Run 2 (best) | 200 | 2.75 m/s | 1.87 m/s | 43.4° |

Bottleneck: only 80 drone samples per training example (~8s traverse). Fixed to
2400 samples (4-minute realistic leg) for Phase 2 and all future U-FNO retrains.

## Phase 2 Architecture (Flow Matching)

- **FlowMatchingModel**: GeometryEncoder (mask+SDF → 8ch CNN) + SpatiotemporalUNet
- **Physics-informed prior**: flow-matching source distribution is a
  confidence-weighted, Leray-projected ambient field (not pure Gaussian) —
  `use_physics_prior=True` by default, must match between train and sample
- **Training**: straight-line flow matching loss; AMP + grad checkpointing for 512²
- **Inference**: DPS-guided ODE (20 steps), Leray projection, obstacle mask zeroing,
  calibration metrics (spread-error correlation, 90% coverage) printed every run
- **Config**: hidden=32, batch=4, T_out=10, n_levels=4 → 23.7 GB peak VRAM
- **Smoke tests**: (3 epochs, 19 conditions) train 0.106→0.022, val 0.056→0.019;
  (1 epoch, 2 real conditions, post SDF/physics-prior changes) train 0.032, val 0.013 ✓

## Key Implementation Notes

- **Coordinate system**: Z→columns, X→rows inverted; `invert_xaxis()` on all panels
- **LBM inlet**: auto-selects 4-side Zou-He BC from sign of ux_in/uy_in
- **lbm_to_ms = 62.5** (ref_speed=5.0 / LBM_speed=0.08) — fixed conversion constant
- **Attention gating**: SpatialAttnBlock only at ≤64×64 maps to avoid O(H²W²) OOM
- **U-FNO baseline preserved**: `outputs/ufno/wind_fno.pth` always runnable for comparison
- **Obs channel layout**: `[mask, obs_u, obs_v, confidence, x, y]` — a DPS guidance
  bug that read the wrong two channels (`[mask,obs_v]` instead of `[obs_u,obs_v]`)
  was found and fixed while implementing the physics prior; affected every prior
  ensemble inference run's conditioning quality
