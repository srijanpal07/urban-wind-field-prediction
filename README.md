# Urban Wind Field Prediction Pipeline

End-to-end pipeline: LBM solver → synthetic drone sampling → U-FNO prediction → live dashboard.

## Setup

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Files

```
urban-wind-field-prediction/
├── run_pipeline.py        ← START HERE: runs everything end-to-end
├── requirements.txt
├── src/
│   ├── lbm_solver.py      ← 2D LBM wind field generator (GPU-native)
│   ├── geometry.py        ← STL → 2D obstacle mask converter
│   ├── drone_sampler.py   ← Synthetic drone trajectory + noisy sampling
│   ├── model.py           ← U-FNO model (Fourier Neural Operator)
│   ├── train.py           ← Training loop
│   └── visualize.py       ← Interactive dashboard
└── docs/
    ├── ARCHITECTURE.md
    ├── ROADMAP.md
    └── DECISIONS.md
```

## Quick Start

### With your STL file
```bash
cd wind_prediction
python run_pipeline.py --stl /path/to/city.stl
```

### Without STL (synthetic city)
```bash
python run_pipeline.py
```

### Save animation to GIF (no display needed)
```bash
python run_pipeline.py --stl city.stl --save output.gif
```

### Just visualize (skip training, use persistence baseline)
```bash
python run_pipeline.py --stl city.stl --no-train
```

## Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--stl` | None | Path to city .stl file |
| `--grid` | 128 | Grid resolution (128 or 256) |
| `--warmup` | 400 | LBM warmup steps (more = more developed flow) |
| `--steps` | 150 | Number of time snapshots to collect |
| `--speed` | 0.08 | Inlet wind speed (LB units, ~0.05–0.15) |
| `--angle` | 45.0 | Wind inlet angle in degrees (0=East, 90=North) |
| `--epochs` | 60 | Training epochs |
| `--horizon` | 10 | Prediction horizon (timesteps ahead) |
| `--device` | cuda | `cuda` or `cpu` |

## Dashboard Panels

```
┌─────────────────┬─────────────────┬─────────────────┐
│  LBM Ground     │  U-FNO          │  Uncertainty σ  │
│  Truth          │  Prediction     │  (where model   │
│  (wind speed    │  (same colormap)│   is unsure)    │
│  + quiver)      │                 │                  │
├─────────────────┼─────────────────┬─────────────────┤
│  Absolute Error │  Drone          │  RMSE / MAE     │
│  |GT - Pred|    │  Trajectory     │  over time      │
│                 │  (live path +   │  (live plot)    │
│                 │  obs scatter)   │                  │
└─────────────────┴─────────────────┴─────────────────┘
```

**Green dot** = current drone position  
**Orange diamonds** = waypoints  
**Colored dots on trajectory** = wind observations (blue=negative u, red=positive u)  
**Quiver arrows** = wind direction vectors  

## STL Notes

- Buildings should be solid volumes (yours are ✓)
- Ground plane included is fine (it's filtered out at slice height)
- STL units: the script auto-detects bounding box and slices at 30% height
- For 1:40 scale model (5m × 5m real), the grid maps directly to your STL units

## Architecture: U-FNO

```
Input [B, 6, H, W]:
  ch0: building mask (1=solid)
  ch1: drone-observed u (Gaussian-splatted)
  ch2: drone-observed v
  ch3: observation confidence
  ch4: x coordinate grid
  ch5: y coordinate grid
          ↓
  Lift → FNO Layers (spectral conv + pointwise) → Skip → Project
          ↓
Output [B, 4, H, W]:
  ch0: predicted u(x,y)
  ch1: predicted v(x,y)
  ch2: uncertainty σ_u
  ch3: uncertainty σ_v
```

## Next Steps (Phase 2)

- Replace U-FNO with **Latent Diffusion Model** for richer uncertainty
- Add **real STL-based CFD** (OpenFOAM) runs for training data
- Implement **adaptive drone trajectory** using uncertainty maps
- **World model** (RSSM): latent state updated by drone observations
