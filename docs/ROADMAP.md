# ROADMAP.md — Development Phases

## Phase 1: Prototype ✅ COMPLETE

Goal: Validate that the ML pipeline works end-to-end on real GPU with real STL.

Status:
  [x] LBM solver          — D2Q9, GPU PyTorch, steady + transient modes
  [x] STL geometry loader — triangle rasterization, correct diagonal shapes
  [x] Drone sampler       — A* street-following path, NaN for solid cells
  [x] U-FNO model         — ~7.4M params, heteroscedastic output
  [x] Training loop       — NLL + divergence loss, tqdm progress bars
  [x] Visualization       — 6-panel dashboard, m/s colorbars, dark theme
  [x] Run on real GPU     — RTX 5000 Ada, 256×256 grid, ~10 epochs
  [x] Load real STL       — city_model.STL, Y-up, correct orientation
  [x] Evaluate metrics    — live RMSE/MAE in dashboard
  [x] Physical units      — m/s colorbars via lbm_to_ms scaling
  [x] Orientation fixed   — matches STL top-view, wind right→left

### What Phase 1 delivers
- Full synthetic data pipeline (LBM → drone obs → U-FNO training)
- Working prediction with uncertainty quantification
- 6-panel interactive dashboard showing GT vs prediction vs error
- LBM data caching (MD5-keyed, ~3 min to regenerate at 256²)
- Training time: ~15–30 min at 256², 10 epochs, RTX 5000 Ada

---

## Phase 2: Latent Diffusion Model

Goal: Replace U-FNO with a diffusion model for richer probabilistic output.
      Instead of a single mean + variance, get full posterior samples.

Why:
  - Urban wind is multimodal near building corners (flow can go either way)
  - A diffusion model samples from p(wind_field | observations, geometry)
  - Each sample is a physically plausible full wind field
  - Ensemble of samples → uncertainty + decision support for drone planning

Architecture plan:
```
  VAE Encoder:   wind field [2, H, W] → latent z [C, H/4, W/4]

  Diffusion:     score network = U-Net conditioned on
                   - geometry embedding (CNN of obstacle mask)
                   - observation embedding (FNO or PointNet of sparse obs)
                   - time embedding (diffusion timestep t)

  VAE Decoder:   latent z → wind field [2, H, W]

  Inference:     DDIM (50 steps, ~0.5s on RTX 5000 Ada)
                 or Consistency Model (1–4 steps, ~50ms)
```

Key references:
  - Latent Diffusion Models (Rombach et al., 2022)
  - Score-based generative models (Song et al., 2021)
  - CorrDiff (NVIDIA, 2024) — diffusion for weather downscaling

Training data: same LBM-generated fields as Phase 1
New requirement: need more timesteps (500+) for diffusion training diversity

### Phase 2 TODO
- [ ] Build VAE for wind fields (encoder + decoder)
- [ ] Train VAE, verify reconstruction quality
- [ ] Build conditional score network (U-Net + cross-attention on obs)
- [ ] Train diffusion model
- [ ] Implement DDIM sampler
- [ ] Update visualization: show multiple posterior samples
- [ ] Compare: U-FNO (deterministic) vs diffusion (probabilistic)

---

## Phase 3: World Model

Goal: Learn a latent dynamics model of urban wind that the drone can query
      and update as it flies.

Concept:
```
  State:  h_t = latent belief about current wind field

  Dynamics:   h_{t+1} = f_dynamics(h_t, boundary_conditions_t)

  Observation update:  h_t → h_t' = f_update(h_t, drone_obs_t)
  (like a learned Kalman filter / particle filter in latent space)

  Decoder:    h_t → wind_field_t  (dense 2D prediction + uncertainty)
```

This enables:
  1. Drone takes a measurement → updates latent state → better prediction
  2. Model rolls out future states → drone can plan ahead
  3. Adaptive trajectory: fly where uncertainty σ(x,y) is highest
     (active sensing / informative path planning)

Architecture options:
  - RSSM (Recurrent State Space Model) — from DreamerV3
  - Transformer world model — from Genie / DIAMOND
  - Neural ODE / flow matching — continuous-time dynamics

Key references:
  - DreamerV3 (Hafner et al., 2023)
  - DIAMOND (Alonso et al., 2024)
  - Neural Process family (Garnelo et al.) — for observation conditioning

### Phase 3 TODO
- [ ] Decide architecture: RSSM vs transformer
- [ ] Build latent dynamics model
- [ ] Train on LBM time series (longer runs: 1000+ steps, transient mode)
- [ ] Implement observation update step
- [ ] Build adaptive drone path planner (maximize info gain)
- [ ] Evaluate: does adaptive path beat fixed A* street path?

---

## Phase 4: Real CFD Integration

Goal: Replace LBM prototype with proper CFD data for publication quality.

CFD strategy (minimal runs using superposition):
```
  4 base runs:   N, S, E, W unit inflows (RANS, steady)
  4 diagonal:    NE, NW, SE, SW
  2 speed scale: low, high
  3–5 unsteady:  LES snapshots for temporal dynamics
  ─────────────────────────────────────────────────────
  ~15 total CFD runs  →  synthetic training via superposition
```

Solver recommendation: OpenFOAM (free) or STAR-CCM+ / Fluent
  - Domain: real city geometry at 1:1 scale or 1:40 scale
  - Turbulence: k-ω SST (RANS) or dynamic Smagorinsky (LES)
  - Output: U, V fields on horizontal slice at 20m (real) / 0.5m (model)

### Phase 4 TODO
- [ ] Set up OpenFOAM case from city_model.STL geometry
- [ ] Run 4 cardinal direction base cases
- [ ] Validate against known urban flow benchmarks
- [ ] Replace LBM data with CFD data in training pipeline
- [ ] Optionally: real drone flight data for validation

---

## Open Questions / Future Decisions

1. **Fixed horizon vs steady-state prediction?**
   Current: fixed horizon (t+10 steps). If boundary conditions are stable,
   steady-state prediction is more useful. Worth experimenting.

2. **Multi-altitude prediction?**
   Start with single 20m slice. Extension: add Z as conditioning input
   (predict any slice given Z coordinate).

3. **Domain generalization?**
   Currently city-specific (geometry baked in). Future: encode geometry
   via graph network or implicit neural field for generalization across cities.

4. **Real drone data?**
   Phase 1–3 use synthetic drone paths. Validation against real drone
   flights (when available) would strengthen the pipeline significantly.
