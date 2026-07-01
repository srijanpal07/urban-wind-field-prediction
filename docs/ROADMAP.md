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

## Phase 2: Spatiotemporal Flow-Matching Forecast Model 🔄 IN PROGRESS

Goal: Replace U-FNO with a flow-matching generative model that forecasts the
      full 5-minute future wind field evolution from the past 5 minutes of
      drone observations. Output is an ensemble of N plausible future field
      sequences, not a single mean+variance snapshot.

### Problem framing (precise)

```
  Input:   obstacle_mask [H, W]                       (fixed geometry)
           drone obs over past 5 min:
             obs_u, obs_v, confidence [H, W]           (splatted to grid)
             (same 6-channel format as Phase 1)

  Output:  N ensemble samples of next-5-min field evolution
             each sample: (u, v) [T_out, H, W]
             T_out = number of saved timesteps in 5-min forecast window

  Online:  ensemble → path-conditioned exceedance probability → next leg
```

### Why this replaces the original VAE+DDPM plan

- **Temporal forecasting, not snapshot prediction**: the drone needs to plan
  the next 5-min leg against how the wind *evolves*, not just what it looks
  like now. Target is a (T_out, H, W) field sequence, not a single frame.
- **Flow matching over DDPM**: straight-line ODE paths → 10–20 integration
  steps vs 100–1000. Critical for 30–60s online inference budget per waypoint.
  Same uncertainty quality; simpler training loss (no noise schedule to tune).
- **No VAE needed yet**: pixel-space operation at 256×256 with T_out~30
  timesteps is feasible on the RTX 5000 Ada (32 GB). Add latent space only
  if memory benchmarking shows a bottleneck.

### Key design decisions (June 2026)

- **LBM data stays**: no URANS/LES. Existing 128-condition LBM cache is the
  training foundation. OpenFOAM CFD arrives later (Phase 4).
- **Sparse obs encoding unchanged**: grid splatting (obs_u, obs_v, confidence
  channels) carries forward from Phase 1. GNN trajectory encoder is deferred.
- **Hard physics constraints post-generation** (not soft losses):
    - Leray projection: exact ∇·u = 0 (incompressibility) in 2D Fourier space
    - Obstacle mask: zero velocity inside buildings (exact no-slip)
- **15-minute wind stability**: atmospheric BCs are stable for ~15-min windows.
  The 10-min window (5-min obs + 5-min forecast) fits entirely within one
  stable period. The model never handles BC changes mid-window. This means
  each LBM condition yields many clean (obs, forecast) pairs under a fixed BC.
- **2D only**: no 3D extension in Phase 2. Single horizontal slice at fixed
  altitude, same as Phase 1.

### Architecture

```
Offline (trained once on LBM data):

  Geometry encoder  obstacle_mask [H, W]
  (small CNN)       → geometry_features [H, W, C_geo]
                    injected at every scale of the U-Net

  Spatiotemporal    Input: noisy interpolation x_t [T, H, W] (noise↔target)
  U-Net backbone    Conditioning: geometry_features + obs channels
    - 3D conv in early layers     (local spatiotemporal correlations)
    - 2D spatial attention        (global flow structure, lower resolutions)
    - 1D temporal attention       (flow evolution, full resolution)
    - ODE timestep embedding t    (position in flow-matching process)
  Output: predicted velocity field v_θ(x_t, t, c) [T, H, W]

  Training loss (flow matching):
    x_t = (1−t)·x_noise + t·x_target    (straight-line interpolation)
    L = E[‖v_θ(x_t, t, c) − (x_target − x_noise)‖²]
    where c = geometry channels + splatted drone observation channels

Online inference at each waypoint (target: <60s for N=20 on RTX 5000 Ada):

  1. Splat drone obs onto grid → obs_u, obs_v, confidence [H, W]
  2. Generate N=20 ensemble samples (batched on GPU) via DPS-guided ODE:
       x_0 ~ N(0, I)   (pure noise)
       For each step s = 0 → 1 (20 steps):
         v̂ = v_θ(x_s, s, c_geo)              model predicts velocity field
         x̂_1 = x_s + (1−s)·v̂               estimated clean field
         g_obs = ∇_{x_s}‖obs − H(x̂_1)‖²    DPS guidance (autodiff through H)
         x_{s+ds} = x_s + ds·v̂ − ρ·g_obs   guided Euler step
         (H is the grid-splatting operator extracting obs at drone positions)
  3. Post-process each sample (hard constraints):
       Leray projection:  u ← u − ∇(∇⁻²·∇·u)   (exact ∇·u = 0, 2D FFT)
       Obstacle mask:     velocity = 0 inside buildings
  4. For each candidate next-leg path, compute exceedance probability:
       exceedance_prob[path] = fraction of samples where
         max wind along path over 5 min > safety threshold
  5. Return ensemble mean + exceedance probabilities to trajectory optimizer
```

### Training data requirements

LBM cache regenerated at 500 snapshots/condition, 512×512 (`--steps 500 --warmup
2000`), 128 train + 16 test conditions — completed June 2026. Note: 500 LBM
snapshots is ~6.7s of simulated LBM time (see `dt_snapshot` derivation in
CLAUDE.md), not 10–15 minutes — the "5-minute forecast" refers to real drone
flight time under the 15-minute atmospheric stability assumption, not LBM
simulated time. The snapshot count was raised for more training *windows* per
condition, not temporal coverage.

### Key references

- Conditional Neural Field Latent Diffusion (CoNFiLD) — Cornell, 2024
  The DPS conditioning strategy (Bayesian posterior sampling via guided score)
  is taken directly from this paper. Their F = sparse sensor masking operator;
  our H = grid-splatting + path extraction operator. Same math, different F.
- Flow Matching for Generative Modeling — Lipman et al., 2022
- Rectified Flow — Liu et al., 2022
- Diffusion Posterior Sampling (DPS) — Chung et al., 2022

### Phase 2 TODO

- [x] Extend LBM runs: `--steps 500` per condition; rebuild data cache (running, 512×512)
- [x] Build spatiotemporal U-Net backbone (3D conv + 2D spatial + 1D temporal attention)
- [x] Implement flow-matching training loop (straight-line x_t, velocity loss)
- [x] Wire geometry conditioning: CNN encoder + channel concat at every U-Net scale
- [x] Implement DPS guidance at inference (autodiff through H operator)
- [x] Implement Leray projection post-processing (2D FFT)
- [x] Implement obstacle mask hard constraint post-processing
- [x] Batch N=20 ensemble samples on GPU; benchmark inference time vs 60s target
- [x] SDF geometry channel (binary mask + signed distance field conditioning)
- [x] Physics-informed flow-matching prior (divergence-free ambient field, not pure noise)
- [x] Ensemble calibration utility (spread-skill correlation, interval coverage)
- [x] Fix DPS guidance channel-slicing bug (was reading [mask,obs_v] instead of [obs_u,obs_v])
- [x] Divergence-residual diagnostic (`src/evaluation/calibration.py`), global + near-obstacle split
- [x] Soft divergence + no-penetration physics penalties baked into the training loss
      (`lambda_div`, `lambda_solid`), not just relied on post-hoc at inference
- [x] Observation-consistency penalty (`lambda_obs=1.0`) — normalized by sum(obs_conf)
      so it measures average error at observed locations, not diluted by unobserved
      cells. Fixes model predicting low wind where drone measured high wind.
- [x] AMP in sample() (~30% inference speedup, 133s → 94s); chunk_size=1 OOM fix;
      del solver before sampling; wall-clock timing printed vs 60s budget
- [x] Multi-segment animated dashboard (scripts/viz_fm.py): A→W1→W2→B with three
      DPS inference updates showing prediction sharpening with observations
- [x] Rho sweep (0.05–0.50): no meaningful effect on quality — rho is not the lever
- [ ] Full 200-epoch training run — **Run #3 active** (lambda_div=0.1, lambda_solid=0.1,
      lambda_obs=1.0; at epoch 10 RMSE=3.14 m/s, corr=+0.352 vs run #2 corr=-0.012)
- [ ] Evaluate at epoch 40: full comparison run #2 ep42 vs run #3 ep40
- [ ] Compare U-FNO (deterministic) vs flow-matching (ensemble)
- [ ] Path-conditioned exceedance probability risk metric
- [ ] Update visualization: spread, mean field, risk overlay on candidate paths

### Implementation Notes (June 2026)

- **Attention memory constraint**: SpatialAttnBlock is O(H²W²). At 512×512, attention
  is only applied at ≤64×64 feature maps (`attn_start = ceil(log2(grid_size/64))`).
  Applying attention at 128×128 requires 1.1 GB/head and causes OOM.
- **Training memory**: AMP (float16) + gradient checkpointing required at 512×512.
  Recommended: `--hidden 32 --batch 4` (23.7 GB peak VRAM on RTX 5000 Ada).
- **T_out vs physical time**: T_out=10 frames at collect_every=3 ≈ 0.4s of LBM
  simulated time. The "5-minute forecast" refers to real flight time under the
  15-minute atmospheric stability assumption, not 5 min of LBM dynamics.
- **Smoke test (3 epochs, 19 conditions)**: Train 0.106→0.022, Val 0.056→0.019.
  Clean convergence confirmed. Pipeline end-to-end verified.
- **Smoke test #2, post architecture changes (1 epoch, 2 conditions, 512×512,
  hidden=32, AMP+checkpoint, physics prior on)**: Train 0.032, Val 0.013. Confirms
  SDF channels + physics-informed prior + DPS bug fix work end-to-end on real
  data before committing to the full 200-epoch run.
- **Training run #1 stopped at epoch 30** (val 0.000450) to add divergence +
  no-penetration physics penalties to the loss — see "Divergence-Free Guarantee
  Review" below. Checkpoint archived to
  `outputs/flow_matching/fm_model_pre_physics_loss.pth`.
- **Smoke test #3, post physics-loss changes (1 epoch, 2 conditions)**: Train
  0.0366 (data=0.0316 div=0.0109 solid=0.0395), Val 0.0115. Component balance
  confirmed sane (data term still dominates) before restarting the full run.
  Training run #2 (with physics-loss penalties) launched from epoch 0.

### Recommendations Triage (June 2026)

An external 8-point architectural review was assessed against the actual state
of the codebase before the full training run. Disposition:

**Implemented now** (cheap, additive, didn't require restarting training since
no full run had started yet):
- SDF geometry channel (`src/data/geometry.mask_to_sdf`) — smoother gradients
  near building walls than the binary mask alone.
- Physics-informed flow-matching prior (`FlowMatchingModel.physics_prior`) —
  source distribution is now a confidence-weighted ambient field, zeroed inside
  obstacles and Leray-projected to divergence-free, instead of pure Gaussian
  noise. The network now only has to learn the obstacle-wake correction, not
  the whole field from scratch.
- Ensemble calibration utility (`src/evaluation/calibration.py`) — spread-skill
  correlation + interval coverage, wired into `infer_fm.py` output.
- **Bug found and fixed during this work**: `FlowMatchingModel.sample()`'s DPS
  guidance term sliced `obs[:, :2]` / `obs[:, 2:3]` as if channel 0 were obs_u —
  it's actually the geometry mask. DPS was guiding against `[mask, obs_v]`
  instead of `[obs_u, obs_v]` weighted by confidence. Fixed to `obs[:, 1:3]` /
  `obs[:, 3:4]`. This was silently degrading every prior ensemble inference run.

**Already on the roadmap, not pulled forward**: state-update/data-assimilation
formulation and "flow state evolves with each observation" (recursive latent
dynamics) are Phase 3's RSSM/world-model concept (see Phase 3 below), not a
Phase 2 change. Pulling them into Phase 2 now would mean replacing an
unevaluated model with an unbuilt one. Decision: finish the Phase 2 ensemble
run, evaluate it, and only escalate to a stateful formulation if evaluation
shows partial-observability failures the ensemble can't address.

**Already deferred for a documented reason, no new information changes that**:
richer observation encoding (GNN over grid-splatting) — see Open Question 3
below. Still deferred until ablation shows splatting is the bottleneck.

**Not pursued**: physics-prior-as-residual-target (predicting a correction to
a full potential-flow/Stokes solve rather than using the prior as the
flow-matching source distribution) — the source-distribution version
implemented above captures most of the benefit (network starts closer to the
answer) at a fraction of the engineering cost (no PDE solve per sample, reuses
the existing Leray projection).

### Divergence-Free Guarantee Review (June 2026)

A follow-up review specifically asked whether the pipeline's incompressibility
claim (the Leray projection in `infer_fm.py`) is actually guaranteed or just
plausible. Checked directly against the code:

- The Leray projection (`FlowMatchingModel.leray_project`, periodic-domain FFT)
  is applied to the full field, then solid cells are zeroed *afterward*
  (`infer_fm.py`). The projection has no knowledge that those cells will be
  masked, so this ordering can reintroduce divergence right at building edges
  — confirmed with a synthetic test (divergence-free field, masked the same
  way): near-obstacle residual was ~47x the interior residual (1.79 vs. 0.0).
- DPS guidance during the 20 sampling steps has no physics term at all — only
  an observation-matching gradient (`flow_matching.py`, `sample()`). This is
  standard practice for DPS (relies on the trained model staying near the data
  manifold, with the projection as one-shot cleanup, not a per-step
  constraint) — not a bug, but it means nothing is enforced mid-sampling.
- Solid cells were entirely excluded from the training loss — the network got
  zero gradient signal about what to predict there, which is exactly why
  post-hoc zero-masking creates a sharp seam.
- No diagnostic existed to check any of this. `calibration.py` covered
  spread-skill and coverage but not divergence.

Two changes made as a result:
1. `divergence_residual()` added to `src/evaluation/calibration.py`, wired
   into `infer_fm.py`'s printed output — global mean |div u| plus a
   near-obstacle vs. interior split, computed on the actual post-projection,
   post-masking field used downstream. Inference-time only, no model changes.
2. Two soft physics penalties added to `flow_match_loss()` —
   `lambda_div` (divergence in fluid cells) and `lambda_solid` (velocity at
   solid cells, no-penetration), both computed on the model's own one-step
   clean-field estimate `x_hat1 = x_t + (1-s)·v_pred` (the same quantity DPS
   guidance already computes at inference). This gives the network explicit
   gradient signal toward both constraints instead of relying purely on data
   resemblance plus a post-hoc fix now known to leak at building boundaries.
   Judged a large enough change to the training objective to restart the
   200-epoch run from scratch rather than resume partway through.

Neither change makes the guarantee exact — the projection is still only exact
for a periodic domain, and the training penalties are soft, not hard
constraints. The honest framing: this makes the divergence-free claim
checkable (every `infer_fm.py` run now reports it) and measurably better
trained-for, not mathematically guaranteed. A harder guarantee would require
solving a proper Poisson equation with no-penetration boundary conditions at
the obstacle walls instead of a periodic-domain FFT — judged out of scope
until the soft-penalty approach is evaluated and found insufficient.

---

## Phase 3: Trajectory Optimizer + Active Sensing

Goal: Close the loop between the Phase 2 generative model and the drone
      flight planner. The drone actively chooses its next leg to minimize
      risk and reduce forecast uncertainty.

Concept:
```
  At each waypoint:
    ensemble of N future field sequences   (from Phase 2)
         ↓
    risk metric per candidate path         (exceedance probability)
         ↓
    trajectory optimizer                   (energy + risk tradeoff)
         ↓
    selected next leg A→B
         ↓
    drone flies, collects new observations
         ↓
    (loop back to Phase 2 inference)
```

Extensions beyond Phase 2:
  1. Active sensing: steer the drone toward high-uncertainty regions
     (maximise information gain on the next leg, not just minimise risk)
  2. Multi-leg planning: optimise over K future legs, not just the next one
  3. Latent dynamics: if Phase 2 ensemble is too slow for long-horizon planning,
     learn a fast latent predictor on top of Phase 2's latent space

Architecture options:
  - Energy + risk penalty optimizer (simple, Phase 3 start):
      min: travel_cost(path) + λ · exceedance_prob(path)
      subject to: avoids obstacles, connects waypoints
  - RSSM / Transformer world model (if multi-leg rollout is needed):
      roll out future states in latent space without re-running full diffusion
  - Informative path planning (active sensing):
      maximise ensemble variance reduction along candidate paths

Key references:
  - DreamerV3 (Hafner et al., 2023) — RSSM latent dynamics
  - DIAMOND (Alonso et al., 2024) — Transformer world model
  - Informative path planning literature (active sensing in uncertain fields)

### Phase 3 TODO
- [ ] Implement simple energy + risk optimizer using Phase 2 ensemble output
- [ ] Evaluate: does risk-aware path beat fixed A* street path?
- [ ] Implement active sensing path: maximise uncertainty reduction per leg
- [ ] Evaluate: does adaptive path improve forecast quality on the next leg?
- [ ] (Optional) Learn fast latent dynamics for multi-leg rollout

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

1. **Latent space for Phase 2?**
   Pixel-space at 256×256 × T_out~30 may hit memory or inference time limits.
   If N=20 samples × 20 ODE steps cannot fit in 60s, add VAE compression before
   the flow-matching U-Net (same approach as CoNFiLD).

2. **Multi-altitude prediction?**
   Start with single 20m slice. Extension: add Z as conditioning input
   (predict any slice given Z coordinate). Requires multi-altitude LBM data.

3. **GNN trajectory encoder?**
   Deferred from Phase 2. If grid splatting proves to be the accuracy bottleneck
   (encoder missing path geometry), replace with a small GNN that processes drone
   observations at their exact (x, y, t) positions without grid interpolation.

4. **Real drone data?**
   Phase 1–3 use synthetic drone paths. Validation against real drone
   flights (when available) would strengthen the pipeline significantly.

5. **Domain generalization?**
   Currently city-specific (geometry baked in). Future: encode geometry
   via graph network or implicit neural field for generalization across cities.
