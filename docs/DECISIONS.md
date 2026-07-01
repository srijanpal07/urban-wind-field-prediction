# DECISIONS.md — Design Decisions & Rationale

### Why lambda_obs (observation-consistency penalty) was added to training?

During inference inspection at epoch 10 of run #2, the ensemble mean was nearly
uniform blue (low wind) across the entire domain — including regions where the
drone had just measured high wind. The model was predicting low wind exactly
where it was told the wind was high, a fundamental data-assimilation failure.

Root cause: the flow-matching loss trains the model to produce "plausible wind
fields given the obs as a hint" — not "wind fields that reproduce the specific
measurements in the conditioning input." DPS guidance at inference tries to
enforce consistency, but its gradient (`rho × autograd.grad(obs_loss, x_req)`)
was too weak: `obs_conf` is small in absolute scale, and the `.mean()` over all
grid cells (including the ~97% with zero confidence) diluted the obs signal
~29× relative to what it would be if measured only at observed locations.

Fix: added `lambda_obs=1.0` penalty to `flow_match_loss()`:
  `L_obs = sum(obs_conf × (x_hat1[:,:,0] − obs_uv)²) / sum(obs_conf)`
Two key details: (a) computed on `x_hat1` (same one-step estimate as all other
physics penalties, no new concept), and (b) normalized by `sum(obs_conf)` not
`.mean()` — this measures "confidence-weighted mean error at observed locations"
rather than that signal diluted across unobserved cells. At epoch 1 of run #3
the obs component was ~32% of total loss and had dropped 85% by epoch 2, exactly
as expected (same rapid-learning pattern as lambda_solid in run #2).

At epoch 10 of run #3 vs run #2: spread-error correlation improved from -0.012
→ +0.352 (ensemble now meaningfully uncertain in the right places), and the
ensemble mean visually showed high wind in the correct high-speed inlet regions
instead of the uniform-blue failure. This is the single most important training
improvement made to the system.

### Why AMP was added to sample() and what rho does

`sample()` previously ran in fp32 (no autocast), while training used fp16 AMP.
Adding `torch.amp.autocast('cuda')` around `self.forward(...)` inside the DPS
loop cut sampling time 133s → ~94s (~30%) for free. Gradients from
`torch.autograd.grad(obs_loss, x_req)` remain fp32 — autograd always returns
gradients in the leaf tensor's dtype regardless of the autocast context used
during the forward pass.

**Rho sweep finding (0.05, 0.1, 0.2, 0.3, 0.5 tested on fixed condition)**:
RMSE varied <5% and coverage varied <5% across the full range. Rho is not a
meaningful lever — the main driver of calibration quality is the model itself
(i.e., training). Default rho=0.5 retained.

### Why chunk_size=1 in sample() (OOM fix)

`_per_frame` reshapes [B,C,T,H,W] → [B*T, C, H, W] for per-frame convolutions.
With N=20 samples and T_out=10, inner batch = 200. A single fp32 feature map
at 200×32×512×512 ≈ 26 GB → OOM on 32 GB GPU. chunk_size=1 processes one
sample at a time through all 20 ODE steps: inner batch = T_out=10, feature map
≈ 1.3 GB. Only increase chunk_size if confirmed VRAM headroom exists (AMP in
sample() freed ~15% VRAM via fp16 activations, but chunk_size=2 still didn't
improve speed — bottleneck is Python loop overhead per ODE step, not GPU compute).

## Why LBM instead of Navier-Stokes FVM?

LBM (Lattice Boltzmann Method) was chosen for the Phase 1 prototype because:
- Obstacles are trivially defined: solid nodes = bounce-back condition
- Naturally parallel: entire grid updates as tensor ops → GPU-native in PyTorch
- Gives unsteady, time-varying flow (vortex shedding) without pressure solver
- Simple to implement in ~200 lines of pure PyTorch
- Sufficient for prototype validation; not intended for publication accuracy

For Phase 4, this is replaced by OpenFOAM CFD which handles:
- 3D effects, turbulence modeling, atmospheric boundary layer
- Complex building geometries accurately
- Pressure-velocity coupling correctly

## Why U-FNO instead of ConvLSTM or plain U-Net?

Wind fields are solutions to PDEs (Navier-Stokes). The Fourier Neural Operator
learns in frequency space — it learns the solution operator of the PDE rather
than fitting a function. This means:
- Better generalization to different inlet conditions
- Captures long-range spatial correlations efficiently (global in one layer)
- Fast inference: milliseconds per prediction
- ~7.4M parameters is moderate — trainable on limited data

Plain ConvLSTM was considered but has limited receptive field. A standard U-Net
was also considered but lacks the global spectral reasoning FNO provides.

## Why fixed city geometry (not generalizable)?

Generalizing across different city geometries requires:
- Much more training data (hundreds of cities × hundreds of conditions)
- More powerful geometry encoders (graph networks, implicit neural fields)
- Higher model capacity

For Phase 1, fixing the geometry lets us:
- Bake the geometry into the obstacle mask channel (static input)
- Train with far less data (LBM runs → thousands of samples via time-windowing)
- Validate the ML pipeline before scaling up

Generalization is a future research direction, not a Phase 1 goal.

## Why a horizontal slice, not volumetric?

The drone flies at a fixed altitude (20m real-world). The relevant wind
information for drone control is on that slice. Volumetric prediction would:
- 64× increase in output size (for 64 vertical levels)
- Require 3D convolutions (higher memory, slower)
- Need vertical boundary conditions (harder)

A single horizontal slice is sufficient for drone path planning at fixed altitude.
If multi-altitude operation is needed later, Z can be added as a conditioning
input (predict any slice given Z coordinate).

## Why probabilistic output (heteroscedastic uncertainty)?

Urban wind has genuine aleatoric uncertainty:
- Wake regions behind buildings: chaotic, hard to predict
- Far-field regions: smooth and well-determined
- Regions unobserved by the drone: high uncertainty from lack of data

A single point prediction (MSE loss) cannot express this. The model outputs
log_var per grid cell (learned uncertainty). This allows:
- Downstream drone planner to avoid high-σ regions (safety)
- Or target high-σ regions (active sensing to reduce uncertainty)
- Honest evaluation: well-calibrated uncertainty = trustworthy model

## Why NLL loss + divergence regularizer?

NLL (Negative Log-Likelihood) with Gaussian assumption is the natural loss for
heteroscedastic regression. It jointly optimizes mean prediction and uncertainty.

Divergence regularizer (∂u/∂x + ∂v/∂y = 0) enforces incompressible flow
physics weakly. This:
- Prevents non-physical predictions (diverging/converging flow in open space)
- Acts as a physics-informed regularizer
- Weight λ=0.01 keeps it secondary to data fidelity

## Why Gaussian splatting for obs_to_grid?

Drone observations are point measurements. The model expects gridded input.
Options considered:
1. Nearest-neighbor: fast but discontinuous, artifacts at cell boundaries
2. Inverse distance weighting: better but sharp falloff
3. Gaussian splatting: smooth, differentiable, physically motivated (each
   observation has a "radius of influence" proportional to measurement uncertainty)

Gaussian splatting with σ=3 grid cells was chosen. The confidence channel
tells the model exactly where observations are dense vs sparse.

## Why A* street-following drone path?

The original lawnmower path flew straight lines regardless of buildings,
resulting in observations inside solid obstacle cells (meaningless). A* with
3-cell building clearance:
- Routes the drone through actual street corridors
- Samples only valid fluid cells
- Better mirrors real drone flight constraints
- Creates more physically informative training trajectories

The lawnmower path (`make_lawnmower_path`) is still available for comparison
but A* (`make_street_path`) is the default.

## Why triangle rasterization instead of bounding-box projection?

The original geometry code used axis-aligned bounding boxes per triangle,
causing diagonal/angular buildings (common in this STL model) to appear as
rectangles. Root cause: the second filter `up_verts.min() > slice_height + 5%`
skipped 100% of roof triangles (all buildings have roofs at Y=1135–2435mm,
all above the 851mm threshold at 30% of max Y).

Fix: removed the second filter, switched to `skimage.draw.polygon` for roof
triangles (which define the correct 2D footprint) and `skimage.draw.line` for
wall triangle edges. `binary_fill_holes` then fills building interiors.

Result: correct diagonal/angular building shapes, 27.5% solid fraction.

## Coordinate system and display orientation

The STL model is Y-up with:
- X: 0–10000mm (building length axis)
- Z: 0–11250mm (building width axis, slightly larger span)

Axis mapping chosen so that the visualization matches the STL viewer's
top-down orientation (Z horizontal, X vertical):
- h0 = Z → columns (horizontal display axis)
- h1 = X → rows, inverted (X_max at bottom row)

`invert_xaxis()` is applied to all spatial panels so that:
- col=0 (Z_min, LBM inlet with angle=0°) appears on the RIGHT of the display
- Wind flows visually from right to left (matching physical wind tunnel setup)

## LBM inlet side selection

The Zou-He BC has two forms:
- Left inlet (col=0, ux≥0): `rho = (f0+f2+f4 + 2*(f3+f6+f7)) / (1 - ux_in)`
- Right inlet (col=W-1, ux<0): `rho = (f0+f2+f4 + 2*(f1+f5+f8)) / (1 + ux_in)`

The solver auto-selects based on sign of ux_in, enabling wind direction to be
controlled by `--angle` without modifying the boundary condition logic separately.

## Why 256×256 grid (not 128)?

At 128×128, the building shapes are visible but the street corridors are only
a few cells wide, making the A* path and wind field look coarse. At 256×256:
- Building edges are noticeably sharper
- Street channels are wider in cell count (~8–15 cells vs ~4–7)
- The model has more spatial detail to learn from
- Still fast on RTX 5000 Ada (32GB VRAM handles it easily)

The FNO modes scale accordingly: `max(20, grid // 8)` = 32 at 256².

## Why 4-side Zou-He BCs (not just left/right)?

The original LBM only had left/right inlet BCs, which meant wind could only blow
horizontally (angle=0° or 180°). Adding top/bottom BCs enables:
- Any wind direction angle (0°–360°) via `--angle`
- Gradual rotation during collection via `--angle-end`
- Diagonal inlet conditions (e.g. 45°) activate two sides simultaneously

Zou-He rho is computed from the correct subset of known distributions for each
side. Corners are handled by letting the y-BC overwrite the x-BC (harmless since
both set equilibrium at the same (ux_in, uy_in)).

## Why is transient mode ON by default?

The original `--transient` flag defaulted to off, so steady wind was the default.
But steady LBM produces nearly identical snapshots — the neural network trains on
effectively duplicate data and the visualization looks static.

Transient (gusty) wind with two-frequency speed variation creates diverse
snapshots that better train the U-FNO and produce a more realistic, dynamic
visualization. The flag is now `--no-transient` to disable.

## Why negate U in quiver set_UVC()?

With `invert_xaxis()` on the plot axes, matplotlib's `quiver` still draws arrows
in the raw screen-right direction for positive U, ignoring the axis flip. This
caused arrows to point rightward (toward the inlet) instead of leftward (in the
flow direction) for right-to-left wind.

Fix: pass `-u` instead of `u` to `set_UVC()`. This makes the arrow appear
leftward on screen, correctly indicating where the wind is going. The V component
does not need negation — `origin='lower'` is handled correctly by quiver.

---

## Drone Sampling: 80 → 2400 Samples (June 2026)

The original training used `total_steps=80` drone samples per training example
(~8 seconds at 10 Hz). This was unrealistic — a real 4-minute survey leg at 10 Hz
yields 2400 samples. The fix was straightforward:

1. Changed `total_steps = 2400` in both `train_ufno.py` and `train_fm.py`
2. Vectorized `obs_to_grid()` — the old per-observation Python loop was O(N×patch²)
   and would take ~50ms at 2400 samples. Replaced with bilinear scatter
   (`np.add.at`) + `scipy.ndimage.gaussian_filter`, reducing to ~1ms.
3. Updated `evaluate_ufno.py` to match the new sampling protocol (old rolling-buffer
   approach couldn't reach 2400 obs within T=150 LBM snapshots).

---

## Repo Reorganization (June 2026)

Root-level scripts were moved to `scripts/` and `src/` was split into subpackages
(`data/`, `models/`, `training/`, `viz/`) to match a professional ML project layout.
Each `scripts/*.py` file adds `sys.path.insert(0, project_root)` to find `src`.

Phase 1 outputs were archived to `outputs/ufno/` to preserve the U-FNO baseline
while making `outputs/flow_matching/` the active output directory for Phase 2.

A `.gitignore` bug was also fixed: unanchored `data/` and `outputs/` patterns were
silently ignoring `src/data/` Python files. Fixed to `/data/` and `/outputs/` (root-only).

---

## Phase 2 Design Decisions (June 2026)

### Why flow matching instead of DDPM for Phase 2?

DDPM (the original Phase 2 plan) trains a noise prediction network and samples
via a 100–1000 step reverse diffusion chain. Flow matching instead learns a
velocity field that transports noise to data along a straight-line path:

    x_t = (1−t)·x_noise + t·x_data
    Loss = E[‖v_θ(x_t, t, c) − (x_data − x_noise)‖²]

Straight paths are easier to integrate numerically → 10–20 ODE steps suffice
vs 100–1000 for DDPM. For N=20 ensemble samples at each waypoint with a 30–60s
inference budget, this difference is critical.

Bayesian conditioning (DPS-style guidance) transfers identically from DDPM to
flow matching — the same gradient guidance logic applies to the ODE integrator
instead of the SDE denoiser. No architectural change needed.

### Why temporal forecasting (5-min obs → 5-min forecast) not snapshot prediction?

Phase 1 predicts a single future snapshot (one field at t+horizon). For drone
path planning, the planner needs to know how the wind evolves *during* the next
leg, not just at the start of it. The Phase 2 model outputs a full (T_out, H, W)
field sequence — the drone can compute risk against wind along its trajectory
at every future timestep, not just at t=0.

### Why DPS (Bayesian posterior sampling) for observation conditioning?

DPS conditions the generative model on drone observations at inference time
without any retraining per route:

    v_guided = v_θ(x_s, s, c_geo) − ρ·∇_{x_s}‖obs − H(x̂_1)‖²

H is the forward operator that extracts the model's predicted wind at the
drone's actual trajectory positions. The gradient is computed via autodiff
through H and the model. This means any drone path can be used as conditioning
input — no need to retrain when the route changes or when testing new paths.

### Why hard Leray projection + obstacle mask instead of soft physics losses?

Phase 1 uses a soft divergence penalty (λ·‖∇·u‖²) as a regularizer. For the
ensemble output in Phase 2, each sample must satisfy physics *exactly*, because
downstream exceedance probability calculations assume physically valid fields.

- Leray projection (2D FFT): u ← u − ∇(∇⁻²·∇·u) gives exact ∇·u = 0 in one
  pass. Cost is negligible (two FFTs per sample).
- Obstacle mask: multiply velocity channels by the fluid mask (0 inside buildings).
  Exact no-slip at zero cost. No soft loss can guarantee this.

Soft losses only push the prediction *toward* physics; hard constraints guarantee
it. Since projection and masking are applied post-generation, they don't interact
with the training objective.

### Why keep LBM for Phase 2 (not switch to URANS/LES)?

LES is inherently 3D — it has no well-defined 2D formulation. URANS requires a
separate CFD solver (OpenFOAM), mesh generation, turbulence model tuning, and
wall-clock days per run at scale. This is a major infrastructure change.

The existing PyTorch LBM runs on our GPU, takes minutes per condition, and
already produces physically consistent transient unsteady flow with vortex
shedding and turbulent-like fluctuations. It is sufficient to validate the Phase
2 architecture. OpenFOAM data replaces LBM in Phase 4.

### Why pixel space (not latent space) for Phase 2 start?

At 256×256 × T_out~30, each training sample is ~18M values. A spatiotemporal
U-Net operating in pixel space is large but within the 32GB VRAM budget of the
RTX 5000 Ada for reasonable batch sizes. Adding a VAE (latent space) introduces
a second model to train and validate.

Pixel space is simpler and avoids the two-stage training complexity. Latent
space is added only if memory or inference benchmarking shows it is needed.

### Why resolution-aware attention gating?

`SpatialAttnBlock` has O(H²W²) memory cost for the attention matrix. At 512×512:
- Level 2 feature map (128×128): 16,384 tokens → 1.1 GB/head → OOM during backward
- Level 3 (64×64): 4,096 tokens → 67 MB/head → feasible

Fix: `attn_start = max(0, ceil(log2(grid_size / 64)))` so attention is only
instantiated at levels where the feature map is ≤64×64. For grid=512: attn_start=3
means only the deepest encoder level (64×64) and bottleneck (32×32) have attention.
This is resolution-adaptive — smaller grids automatically get more attention levels.

### Why AMP + gradient checkpointing for Phase 2 training?

At 512×512 with T_out=10, the backward pass stores activations for all layers.
Level 0 features alone: [B*T, hidden, 512, 512] = [4*10, 32, 512, 512] → ~2.7 GB.
Summing over all levels: ~8–10 GB, plus gradients doubling this.

- **AMP (float16)**: halves activation memory; negligible effect on loss convergence
  for this regression task. Enabled by default via `torch.amp.autocast`.
- **Gradient checkpointing**: recomputes activations during backward instead of
  storing them; ~4× activation memory saving at the cost of ~30% more compute.
  Applied per encoder/decoder level via `torch.utils.checkpoint.checkpoint`.

Result: hidden=32, batch=4, T_out=10 → 23.7 GB peak (fits in 32 GB with headroom).

### Why GNN trajectory encoder is deferred?

The current grid-splatting approach (obs_u, obs_v, confidence channels) places
drone observations at their grid cell locations and is already working in Phase 1.
A GNN encoder handles irregular path geometry more naturally but adds a new model
component and training dependency. The added complexity is only justified if
ablation studies show the grid splatting is the accuracy bottleneck.

### Why SDF geometry channel in addition to the binary mask?

The binary obstacle mask is a step function — zero gradient everywhere except
a one-pixel discontinuity at building walls. A signed distance field (positive
outside, negative inside, computed via `scipy.ndimage.distance_transform_edt`)
gives the network a smooth proximity-to-wall signal everywhere, which is
exactly the quantity that matters for near-wall flow behavior (boundary layers,
wake formation). Cost is negligible — one EDT call per geometry, cached once
in `WindSequenceDataset.__init__` since the city geometry is fixed across the
whole dataset. `GeometryEncoder` input channels: 1 → 2 (mask + SDF); its output
(8 channels into the U-Net) is unchanged, so this is a fully localized change.

### Why a physics-informed prior instead of pure Gaussian noise for flow matching?

Flow matching only requires that you can sample from the source distribution
and compute `x_target − x_noise`; the source doesn't have to be Gaussian. Pure
noise means the network has to learn the entire field — geometry-driven mean
flow and obstacle wakes — from scratch at every ODE step. Instead,
`FlowMatchingModel.physics_prior()` builds a confidence-weighted ambient (u, v)
estimate from the sparse drone observations, broadcasts it to a uniform field,
zeroes it inside obstacles, and Leray-projects it to divergence-free (reusing
the existing projection function, not a new mechanism). Gaussian noise is
still added on top so the prior remains a proper distribution — ensemble
diversity at inference still comes from sampling, not a deterministic prior.

This is strictly additive: `use_physics_prior` defaults to `True` but can be
disabled (`--no-physics-prior`) for ablation, and inference reads the flag
from the training checkpoint so train/sample distributions always match (flow
matching requires the same source distribution at training and sampling time).

### Why an ensemble calibration utility (`src/evaluation/calibration.py`)?

DPS-guided ensembles only support risk-aware path planning if the ensemble
spread is trustworthy — i.e. it's larger exactly where the prediction is more
wrong. This wasn't being measured anywhere. `spread_skill()` (spread-error
correlation) and `coverage()` (does the empirical N% interval actually contain
truth N% of the time) are cheap numpy functions, wired into `infer_fm.py`'s
output. No architecture change, just an evaluation gap that's now closed.

### Why divergence + solid-boundary penalties added to the training loss (not just inference)?

A divergence-residual diagnostic added to `infer_fm.py` (see below) confirmed
a real gap: the post-hoc Leray projection at inference is exact only for a
periodic domain with no internal obstacles. Masking solid cells *after*
projecting reintroduces divergence right at building edges — exactly where
flow physics (separation, wake formation) matters most. A synthetic test
confirmed this concretely: a perfectly divergence-free field, masked the same
way the pipeline does, showed near-obstacle divergence residual ~47x the
interior residual (1.79 vs. 0.0).

The previous design relied entirely on two implicit/post-hoc mechanisms: (1)
training data being divergence-free, so MSE training tends to absorb the
constraint as an emergent property, and (2) a one-shot Leray projection
applied after sampling. Neither gives the network an explicit reason to
produce divergence-free, wall-respecting output on its own — and solid cells
were *entirely excluded* from the training loss (zero gradient signal about
what to predict there), which is precisely why the post-hoc zero-masking step
creates a sharp seam at building boundaries.

Fix: two soft physics penalties added directly to `flow_match_loss()`,
computed on the model's own one-step clean-field estimate
`x_hat1 = x_t + (1-s)·v_pred` — the same quantity DPS guidance already
computes at inference, so no new concept was introduced:
- `lambda_div` (default 0.1): mean squared divergence of `x_hat1` in fluid cells
- `lambda_solid` (default 0.1): mean squared velocity of `x_hat1` at solid
  cells (no-penetration boundary condition)

This doesn't make either constraint exact — it's a soft penalty, not a hard
architectural guarantee — but it gives the network direct gradient signal to
internalize both constraints rather than relying purely on data resemblance
plus a post-hoc fix that's now known to leak at exactly the locations that
matter physically. The existing post-hoc Leray projection + masking at
inference stays as a backstop; the training-time penalties are meant to
shrink how much correction that backstop has to make, especially near walls.

A smoke test (1 epoch, 2 conditions) showed the component loss breakdown
staying well-balanced: `Train: 0.0366 (data=0.0316 div=0.0109 solid=0.0395)`
— the data term still dominates at lambda=0.1, with both penalties
contributing meaningfully rather than being negligible or swamping the fit.
This was judged a large enough change to the training objective to restart
the in-progress 200-epoch run from scratch (was at epoch 30) rather than
resume — resuming would mix ~30 epochs optimized against the old objective
with the new one. The pre-change checkpoint was archived to
`outputs/flow_matching/fm_model_pre_physics_loss.pth` rather than discarded.

### Why a divergence-residual diagnostic in calibration.py / infer_fm.py?

`infer_fm.py` claimed (informally, in comments) to produce divergence-free
output via the Leray projection, but nothing actually measured whether that
held — calibration.py only covered spread-skill and interval coverage.
Checking the code directly confirmed the projection is applied globally
*before* obstacle masking (`infer_fm.py`), which can reintroduce divergence
at obstacle boundaries that a single aggregate number would hide.
`divergence_residual()` reports both a global mean and a near-obstacle vs.
interior split (via `scipy.ndimage.binary_dilation` on the obstacle mask), so
the boundary-localized failure mode is directly visible rather than averaged
away. This is inference-time instrumentation only — it doesn't change model
behavior, just makes an existing (previously unverified) claim checkable on
every run. It directly motivated the training-loss change above.

### Bug fix: DPS guidance was reading the wrong observation channels

`obs` channels are `[mask, obs_u, obs_v, confidence, x, y]` (channel 0 is
geometry, not data). `FlowMatchingModel.sample()`'s DPS guidance term read
`obs_uv = obs[:, :2]` and `conf = obs[:, 2:3]` — i.e. `[mask, obs_v]` as the
"measured u,v" and `obs_v` as "confidence". Every ensemble inference run prior
to this fix was guiding generation toward the wrong target. Fixed to
`obs[:, 1:3]` / `obs[:, 3:4]`. Caught while implementing the physics prior,
which required correctly identifying the same channel layout.

### Why 15-minute wind stability matters for training design?

Atmospheric boundary conditions (wind speed and direction) are approximately
stable for ~15-minute windows in the operational environment. The full training
window (5-min obs + 5-min forecast = 10 min) fits within one stable period.

This means:
- The model is never asked to handle BC changes during a prediction window
- Each LBM run at a fixed (angle, speed) gives many clean (obs, forecast) pairs
  — the simplest possible training data structure
- Turbulent fluctuations (the model's primary uncertainty source) occur on top
  of a stable mean, making the forecast problem well-conditioned

If BC changes become relevant (e.g. wind shifts at 10-min scale), training data
and the model framing would need to change. For now, stable BCs are assumed.

---

## Why m/s labels on colorbars?

LBM units (dimensionless speeds 0.0–0.25) are not immediately interpretable
for domain scientists. Converting via `lbm_to_ms = ref_speed / inlet_speed`
makes the dashboard readable:
- A domain expert immediately knows 10 m/s is a moderate urban wind
- Errors of 0.5 m/s are tangible compared to "0.004 LBM units"
- FuncFormatter is used so underlying data stays in LBM units throughout

The reference speed (default 10 m/s) can be overridden with `--ref-speed`.
