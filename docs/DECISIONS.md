# DECISIONS.md — Design Decisions & Rationale

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

### Why GNN trajectory encoder is deferred?

The current grid-splatting approach (obs_u, obs_v, confidence channels) places
drone observations at their grid cell locations and is already working in Phase 1.
A GNN encoder handles irregular path geometry more naturally but adds a new model
component and training dependency. The added complexity is only justified if
ablation studies show the grid splatting is the accuracy bottleneck.

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
