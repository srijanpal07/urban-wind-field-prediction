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

## Why m/s labels on colorbars?

LBM units (dimensionless speeds 0.0–0.25) are not immediately interpretable
for domain scientists. Converting via `lbm_to_ms = ref_speed / inlet_speed`
makes the dashboard readable:
- A domain expert immediately knows 10 m/s is a moderate urban wind
- Errors of 0.5 m/s are tangible compared to "0.004 LBM units"
- FuncFormatter is used so underlying data stays in LBM units throughout

The reference speed (default 10 m/s) can be overridden with `--ref-speed`.
