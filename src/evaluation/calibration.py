"""
Calibration metrics for the flow-matching ensemble forecast.

Ensemble spread is only useful for risk-aware path planning if it actually
tracks where the model is wrong. These functions turn an ensemble of
[N, ..., H, W] samples plus a single ground-truth field into quantitative
calibration numbers: does spread correlate with error, and does the
ensemble's empirical interval contain the truth as often as it should?
"""

import numpy as np


def spread_skill(ensemble: np.ndarray, truth: np.ndarray, fluid_mask: np.ndarray) -> dict:
    """
    Pixel-wise ensemble std (spread) vs |ensemble_mean - truth| (error).

    ensemble: [N, ..., H, W]
    truth:    [..., H, W]      (same trailing shape as ensemble minus N)
    fluid_mask: [H, W] bool, True = include (non-solid cells)

    A well-calibrated ensemble is more uncertain exactly where it is more
    wrong, i.e. correlation should be positive and not small.
    """
    mean = ensemble.mean(axis=0)
    spread = ensemble.std(axis=0)
    error = np.abs(mean - truth)

    fm = np.broadcast_to(fluid_mask, error.shape[-2:])
    spread_flat = spread[..., fm].reshape(-1)
    error_flat = error[..., fm].reshape(-1)
    corr = float(np.corrcoef(spread_flat, error_flat)[0, 1])

    return {'spread': spread_flat, 'error': error_flat, 'correlation': corr}


def coverage(ensemble: np.ndarray, truth: np.ndarray, fluid_mask: np.ndarray,
             interval: float = 0.9) -> float:
    """
    Empirical coverage: fraction of (component, fluid-cell, ...) locations
    where `truth` falls within the ensemble's central `interval` (e.g. 0.9 ->
    5th-95th percentile band across the N members).

    Coverage << interval -> ensemble overconfident (spread too small).
    Coverage >> interval -> ensemble underconfident (spread too large).
    A calibrated ensemble has coverage ~= interval.
    """
    lo_q, hi_q = (1 - interval) / 2, 1 - (1 - interval) / 2
    lo = np.quantile(ensemble, lo_q, axis=0)
    hi = np.quantile(ensemble, hi_q, axis=0)

    fm = np.broadcast_to(fluid_mask, truth.shape[-2:])
    inside = (truth >= lo) & (truth <= hi)
    return float(inside[..., fm].mean())


def reliability_curve(ensemble: np.ndarray, truth: np.ndarray, fluid_mask: np.ndarray,
                       intervals=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)) -> list:
    """(nominal, empirical) coverage pairs for a reliability diagram — a
    calibrated ensemble lies on y=x."""
    return [(p, coverage(ensemble, truth, fluid_mask, interval=p)) for p in intervals]


def divergence_residual(u: np.ndarray, v: np.ndarray, fluid_mask: np.ndarray,
                         obstacle_mask: np.ndarray = None, boundary_width: int = 2) -> dict:
    """
    Mean |div u| = |du/dx + dv/dy| (finite differences, grid-cell units) over
    fluid cells, as a check on how well the Leray projection's
    divergence-free guarantee survives obstacle masking.

    u, v: [..., H, W]
    fluid_mask: [H, W] bool, True = fluid
    obstacle_mask: [H, W] bool, True = solid. If given, also splits the
        residual into cells within `boundary_width` cells of an obstacle vs.
        the open interior. The Leray projection is an exact divergence-free
        projection only for a periodic domain with no internal obstacles —
        zeroing solid cells *after* projecting can reintroduce divergence
        right at building edges even when the open interior is clean, so a
        single aggregate number can hide a boundary-localized problem.
    """
    div = np.gradient(u, axis=-1) + np.gradient(v, axis=-2)
    out = {'mean_abs_div': float(np.abs(div[..., fluid_mask]).mean())}

    if obstacle_mask is not None:
        from scipy.ndimage import binary_dilation
        near_obstacle = binary_dilation(obstacle_mask, iterations=boundary_width) & fluid_mask
        interior = fluid_mask & ~near_obstacle
        if near_obstacle.any():
            out['mean_abs_div_near_obstacle'] = float(np.abs(div[..., near_obstacle]).mean())
        if interior.any():
            out['mean_abs_div_interior'] = float(np.abs(div[..., interior]).mean())

    return out
