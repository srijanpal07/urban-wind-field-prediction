"""
Ensemble inference script for the spatiotemporal Flow-Matching model.

Loads geometry, runs the LBM solver to get ground truth, simulates a drone
observation pass (2400 samples, same protocol as training), then draws an
ensemble of N future wind-sequence samples from FlowMatchingModel using
DPS-guided flow-matching sampling. Each ensemble member is divergence-cleaned
via a Leray projection and masked to zero inside buildings.

Usage:
    python scripts/infer_fm.py --stl data/city_model.STL
    python scripts/infer_fm.py --stl data/city_model.STL --angle 45 --speed 0.08
    python scripts/infer_fm.py --stl data/city_model.STL --n-samples 20 --n-steps 20 --rho 0.5
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

import numpy as np
import torch

REF_LBM_SPEED = 0.08   # LBM speed that corresponds to --ref-speed m/s
_TOTAL_STEPS = 2400    # drone samples per observation window (matches training)
_OBS_WINDOW = 30       # LBM snapshots spanned by the drone observation window


def main():
    parser = argparse.ArgumentParser(description='Flow-Matching ensemble inference')
    parser.add_argument('--stl', type=str, default=None)
    parser.add_argument('--model', type=str, default='outputs/flow_matching/fm_model.pth')
    parser.add_argument('--angle', type=float, default=None,
                         help='Inlet wind angle in degrees (random in [0,360) if not set)')
    parser.add_argument('--speed', type=float, default=None,
                         help='LBM inlet speed in [0.02, 0.10] (random if not set)')
    parser.add_argument('--ref-speed', type=float, default=5.0,
                         help='Physical wind speed (m/s) corresponding to LBM speed 0.08')
    parser.add_argument('--grid', type=int, default=None,
                         help='Grid resolution (default: read from model checkpoint)')
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--n-samples', type=int, default=20,
                         help='Number of ensemble members to draw')
    parser.add_argument('--n-steps', type=int, default=20,
                         help='Number of flow-matching ODE integration steps')
    parser.add_argument('--rho', type=float, default=0.5,
                         help='DPS guidance strength')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None,
                         help='RNG seed for random angle/speed draw')
    parser.add_argument('--save', type=str, default=None,
                         help='Save the summary figure to this path instead of showing it')
    parser.add_argument('--no-physics-prior', action='store_true',
                         help='Override the checkpoint and sample from plain Gaussian '
                              'noise instead of the physics prior (ablation only — '
                              'must match how the loaded model was trained)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    rng = np.random.default_rng(args.seed)
    angle = args.angle if args.angle is not None else float(rng.uniform(0, 360))
    speed = args.speed if args.speed is not None else float(rng.uniform(0.02, 0.10))

    lbm_to_ms = args.ref_speed / REF_LBM_SPEED
    physical_inlet = speed * lbm_to_ms

    print(f"{'='*55}")
    print(f" Flow-Matching Ensemble Inference")
    print(f" Device        : {device}")
    print(f" Angle         : {angle:.1f} deg")
    print(f" LBM speed     : {speed:.4f}  ->  {physical_inlet:.2f} m/s physical inlet")
    print(f" Ensemble size : {args.n_samples}  |  ODE steps: {args.n_steps}  |  rho: {args.rho}")
    print(f"{'='*55}\n")

    # ── Load model first — need grid_size before running LBM ─────────────────
    print(f"[1/4] Loading model from {args.model}")
    if not os.path.exists(args.model):
        print(f"      Model not found: {args.model}")
        print("      Train first:  python scripts/train_fm.py")
        return

    from src.models.flow_matching import FlowMatchingModel
    ckpt = torch.load(args.model, map_location=device)
    T_out = ckpt.get('T_out', 20)
    grid_size = args.grid if args.grid is not None else ckpt.get('grid_size', 256)
    hidden = ckpt.get('hidden', 64)
    n_levels = ckpt.get('n_levels', 4)
    t_emb_dim = ckpt.get('t_emb_dim', 256)
    use_physics_prior = ckpt.get('use_physics_prior', True) and not args.no_physics_prior

    model = FlowMatchingModel(T_out=T_out, obs_channels=6, geo_channels=8,
                               geo_in_channels=2, hidden=hidden, n_levels=n_levels,
                               t_emb_dim=t_emb_dim,
                               grid_size=grid_size).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"      Loaded  (T_out={T_out}, grid={grid_size}, hidden={hidden}, "
          f"n_levels={n_levels}, physics_prior={use_physics_prior})")

    # ── Geometry ──────────────────────────────────────────────────────────────
    from src.data.geometry import stl_to_obstacle_mask, make_synthetic_city

    if args.stl and os.path.exists(args.stl):
        print(f"[2/4] Loading geometry from {args.stl}")
        obstacle_mask, _ = stl_to_obstacle_mask(args.stl, grid_size=grid_size)
    else:
        print("[2/4] No STL found — using synthetic city")
        obstacle_mask = make_synthetic_city(grid_size=grid_size, seed=42)

    # ── LBM ground truth ──────────────────────────────────────────────────────
    print(f"[3/4] Running LBM  (warmup={args.warmup}, collect={args.steps}, grid={grid_size})...")
    from src.data.lbm_solver import LBMSolver

    solver = LBMSolver(obstacle_mask, inlet_speed=speed,
                        inlet_angle=angle, tau=0.7)
    u_arr, v_arr = solver.run(n_warmup=args.warmup, n_collect=args.steps,
                               collect_every=3, transient=True)
    T, H, W = u_arr.shape
    print(f"      Wind field: {u_arr.shape}")

    t0 = max(0, T - _OBS_WINDOW - T_out)
    t_seq_start = t0 + _OBS_WINDOW
    t_end = min(t_seq_start + T_out, T)

    # ── Drone observations (2400 samples, same protocol as training) ─────────
    print(f"[4/4] Simulating drone observation pass ({_TOTAL_STEPS} samples)...")
    from src.data.drone_sampler import DroneSampler

    sampler = DroneSampler(grid_size=grid_size, obstacle_mask=obstacle_mask)
    waypoints = sampler.make_traverse_path(seed=int(rng.integers(0, 100_000)))
    x_path, y_path = sampler.interpolate_path(waypoints, _TOTAL_STEPS)

    noise_scale = grid_size * 0.02
    x_path = np.clip(x_path + rng.normal(0, noise_scale, _TOTAL_STEPS), 0, grid_size - 1)
    y_path = np.clip(y_path + rng.normal(0, noise_scale, _TOTAL_STEPS), 0, grid_size - 1)

    t_indices = np.linspace(t0, t0 + _OBS_WINDOW - 1, _TOTAL_STEPS).astype(int)
    obs = sampler.sample_field(u_arr, v_arr, x_path, y_path, t_indices)
    obs_u_g, obs_v_g, conf_g = sampler.obs_to_grid(obs, grid_size, sigma=3.0)

    ys = np.linspace(0, 1, H)
    xs = np.linspace(0, 1, W)
    xg, yg = np.meshgrid(xs, ys)

    obs_channels = np.stack([
        obstacle_mask.astype(np.float32),
        obs_u_g.astype(np.float32),
        obs_v_g.astype(np.float32),
        conf_g.astype(np.float32),
        xg.astype(np.float32),
        yg.astype(np.float32),
    ], axis=0)  # [6, H, W]

    from src.data.geometry import build_geo_channels

    obs_t = torch.tensor(obs_channels[None], dtype=torch.float32, device=device)
    mask_t = torch.tensor(build_geo_channels(obstacle_mask)[None], device=device)
    solid_mask_t = torch.tensor(obstacle_mask, dtype=torch.bool, device=device)

    # ── Ground truth target sequence ──────────────────────────────────────────
    u_seq_true = u_arr[t_seq_start:t_end].astype(np.float32)
    v_seq_true = v_arr[t_seq_start:t_end].astype(np.float32)
    n_have = u_seq_true.shape[0]
    if n_have < T_out:
        pad = T_out - n_have
        u_seq_true = np.concatenate([u_seq_true, np.repeat(u_seq_true[-1:], pad, axis=0)], axis=0)
        v_seq_true = np.concatenate([v_seq_true, np.repeat(v_seq_true[-1:], pad, axis=0)], axis=0)

    # ── Ensemble sampling ──────────────────────────────────────────────────────
    print(f"\nSampling {args.n_samples} ensemble members "
          f"({args.n_steps} ODE steps, rho={args.rho})...")
    samples = model.sample(obs_t, mask_t, n_samples=args.n_samples,
                            n_steps=args.n_steps, rho=args.rho, device=device,
                            solid_mask=solid_mask_t, use_physics_prior=use_physics_prior)
    # samples: [n_samples, 2, T_out, H, W]

    u_samp = samples[:, 0]  # [n_samples, T_out, H, W]
    v_samp = samples[:, 1]

    u_proj, v_proj = FlowMatchingModel.leray_project(u_samp, v_samp)

    fluid = torch.tensor(~obstacle_mask, device=device).float()[None, None]
    u_proj = u_proj * fluid
    v_proj = v_proj * fluid

    u_proj_np = u_proj.detach().cpu().numpy()  # [n_samples, T_out, H, W]
    v_proj_np = v_proj.detach().cpu().numpy()

    u_mean = u_proj_np.mean(axis=0)  # [T_out, H, W]
    v_mean = v_proj_np.mean(axis=0)
    u_std = u_proj_np.std(axis=0)
    v_std = v_proj_np.std(axis=0)
    spread = np.sqrt(u_std ** 2 + v_std ** 2)  # [T_out, H, W]

    fluid_mask_np = ~obstacle_mask
    rmse_per_member = []
    for i in range(args.n_samples):
        diff = ((u_proj_np[i] - u_seq_true) ** 2 + (v_proj_np[i] - v_seq_true) ** 2)
        rmse = np.sqrt(diff[:, fluid_mask_np].mean()) * lbm_to_ms
        rmse_per_member.append(rmse)

    mean_diff = ((u_mean - u_seq_true) ** 2 + (v_mean - v_seq_true) ** 2)
    mean_rmse = np.sqrt(mean_diff[:, fluid_mask_np].mean()) * lbm_to_ms
    mean_spread = spread[:, fluid_mask_np].mean() * lbm_to_ms

    from src.evaluation.calibration import spread_skill, coverage

    ensemble_uv = np.stack([u_proj_np, v_proj_np], axis=1)       # [N, 2, T, H, W]
    truth_uv = np.stack([u_seq_true, v_seq_true], axis=0)        # [2, T, H, W]
    calib = spread_skill(ensemble_uv, truth_uv, fluid_mask_np)
    cov90 = coverage(ensemble_uv, truth_uv, fluid_mask_np, interval=0.9)

    print(f"\n{'='*55}")
    print(f" Ensemble Statistics  (lbm_to_ms = {lbm_to_ms:.1f})")
    print(f" Per-member RMSE  : mean={np.mean(rmse_per_member):.4f}  "
          f"std={np.std(rmse_per_member):.4f}  (m/s)")
    print(f" Ensemble-mean RMSE vs ground truth : {mean_rmse:.4f} m/s")
    print(f" Ensemble spread (avg sigma)        : {mean_spread:.4f} m/s")
    print(f" Spread-error correlation           : {calib['correlation']:.3f}  "
          f"(want > 0 — spread should track where the model is wrong)")
    print(f" 90% interval coverage              : {cov90:.3f}  "
          f"(want ~0.90 — <<0.90 overconfident, >>0.90 underconfident)")
    print(f"{'='*55}\n")

    # ── Visualization ──────────────────────────────────────────────────────────
    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_show = 0  # show first predicted frame
    spd_mean = np.sqrt(u_mean[t_show] ** 2 + v_mean[t_show] ** 2) * lbm_to_ms
    spd_true = np.sqrt(u_seq_true[t_show] ** 2 + v_seq_true[t_show] ** 2) * lbm_to_ms
    spread_show = spread[t_show] * lbm_to_ms

    spd_mean_disp = np.where(obstacle_mask, np.nan, spd_mean)
    spd_true_disp = np.where(obstacle_mask, np.nan, spd_true)
    spread_disp = np.where(obstacle_mask, np.nan, spread_show)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0d1117')
    titles = ['Ground Truth (m/s)', 'Ensemble Mean (m/s)', 'Ensemble Spread sigma (m/s)']
    data = [spd_true_disp, spd_mean_disp, spread_disp]
    cmaps = ['RdYlBu_r', 'RdYlBu_r', 'YlOrRd']

    for ax, title, d, cmap in zip(axes, titles, data, cmaps):
        ax.set_facecolor('#161b22')
        im = ax.imshow(d, origin='lower', cmap=cmap, interpolation='bilinear')
        ax.set_title(title, color='#e6edf3', fontsize=10, fontfamily='monospace')
        ax.tick_params(colors='#8b949e', labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f'Flow-Matching Ensemble — angle={angle:.0f} deg, '
                 f'speed={physical_inlet:.2f} m/s, mean RMSE={mean_rmse:.3f} m/s',
                 color='#e6edf3', fontsize=12, fontfamily='monospace')
    plt.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        plt.savefig(args.save, dpi=120, bbox_inches='tight', facecolor='#0d1117')
        print(f"Figure saved → {args.save}")
    else:
        try:
            plt.show()
        except Exception as e:
            fallback = 'outputs/flow_matching/infer_fm_result.png'
            print(f"Interactive display failed ({type(e).__name__}: {e})")
            os.makedirs(os.path.dirname(fallback) or '.', exist_ok=True)
            plt.savefig(fallback, dpi=120, bbox_inches='tight', facecolor='#0d1117')
            print(f"Saved to {fallback}")


if __name__ == '__main__':
    main()
