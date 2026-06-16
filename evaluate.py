"""
Evaluation script: measure model accuracy across wind conditions.

Two modes of operation:

  (A) Random on-the-fly evaluation (default):
      Generates N random (angle, speed) conditions via LBM and scores the model.
      Angles and speeds may overlap with training conditions — useful for quick
      sanity checks but not a rigorous held-out benchmark.

  (B) Held-out test-set evaluation (--test-data):
      Loads pre-generated data from generate_data.py's test pass.
      Angles and speeds are guaranteed to be absent from the training set —
      a clean measure of generalisation to unseen conditions.
      Run:  python generate_data.py --stl data/city_model.STL  (generates both files)
      Then: python evaluate.py --stl data/city_model.STL --test-data data/lbm_test.npz

Metrics (all in m/s, using a fixed physical scale where LBM 0.08 = ref_speed):
  - Vector RMSE : sqrt(mean((u_pred-u_true)^2 + (v_pred-v_true)^2)) over fluid cells
  - Speed MAE   : mean(|speed_pred - speed_true|) over fluid cells
  - Direction error : mean angle between predicted and true wind vector (degrees)

Usage:
    # Quick sanity check (random conditions, may overlap training):
    python evaluate.py --stl data/city_model.STL

    # Rigorous held-out evaluation:
    python evaluate.py --stl data/city_model.STL --test-data data/lbm_test.npz

    # More random conditions:
    python evaluate.py --stl data/city_model.STL --n 20
"""

import argparse
import os

import numpy as np
import torch

REF_LBM_SPEED = 0.08   # LBM speed that maps to --ref-speed m/s (fixed scale)


def _build_obs_grid(obs_buffer, sampler, H, sigma=4.0):
    """Build Gaussian-splatted observation grid from rolling buffer."""
    if not obs_buffer:
        return (np.zeros((H, H)), np.zeros((H, H)), np.zeros((H, H)))
    xs = np.array([o['x'] for o in obs_buffer])
    ys = np.array([o['y'] for o in obs_buffer])
    us = np.array([o['u'] for o in obs_buffer])
    vs = np.array([o['v'] for o in obs_buffer])
    obs = dict(x=xs, y=ys, u_obs=us, v_obs=vs)
    return sampler.obs_to_grid(obs, H, sigma=sigma)


def evaluate_condition(model, obstacle_mask, u_arr, v_arr,
                       device, grid_size, horizon, lbm_to_ms, rng,
                       obs_window=15, pred_every=5):
    """
    Fly a drone over one wind condition and accumulate prediction metrics.
    Returns dict with vector_rmse, speed_mae, dir_error (all in m/s / degrees).
    """
    from src.drone_sampler import DroneSampler
    from src.model import prepare_input

    T, H, W = u_arr.shape
    sampler  = DroneSampler(grid_size=grid_size, obstacle_mask=obstacle_mask)
    waypoints = sampler.make_traverse_path(seed=int(rng.integers(0, 100_000)))
    x_path, y_path = sampler.interpolate_path(waypoints, T)

    obs_buffer   = []
    vec_rmses    = []
    speed_maes   = []
    dir_errors   = []
    n_obs_per_pred = 80   # match training (total_steps=80, full traverse)
    min_obs_needed = n_obs_per_pred

    for t in range(T - horizon):
        xi   = x_path[t]
        yi   = y_path[t]
        xi_i = int(np.clip(xi, 0, W - 1))
        yi_i = int(np.clip(yi, 0, H - 1))
        if not obstacle_mask[yi_i, xi_i]:
            u_t = u_arr[t, yi_i, xi_i]
            v_t = v_arr[t, yi_i, xi_i]
            spd_t = float(np.sqrt(u_t**2 + v_t**2))
            ang_t = float(np.arctan2(v_t, u_t))
            spd_n = max(0.0, spd_t + float(rng.normal(0, sampler.noise_speed_std)))
            ang_n = ang_t + np.deg2rad(float(rng.normal(0, sampler.noise_angle_std)))
            obs_buffer.append({'x': xi, 'y': yi,
                               'u': spd_n * np.cos(ang_n),
                               'v': spd_n * np.sin(ang_n)})

        if len(obs_buffer) < min_obs_needed or t % pred_every != 0:
            continue

        recent = obs_buffer[-n_obs_per_pred:]
        obs_u_g, obs_v_g, conf_g = _build_obs_grid(recent, sampler, H)

        x_in = prepare_input(obs_u_g, obs_v_g, conf_g, obstacle_mask, device=device)

        with torch.no_grad():
            u_p, v_p, _, _ = model(x_in)

        u_pred = u_p[0, 0].cpu().numpy()
        v_pred = v_p[0, 0].cpu().numpy()

        t_target = min(t + horizon, T - 1)
        u_true   = u_arr[t_target]
        v_true   = v_arr[t_target]

        fluid  = ~obstacle_mask
        u_tf   = u_true[fluid]
        v_tf   = v_true[fluid]
        u_pf   = u_pred[fluid]
        v_pf   = v_pred[fluid]

        vec_rmse = float(np.sqrt(np.mean((u_pf - u_tf) ** 2 +
                                          (v_pf - v_tf) ** 2))) * lbm_to_ms
        spd_true = np.sqrt(u_tf ** 2 + v_tf ** 2)
        spd_pred = np.sqrt(u_pf ** 2 + v_pf ** 2)
        spd_mae  = float(np.mean(np.abs(spd_pred - spd_true))) * lbm_to_ms

        valid = spd_true > 0.005
        if valid.sum() > 10:
            eps  = 1e-10
            u_tn = u_tf[valid] / (spd_true[valid] + eps)
            v_tn = v_tf[valid] / (spd_true[valid] + eps)
            u_pn = u_pf[valid] / (spd_pred[valid] + eps)
            v_pn = v_pf[valid] / (spd_pred[valid] + eps)
            dot  = np.clip(u_tn * u_pn + v_tn * v_pn, -1.0, 1.0)
            dir_err = float(np.mean(np.degrees(np.arccos(dot))))
        else:
            dir_err = float('nan')

        vec_rmses.append(vec_rmse)
        speed_maes.append(spd_mae)
        dir_errors.append(dir_err)

    return {
        'vector_rmse': float(np.mean(vec_rmses))     if vec_rmses  else float('nan'),
        'speed_mae':   float(np.mean(speed_maes))    if speed_maes else float('nan'),
        'dir_error':   float(np.nanmean(dir_errors)) if dir_errors else float('nan'),
    }


def _print_summary(results, seed_label, lbm_to_ms):
    vec_rmses  = [r['vector_rmse'] for r in results if not np.isnan(r['vector_rmse'])]
    speed_maes = [r['speed_mae']   for r in results if not np.isnan(r['speed_mae'])]
    dir_errs   = [r['dir_error']   for r in results if not np.isnan(r['dir_error'])]

    print(f"\nSummary over {len(results)} conditions  "
          f"({seed_label}, lbm_to_ms={lbm_to_ms:.1f}):\n")
    print(f"  {'Metric':<22} {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'-'*64}")

    def row(name, vals, unit):
        if vals:
            print(f"  {name:<22} {np.mean(vals):>9.4f}{unit}  "
                  f"{np.std(vals):>9.4f}{unit}  "
                  f"{np.min(vals):>9.4f}{unit}  "
                  f"{np.max(vals):>9.4f}{unit}")
        else:
            print(f"  {name:<22} {'N/A':>10}")

    row('Vector RMSE (m/s)',    vec_rmses,  ' ')
    row('Speed MAE (m/s)',      speed_maes, ' ')
    row('Direction error (°)',  dir_errs,   ' ')


def main():
    parser = argparse.ArgumentParser(description='Evaluate WindFNO across wind conditions')
    parser.add_argument('--stl',       type=str,   default=None)
    parser.add_argument('--model',     type=str,   default='outputs/wind_fno.pth')
    parser.add_argument('--test-data', type=str,   default=None,
                        help='Path to held-out test set from generate_data.py '
                             '(e.g. data/lbm_test.npz). When set, evaluates all '
                             'conditions in the file; --n and --seed are ignored.')
    parser.add_argument('--n',         type=int,   default=10,
                        help='Number of random conditions (ignored when --test-data is set)')
    parser.add_argument('--grid',      type=int,   default=None,
                        help='Grid resolution (default: read from model checkpoint)')
    parser.add_argument('--warmup',    type=int,   default=1000,
                        help='LBM warmup steps (ignored when --test-data is set)')
    parser.add_argument('--steps',     type=int,   default=150,
                        help='LBM snapshots per condition (ignored when --test-data is set)')
    parser.add_argument('--ref-speed', type=float, default=5.0,
                        help='Physical speed (m/s) corresponding to LBM speed 0.08')
    parser.add_argument('--horizon',   type=int,   default=10)
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--device',    type=str,   default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    rng    = np.random.default_rng(args.seed)

    lbm_to_ms = args.ref_speed / REF_LBM_SPEED

    test_mode = args.test_data is not None
    mode_label = 'HELD-OUT TEST SET' if test_mode else f'{args.n} random conditions'

    print(f"{'='*70}")
    print(f" Urban Wind Field Model Evaluation")
    print(f" Device     : {device}")
    print(f" Mode       : {mode_label}")
    if test_mode:
        print(f" Test file  : {args.test_data}")
    print(f" Speed scale: LBM 0.08 = {args.ref_speed} m/s  (lbm_to_ms = {lbm_to_ms:.1f})")
    print(f"{'='*70}\n")

    # ── Load model first — need grid_size before rasterizing geometry ─────────
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Train first:  python train_model.py")
        return

    from src.model import WindFNO
    ckpt      = torch.load(args.model, map_location=device)
    modes     = ckpt.get('modes', 20)
    grid_size = args.grid if args.grid is not None else ckpt.get('grid_size', 256)
    model = WindFNO(in_channels=6, out_channels=4,
                    hidden=48, modes=modes, n_layers=4).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded  (modes={modes}, grid={grid_size})\n")

    # ── Geometry ──────────────────────────────────────────────────────────────
    from src.geometry import stl_to_obstacle_mask, make_synthetic_city

    if args.stl and os.path.exists(args.stl):
        print(f"Loading geometry from {args.stl}")
        obstacle_mask, _ = stl_to_obstacle_mask(args.stl, grid_size=grid_size)
    else:
        if test_mode and args.test_data and os.path.exists(args.test_data):
            # Fall back to geometry embedded in the test file
            print("No STL found — using obstacle_mask from test file")
            obstacle_mask = np.load(args.test_data)['obstacle_mask']
        else:
            print("No STL found — using synthetic city")
            obstacle_mask = make_synthetic_city(grid_size=grid_size, seed=42)

    COL = (f"{'#':>3}  {'Angle':>7}  {'Inlet (m/s)':>11}  "
           f"{'Vec RMSE':>10}  {'Speed MAE':>10}  {'Dir Error':>10}")
    SEP = '-' * len(COL)

    # ── Mode A: held-out test set ─────────────────────────────────────────────
    if test_mode:
        if not os.path.exists(args.test_data):
            print(f"Test file not found: {args.test_data}")
            print("Generate it first:")
            print("  python generate_data.py --stl data/city_model.STL")
            return

        test_npz = np.load(args.test_data)
        u_all_test = test_npz['u']      # [N, T, H, W]
        v_all_test = test_npz['v']
        angles_test = test_npz['angles'].tolist()
        speeds_test = test_npz['speeds'].tolist()
        N_test = u_all_test.shape[0]

        print(f"Held-out test set: {N_test} conditions\n")
        print(COL)
        print(SEP)

        results = []
        for i in range(N_test):
            angle = angles_test[i]
            speed = speeds_test[i]
            physical_inlet = speed * lbm_to_ms
            print(f"{i+1:>3}  {angle:>7.2f}°  {physical_inlet:>11.2f}",
                  end='  ', flush=True)

            u_arr = u_all_test[i]   # [T, H, W]
            v_arr = v_all_test[i]

            metrics = evaluate_condition(
                model, obstacle_mask, u_arr, v_arr,
                device=device, grid_size=grid_size,
                horizon=args.horizon, lbm_to_ms=lbm_to_ms, rng=rng)

            vec  = metrics['vector_rmse']
            mae  = metrics['speed_mae']
            dire = metrics['dir_error']
            print(f"{vec:>10.4f}  {mae:>10.4f}  {dire:>9.2f}°")
            results.append({'angle': angle, 'speed_ms': physical_inlet, **metrics})

        print(SEP)
        _print_summary(results, f'held-out test, rng seed={args.seed}', lbm_to_ms)

    # ── Mode B: random on-the-fly evaluation ──────────────────────────────────
    else:
        angles = rng.uniform(0,    360,  size=args.n).tolist()
        speeds = rng.uniform(0.02, 0.10, size=args.n).tolist()

        from src.lbm_solver import LBMSolver

        print(COL)
        print(SEP)

        results = []
        for i, (angle, speed) in enumerate(zip(angles, speeds)):
            physical_inlet = speed * lbm_to_ms
            print(f"{i+1:>3}  {angle:>7.1f}°  {physical_inlet:>11.2f}",
                  end='  ', flush=True)

            solver = LBMSolver(obstacle_mask, inlet_speed=speed,
                               inlet_angle=angle, tau=0.7)
            u_arr, v_arr = solver.run(n_warmup=args.warmup, n_collect=args.steps,
                                       collect_every=3, transient=False)

            metrics = evaluate_condition(
                model, obstacle_mask, u_arr, v_arr,
                device=device, grid_size=grid_size,
                horizon=args.horizon, lbm_to_ms=lbm_to_ms, rng=rng)

            vec  = metrics['vector_rmse']
            mae  = metrics['speed_mae']
            dire = metrics['dir_error']
            print(f"{vec:>10.4f}  {mae:>10.4f}  {dire:>9.2f}°")
            results.append({'angle': angle, 'speed_ms': physical_inlet, **metrics})

        print(SEP)
        _print_summary(results, f'seed={args.seed}', lbm_to_ms)


if __name__ == '__main__':
    main()
