"""
Data generation script for urban wind field prediction.
Runs LBM solver across training conditions (16 angles × 4 speeds = 64) and a
held-out test set (8 midpoint angles × 2 interpolation speeds = 16) then saves
both datasets for multi-condition training and rigorous held-out evaluation.

Training conditions  : transient (gusty) wind, saved to --output
Test conditions      : steady-state wind at angles/speeds NOT in the training
                       set, saved to --test-output. These are interpolation
                       targets — the model never sees these angles or speeds
                       during training.

Per-condition results are cached in data/cache/ so interrupted runs resume
without re-running completed conditions.

Usage:
    python generate_data.py --stl data/city_model.STL
    python generate_data.py --stl data/city_model.STL --grid 512 --warmup 2000
    python generate_data.py --stl data/city_model.STL --skip-test
"""

import argparse
import hashlib
import os
from itertools import product

import numpy as np

# Training condition grid: 16 directions × 4 speeds = 64 conditions
DEFAULT_ANGLES = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                  180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
DEFAULT_SPEEDS = [0.02, 0.04, 0.08, 0.10]

# Held-out test set: midpoints between every other pair of training angles,
# and speeds strictly between training speeds.  Neither angle nor speed
# appears in the training set — pure interpolation test.
DEFAULT_TEST_ANGLES = [11.25, 56.25, 101.25, 146.25, 191.25, 236.25, 281.25, 326.25]
DEFAULT_TEST_SPEEDS = [0.03, 0.06]


def _run_conditions(conditions, mode_str, warmup, steps, obstacle_mask,
                    mask_hash, cache_dir, LBMSolver, transient, label=''):
    """Run LBM for a list of (angle, speed) conditions; cache each result."""
    N = len(conditions)
    u_list, v_list, angles_out, speeds_out = [], [], [], []

    for i, (angle, speed) in enumerate(conditions):
        tag = f'[{label}{i+1:2d}/{N}]' if label else f'[{i+1:2d}/{N}]'
        cache_file = os.path.join(
            cache_dir,
            f'lbm_{mode_str}_a{angle:07.3f}_s{speed:.4f}.npz')

        if os.path.exists(cache_file):
            try:
                cached = np.load(cache_file)
                if str(cached['mask_hash'][0]) == mask_hash:
                    print(f"{tag} CACHED   angle={angle:6.2f}°  speed={speed:.4f}")
                    u_list.append(cached['u'])
                    v_list.append(cached['v'])
                    angles_out.append(angle)
                    speeds_out.append(speed)
                    continue
                print(f"{tag} STALE    angle={angle:6.2f}°  speed={speed:.4f}  (geometry changed)")
            except Exception as e:
                print(f"{tag} CORRUPT  angle={angle:6.2f}°  speed={speed:.4f}  ({e})")

        print(f"{tag} RUNNING  angle={angle:6.2f}°  speed={speed:.4f}", end='', flush=True)
        solver = LBMSolver(obstacle_mask, inlet_speed=speed,
                           inlet_angle=float(angle), tau=0.7)
        u_arr, v_arr = solver.run(n_warmup=warmup, n_collect=steps,
                                  collect_every=3, transient=transient)
        print(f"  → {u_arr.shape}")

        np.savez(cache_file, u=u_arr, v=v_arr,
                 mask_hash=np.array([mask_hash]))
        u_list.append(u_arr)
        v_list.append(v_arr)
        angles_out.append(angle)
        speeds_out.append(speed)

    return u_list, v_list, angles_out, speeds_out


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-condition LBM wind field dataset')
    parser.add_argument('--stl',          type=str,  default=None)
    parser.add_argument('--grid',         type=int,  default=256)
    parser.add_argument('--warmup',       type=int,  default=1000,
                        help='LBM warmup steps (train)')
    parser.add_argument('--steps',        type=int,  default=150,
                        help='Snapshots collected per condition')
    parser.add_argument('--output',       type=str,  default='data/lbm_multicond.npz')
    parser.add_argument('--test-output',  type=str,  default='data/lbm_test.npz')
    parser.add_argument('--cache-dir',    type=str,  default='data/cache')
    parser.add_argument('--angles',       nargs='+', type=float,
                        default=DEFAULT_ANGLES)
    parser.add_argument('--speeds',       nargs='+', type=float,
                        default=DEFAULT_SPEEDS)
    parser.add_argument('--test-angles',  nargs='+', type=float,
                        default=DEFAULT_TEST_ANGLES,
                        help='Held-out angles (must not overlap --angles)')
    parser.add_argument('--test-speeds',  nargs='+', type=float,
                        default=DEFAULT_TEST_SPEEDS,
                        help='Held-out speeds (must not overlap --speeds)')
    parser.add_argument('--no-transient', action='store_true',
                        help='Disable gusty inlet for training data')
    parser.add_argument('--keep-cache',   action='store_true',
                        help='Keep cache files (default: wipe and regenerate)')
    parser.add_argument('--skip-test',    action='store_true',
                        help='Skip held-out test set generation')
    parser.add_argument('--device',       type=str,  default='cuda')
    args = parser.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'

    transient = not args.no_transient
    train_mode = 'transient' if transient else 'steady'

    print(f"{'='*60}")
    print(f" Wind Field Data Generation")
    print(f" Device       : {device}")
    print(f" Grid         : {args.grid}×{args.grid}")
    print(f" Train angles : {args.angles}")
    print(f" Train speeds : {args.speeds}")
    print(f" Train mode   : {train_mode}")
    if not args.skip_test:
        print(f" Test angles  : {args.test_angles}  (steady)")
        print(f" Test speeds  : {args.test_speeds}")
    print(f"{'='*60}\n")

    # ── Geometry ──────────────────────────────────────────────────────────────
    from src.geometry import stl_to_obstacle_mask, make_synthetic_city

    if args.stl and os.path.exists(args.stl):
        print(f"[Geometry] Loading STL: {args.stl}")
        obstacle_mask, _ = stl_to_obstacle_mask(args.stl, grid_size=args.grid)
    else:
        if args.stl:
            print(f"[Geometry] STL '{args.stl}' not found — using synthetic city")
        else:
            print("[Geometry] No STL provided — using synthetic city")
        obstacle_mask = make_synthetic_city(grid_size=args.grid, seed=42)

    mask_hash = hashlib.md5(obstacle_mask.tobytes()).hexdigest()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Remove stale combined outputs
    for path in [args.output, args.test_output]:
        if os.path.exists(path):
            os.remove(path)
            print(f"[Cleanup] Removed old {path}")

    # Wipe per-condition cache by default
    if not args.keep_cache and os.path.isdir(args.cache_dir):
        removed = sum(
            1 for f in os.listdir(args.cache_dir)
            if f.endswith('.npz') and not os.remove(
                os.path.join(args.cache_dir, f)))
        print(f"[Cleanup] Removed {removed} cache file(s) from {args.cache_dir}/")

    np.save('data/obstacle_mask.npy', obstacle_mask)

    from src.lbm_solver import LBMSolver

    # ── Training conditions ───────────────────────────────────────────────────
    train_conds = list(product(args.angles, args.speeds))
    print(f"\n[TRAIN] {len(train_conds)} conditions  ({train_mode})\n")

    u_tr, v_tr, ang_tr, spd_tr = _run_conditions(
        train_conds, train_mode, args.warmup, args.steps,
        obstacle_mask, mask_hash, args.cache_dir, LBMSolver,
        transient=transient, label='T')

    u_all = np.stack(u_tr, axis=0)
    v_all = np.stack(v_tr, axis=0)
    np.savez(args.output,
             u=u_all, v=v_all,
             angles=np.array(ang_tr, dtype=np.float32),
             speeds=np.array(spd_tr, dtype=np.float32),
             obstacle_mask=obstacle_mask.astype(bool))

    print(f"\n{'='*60}")
    print(f" Training dataset saved → {args.output}")
    print(f"   Shape : u{u_all.shape}")
    print(f"{'='*60}")

    # ── Held-out test conditions (always steady-state) ────────────────────────
    if args.skip_test:
        print("\n[TEST] Skipped (--skip-test)")
        return

    test_conds = list(product(args.test_angles, args.test_speeds))
    print(f"\n[TEST] {len(test_conds)} held-out conditions  (steady)\n")

    u_te, v_te, ang_te, spd_te = _run_conditions(
        test_conds, 'steady', args.warmup, args.steps,
        obstacle_mask, mask_hash, args.cache_dir, LBMSolver,
        transient=False, label='E')

    u_test = np.stack(u_te, axis=0)
    v_test = np.stack(v_te, axis=0)
    np.savez(args.test_output,
             u=u_test, v=v_test,
             angles=np.array(ang_te, dtype=np.float32),
             speeds=np.array(spd_te, dtype=np.float32),
             obstacle_mask=obstacle_mask.astype(bool))

    print(f"\n{'='*60}")
    print(f" Test dataset saved → {args.test_output}")
    print(f"   Shape : u{u_test.shape}")
    print(f"   Angles: {ang_te}")
    print(f"   Speeds: {spd_te}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
