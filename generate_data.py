"""
Data generation script for urban wind field prediction.
Runs LBM solver across training conditions (N angles × M speeds) and a
held-out test set then saves both datasets.

Per-condition results are cached as compressed npz files in data/cache/.
Training data is NOT combined into a single large file — WindDataset loads
conditions lazily on demand (no multi-GB npz needed, ~10 GB total at 512²).
A tiny metadata file (angles, speeds, mask) is written to --output.
Test data (16 conditions) is assembled into one compressed npz for evaluate.py.

Usage:
    python generate_data.py --stl data/city_model.STL
    python generate_data.py --stl data/city_model.STL --grid 512 --warmup 2000
    python generate_data.py --stl data/city_model.STL --skip-test --keep-cache
"""

import argparse
import hashlib
import os
from itertools import product

import numpy as np

# Training condition grid: 32 directions × 4 speeds = 128 conditions
# Every 11.25° gives denser interpolation coverage over the full 360° range.
DEFAULT_ANGLES = [
    0, 11.25, 22.5, 33.75, 45, 56.25, 67.5, 78.75,
    90, 101.25, 112.5, 123.75, 135, 146.25, 157.5, 168.75,
    180, 191.25, 202.5, 213.75, 225, 236.25, 247.5, 258.75,
    270, 281.25, 292.5, 303.75, 315, 326.25, 337.5, 348.75
]
DEFAULT_SPEEDS = [0.02, 0.04, 0.08, 0.10]

# Held-out test set: midpoints between every 4th pair of training angles
# (5.625° offset), guaranteed absent from training set.
DEFAULT_TEST_ANGLES = [5.625, 50.625, 95.625, 140.625, 185.625, 230.625, 275.625, 320.625]
DEFAULT_TEST_SPEEDS = [0.03, 0.06]


def _run_conditions(conditions, mode_str, warmup, steps, obstacle_mask,
                    mask_hash, cache_dir, LBMSolver, transient, label=''):
    """Run LBM for each condition, save to cache, return (paths, angles, speeds).

    Arrays are deleted from RAM immediately after caching — the caller never
    holds all conditions in memory at once.
    """
    N = len(conditions)
    cache_paths, angles_out, speeds_out = [], [], []

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
                    cache_paths.append(cache_file)
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

        np.savez_compressed(cache_file, u=u_arr, v=v_arr,
                            mask_hash=np.array([mask_hash]))
        del u_arr, v_arr   # free RAM immediately — only the cache file remains
        cache_paths.append(cache_file)
        angles_out.append(angle)
        speeds_out.append(speed)

    return cache_paths, angles_out, speeds_out


def _assemble_npz_compressed(cache_paths, angles, speeds, obstacle_mask, output_path):
    """Assemble combined compressed npz from a small number of condition files.

    Loads each condition sequentially (one at a time) then saves as compressed npz.
    Suitable for the held-out test set (16 conditions × ~160 MB = ~2.5 GB peak RAM).
    Do NOT use for large training sets — use lazy loading instead.
    """
    N = len(cache_paths)
    if N == 0:
        return

    d0 = np.load(cache_paths[0])
    T, H, W = d0['u'].shape
    del d0

    print(f"  Assembling {N} conditions → {output_path}")
    print(f"  Shape: ({N}, {T}, {H}, {W})  "
          f"≈ {N * T * H * W * 4 / 1e9:.1f} GB per field (before compression)")

    u_all = np.empty((N, T, H, W), dtype=np.float32)
    v_all = np.empty((N, T, H, W), dtype=np.float32)

    for i, path in enumerate(cache_paths):
        d = np.load(path)
        u_all[i] = d['u']
        v_all[i] = d['v']
        print(f"  [{i+1:3d}/{N}]", end='\r', flush=True)
    print()

    np.savez_compressed(output_path,
                        u=u_all, v=v_all,
                        angles=np.array(angles, dtype=np.float32),
                        speeds=np.array(speeds, dtype=np.float32),
                        obstacle_mask=obstacle_mask.astype(bool))
    del u_all, v_all


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-condition LBM wind field dataset')
    parser.add_argument('--stl',          type=str,  default=None)
    parser.add_argument('--grid',         type=int,  default=256)
    parser.add_argument('--warmup',       type=int,  default=1000,
                        help='LBM warmup steps')
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
                        default=DEFAULT_TEST_ANGLES)
    parser.add_argument('--test-speeds',  nargs='+', type=float,
                        default=DEFAULT_TEST_SPEEDS)
    parser.add_argument('--no-transient', action='store_true')
    parser.add_argument('--keep-cache',   action='store_true',
                        help='Keep existing cache files (skip re-running LBM)')
    parser.add_argument('--skip-test',    action='store_true',
                        help='Skip held-out test set generation')
    parser.add_argument('--device',       type=str,  default='cuda')
    args = parser.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'

    transient = not args.no_transient
    train_mode = 'transient' if transient else 'steady'

    n_train = len(args.angles) * len(args.speeds)
    n_test  = len(args.test_angles) * len(args.test_speeds)

    print(f"{'='*60}")
    print(f" Wind Field Data Generation")
    print(f" Device       : {device}")
    print(f" Grid         : {args.grid}×{args.grid}")
    print(f" Train        : {n_train} conditions  ({train_mode})")
    if not args.skip_test:
        print(f" Test         : {n_test} conditions  (steady)")
    print(f" Keep cache   : {args.keep_cache}")
    print(f"{'='*60}\n")

    # ── Geometry ──────────────────────────────────────────────────────────────
    from src.geometry import stl_to_obstacle_mask, make_synthetic_city

    if args.stl and os.path.exists(args.stl):
        print(f"[Geometry] Loading STL: {args.stl}")
        obstacle_mask, _ = stl_to_obstacle_mask(args.stl, grid_size=args.grid)
    else:
        print("[Geometry] No STL — using synthetic city")
        obstacle_mask = make_synthetic_city(grid_size=args.grid, seed=42)

    mask_hash = hashlib.md5(obstacle_mask.tobytes()).hexdigest()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Remove stale combined outputs (they will be regenerated)
    for path in [args.output, args.test_output]:
        if os.path.exists(path):
            os.remove(path)
            print(f"[Cleanup] Removed old {path}")

    if not args.keep_cache and os.path.isdir(args.cache_dir):
        removed = sum(
            1 for f in os.listdir(args.cache_dir)
            if f.endswith('.npz') and not os.remove(
                os.path.join(args.cache_dir, f)))
        print(f"[Cleanup] Removed {removed} cache file(s)")

    np.save('data/obstacle_mask.npy', obstacle_mask)

    from src.lbm_solver import LBMSolver

    # ── Training conditions ───────────────────────────────────────────────────
    train_conds = list(product(args.angles, args.speeds))
    print(f"\n[TRAIN] {len(train_conds)} conditions  ({train_mode})\n")

    tr_paths, ang_tr, spd_tr = _run_conditions(
        train_conds, train_mode, args.warmup, args.steps,
        obstacle_mask, mask_hash, args.cache_dir, LBMSolver,
        transient=transient, label='T')

    # Write tiny metadata file — training data stays in per-condition cache files
    # (lazy-loaded by WindDataset at training time, no multi-GB combined npz needed)
    print()
    np.savez_compressed(args.output,
                        angles=np.array(ang_tr, dtype=np.float32),
                        speeds=np.array(spd_tr, dtype=np.float32),
                        obstacle_mask=obstacle_mask.astype(bool))

    print(f"\n{'='*60}")
    print(f" Training metadata saved → {args.output}  ({len(tr_paths)} conditions in cache)")
    print(f" Per-condition files: {args.cache_dir}/lbm_{train_mode}_*.npz")
    print(f"{'='*60}")

    # ── Held-out test conditions (always steady-state) ────────────────────────
    if args.skip_test:
        print("\n[TEST] Skipped (--skip-test)")
        return

    test_conds = list(product(args.test_angles, args.test_speeds))
    print(f"\n[TEST] {len(test_conds)} held-out conditions  (steady)\n")

    te_paths, ang_te, spd_te = _run_conditions(
        test_conds, 'steady', args.warmup, args.steps,
        obstacle_mask, mask_hash, args.cache_dir, LBMSolver,
        transient=False, label='E')

    print()
    _assemble_npz_compressed(te_paths, ang_te, spd_te, obstacle_mask, args.test_output)

    print(f"\n{'='*60}")
    print(f" Test dataset saved → {args.test_output}")
    print(f" Angles : {ang_te}")
    print(f" Speeds : {spd_te}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
