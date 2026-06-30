"""
Standalone training script for WindFNO.
Loads a multi-condition dataset produced by generate_data.py and trains the model.

Usage:
    python train_model.py
    python train_model.py --data data/lbm_multicond.npz --epochs 50
    python train_model.py --data data/lbm_multicond.npz --model outputs/wind_fno.pth
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import os
import re

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='Train WindFNO on multi-condition LBM data')
    parser.add_argument('--data',      type=str,   default='data/lbm_multicond.npz',
                        help='Metadata file from generate_data.py (or legacy combined npz)')
    parser.add_argument('--cache-dir', type=str,   default='data/cache',
                        help='Directory containing per-condition npz files')
    parser.add_argument('--model',     type=str,   default='outputs/wind_fno.pth',
                        help='Where to save the best checkpoint')
    parser.add_argument('--epochs',    type=int,   default=100)
    parser.add_argument('--batch',     type=int,   default=32)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--horizon',   type=int,   default=10,
                        help='Forecast horizon in timesteps')
    parser.add_argument('--device',    type=str,   default='cuda')
    parser.add_argument('--resume',    action='store_true',
                        help='Resume training from the --model checkpoint; '
                             '--epochs is the TOTAL target (not additional)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print(f"{'='*60}")
    print(f" WindFNO Training")
    print(f" Device  : {device}")
    print(f" Output  : {args.model}")
    print(f" Epochs  : {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(args.model) or '.', exist_ok=True)

    # ── Detect data format ─────────────────────────────────────────────────────
    # New format: per-condition compressed npz files in cache dir
    # Old format: single combined npz with 'u' and 'v' arrays (legacy)
    condition_files = sorted(
        glob.glob(os.path.join(args.cache_dir, 'lbm_transient_*.npz')))

    if condition_files:
        # ── New format: lazy loading from per-condition cache files ────────────
        print(f"Found {len(condition_files)} training conditions in {args.cache_dir}")

        mask_path = 'data/obstacle_mask.npy'
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            print("Run generate_data.py first:  python generate_data.py --stl data/city_model.STL")
            return
        obstacle_mask = np.load(mask_path)

        # Parse angles / speeds from filenames for display only
        angles, speeds = [], []
        for f in condition_files:
            m = re.search(r'_a([\d.]+)_s([\d.]+)\.npz$', os.path.basename(f))
            if m:
                angles.append(float(m.group(1)))
                speeds.append(float(m.group(2)))

        d0 = np.load(condition_files[0])
        T, H, W = d0['u'].shape
        print(f"  {len(condition_files)} conditions  |  {T} timesteps  |  {H}×{W} grid")
        print(f"  Angle range: {min(angles):.2f}° – {max(angles):.2f}°")
        print(f"  Speed range: {min(speeds):.4f} – {max(speeds):.4f}\n")

        from src.training.train_ufno import train
        model, history = train(
            condition_files=condition_files,
            obstacle_mask=obstacle_mask,
            save_path=args.model,
            n_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            horizon=args.horizon,
            device=device,
            resume_path=args.model if args.resume else None)

    elif os.path.exists(args.data):
        # ── Legacy format: single combined npz ────────────────────────────────
        data = np.load(args.data)
        if 'u' not in data:
            print(f"No training conditions found in {args.cache_dir}")
            print(f"and {args.data} contains only metadata (no 'u' array).")
            print("Run generate_data.py first:  python generate_data.py --stl data/city_model.STL")
            return

        print(f"Loading legacy combined dataset: {args.data}")
        u_all         = data['u']
        v_all         = data['v']
        obstacle_mask = data['obstacle_mask']
        angles        = data['angles']
        speeds        = data['speeds']
        N, T, H, W = u_all.shape
        print(f"  {N} conditions  |  {T} timesteps  |  {H}×{W} grid")
        print(f"  Angles : {angles.tolist()}")
        print(f"  Speeds : {speeds.tolist()}\n")

        from src.training.train_ufno import train
        model, history = train(
            u_all, v_all, obstacle_mask,
            save_path=args.model,
            n_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            horizon=args.horizon,
            device=device,
            resume_path=args.model if args.resume else None)

    else:
        print("No training data found.")
        print(f"  Cache dir : {args.cache_dir}  (no lbm_transient_*.npz files)")
        print(f"  Data file : {args.data}  (not found)")
        print("Run generate_data.py first:  python generate_data.py --stl data/city_model.STL")
        return

    # ── Training curve ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(history['train'], color='#58a6ff', label='Train', linewidth=2)
    ax.plot(history['val'],   color='#f0883e', label='Val',   linewidth=2)
    ax.set_xlabel('Epoch', color='#8b949e')
    ax.set_ylabel('NLL Loss', color='#8b949e')
    ax.set_title(f'Training History — {len(condition_files) if condition_files else N} conditions', color='#e6edf3')
    ax.legend(facecolor='#21262d', labelcolor='#e6edf3')
    ax.tick_params(colors='#8b949e')
    for s in ax.spines.values():
        s.set_edgecolor('#21262d')
    plt.tight_layout()

    curve_path = args.model.replace('.pth', '_history.png')
    plt.savefig(curve_path, dpi=120, bbox_inches='tight')
    print(f"\nTraining curve saved → {curve_path}")


if __name__ == '__main__':
    main()
