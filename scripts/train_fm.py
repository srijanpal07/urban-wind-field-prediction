"""
Standalone training script for the spatiotemporal Flow-Matching model.
Loads a multi-condition dataset produced by generate_data.py and trains
FlowMatchingModel to predict a future wind sequence conditioned on sparse
drone observations and geometry.

Usage:
    python scripts/train_fm.py
    python scripts/train_fm.py --cache-dir data/cache --epochs 200 --batch 4
    python scripts/train_fm.py --resume --epochs 400
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
    parser = argparse.ArgumentParser(
        description='Train the spatiotemporal Flow-Matching model on multi-condition LBM data')
    parser.add_argument('--cache-dir', type=str, default='data/cache',
                         help='Directory containing per-condition npz files')
    parser.add_argument('--model', type=str, default='outputs/flow_matching/fm_model.pth',
                         help='Where to save the best checkpoint')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--T-out', type=int, default=20,
                         help='Forecast sequence length (timesteps)')
    parser.add_argument('--obs-window', type=int, default=30,
                         help='LBM snapshots spanned by the drone observation window')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n-levels', type=int, default=4)
    parser.add_argument('--t-emb-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', action='store_true',
                         help='Resume training from the --model checkpoint; '
                              '--epochs is the TOTAL target (not additional)')
    parser.add_argument('--no-amp', action='store_true',
                         help='Disable automatic mixed precision (float16)')
    parser.add_argument('--no-checkpoint', action='store_true',
                         help='Disable gradient checkpointing (faster but more VRAM)')
    parser.add_argument('--no-physics-prior', action='store_true',
                         help='Use plain Gaussian noise as the flow-matching source '
                              'distribution instead of the divergence-free, '
                              'obstacle-aware physics prior (ablation only)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print(f"{'='*60}")
    print(f" Flow-Matching Training")
    print(f" Device  : {device}")
    print(f" Output  : {args.model}")
    print(f" Epochs  : {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f" T_out   : {args.T_out}  |  Obs window: {args.obs_window}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(args.model) or '.', exist_ok=True)

    condition_files = sorted(
        glob.glob(os.path.join(args.cache_dir, 'lbm_transient_*.npz')))

    if not condition_files:
        print("No training data found.")
        print(f"  Cache dir : {args.cache_dir}  (no lbm_transient_*.npz files)")
        print("Run generate_data.py first:  python scripts/generate_data.py --stl data/city_model.STL")
        return

    print(f"Found {len(condition_files)} training conditions in {args.cache_dir}")

    mask_path = 'data/obstacle_mask.npy'
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        print("Run generate_data.py first:  python scripts/generate_data.py --stl data/city_model.STL")
        return
    obstacle_mask = np.load(mask_path)

    angles, speeds = [], []
    for f in condition_files:
        m = re.search(r'_a([\d.]+)_s([\d.]+)\.npz$', os.path.basename(f))
        if m:
            angles.append(float(m.group(1)))
            speeds.append(float(m.group(2)))

    d0 = np.load(condition_files[0])
    T, H, W = d0['u'].shape
    print(f"  {len(condition_files)} conditions  |  {T} timesteps  |  {H}x{W} grid")
    if angles and speeds:
        print(f"  Angle range: {min(angles):.2f} deg - {max(angles):.2f} deg")
        print(f"  Speed range: {min(speeds):.4f} - {max(speeds):.4f}\n")

    from src.training.train_fm import train_fm
    model, history = train_fm(
        condition_files=condition_files,
        obstacle_mask=obstacle_mask,
        save_path=args.model,
        T_out=args.T_out,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        obs_window=args.obs_window,
        device=device,
        hidden=args.hidden,
        n_levels=args.n_levels,
        t_emb_dim=args.t_emb_dim,
        resume_path=args.model if args.resume else None,
        use_amp=not args.no_amp,
        use_checkpoint=not args.no_checkpoint,
        use_physics_prior=not args.no_physics_prior)

    # ── Training curve ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot(history['train'], color='#58a6ff', label='Train', linewidth=2)
    ax.plot(history['val'], color='#f0883e', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', color='#8b949e')
    ax.set_ylabel('Flow-Matching Loss', color='#8b949e')
    ax.set_title(f'Training History — {len(condition_files)} conditions', color='#e6edf3')
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
