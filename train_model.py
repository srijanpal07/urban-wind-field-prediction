"""
Standalone training script for WindFNO.
Loads a multi-condition dataset produced by generate_data.py and trains the model.

Usage:
    python train_model.py
    python train_model.py --data data/lbm_multicond.npz --epochs 50
    python train_model.py --data data/lbm_multicond.npz --model outputs/wind_fno.pth
"""

import argparse
import os

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='Train WindFNO on multi-condition LBM data')
    parser.add_argument('--data',    type=str,   default='data/lbm_multicond.npz',
                        help='Path to dataset from generate_data.py')
    parser.add_argument('--model',   type=str,   default='outputs/wind_fno.pth',
                        help='Where to save the best checkpoint')
    parser.add_argument('--epochs',  type=int,   default=50)
    parser.add_argument('--batch',   type=int,   default=32)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--horizon', type=int,   default=10,
                        help='Forecast horizon in timesteps')
    parser.add_argument('--device',  type=str,   default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print(f"{'='*60}")
    print(f" WindFNO Training")
    print(f" Device  : {device}")
    print(f" Data    : {args.data}")
    print(f" Output  : {args.model}")
    print(f" Epochs  : {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    if not os.path.exists(args.data):
        print(f"Dataset not found: {args.data}")
        print("Run generate_data.py first:  python generate_data.py --stl data/city_model.STL")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.data} ...")
    data = np.load(args.data)
    u_all         = data['u']             # [N, T, H, W]
    v_all         = data['v']
    obstacle_mask = data['obstacle_mask']
    angles        = data['angles']
    speeds        = data['speeds']

    N, T, H, W = u_all.shape
    print(f"  {N} conditions  |  {T} timesteps  |  {H}×{W} grid")
    print(f"  Angles : {angles.tolist()}")
    print(f"  Speeds : {speeds.tolist()}\n")

    os.makedirs(os.path.dirname(args.model) or '.', exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    from src.train import train

    model, history = train(
        u_all, v_all, obstacle_mask,
        save_path=args.model,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        horizon=args.horizon,
        device=device)

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
    ax.set_title(f'Training History — {N} conditions', color='#e6edf3')
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
