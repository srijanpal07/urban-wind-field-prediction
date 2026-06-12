"""
Wind Field Prediction Pipeline - End-to-End Runner
====================================================

Usage:
  # Full pipeline: generate data, train, visualize
  python run_pipeline.py --stl city.stl

  # Skip training (just generate data + visualize with persistence)
  python run_pipeline.py --stl city.stl --no-train

  # Train only
  python run_pipeline.py --train-only

  # Visualize with existing model
  python run_pipeline.py --stl city.stl --model wind_fno.pth

  # Save animation to GIF (headless)
  python run_pipeline.py --stl city.stl --save output.gif
"""

import argparse
import hashlib
import os
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stl',        type=str,   default=None)
    parser.add_argument('--model',      type=str,   default='outputs/wind_fno.pth')
    parser.add_argument('--grid',       type=int,   default=256)
    parser.add_argument('--warmup',     type=int,   default=400)
    parser.add_argument('--steps',      type=int,   default=150)
    parser.add_argument('--speed',      type=float, default=0.08)
    parser.add_argument('--ref-speed',  type=float, default=10.0,
                        help='Physical wind speed at inlet in m/s (for colorbar scaling)')
    parser.add_argument('--angle',      type=float, default=0.0)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch',      type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--horizon',    type=int,   default=10)
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--save',       type=str,   default=None)
    parser.add_argument('--no-train',   action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--data-cache',    type=str,   default='data/lbm_data.npz')
    parser.add_argument('--slice-height',  type=float, default=None,
                        help='Height along up-axis to slice STL (default: 30%% of height range)')
    parser.add_argument('--transient', action='store_true',
                        help='Vary inlet wind speed over time (gusty/transient flow)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"{'='*60}")
    print(f" Wind Field Prediction Pipeline")
    print(f" Device: {device}")
    print(f"{'='*60}\n")

    # ── Step 1: Geometry ──────────────────────────────────────────────────
    from src.geometry import stl_to_obstacle_mask, make_synthetic_city

    if args.stl and os.path.exists(args.stl):
        print(f"[1/4] Loading geometry from STL: {args.stl}")
        obstacle_mask, bounds = stl_to_obstacle_mask(
            args.stl, grid_size=args.grid,
            slice_height=args.slice_height)
    else:
        if args.stl:
            print(f"[1/4] STL '{args.stl}' not found — using synthetic city")
        else:
            print(f"[1/4] No STL provided — using synthetic city")
        obstacle_mask = make_synthetic_city(grid_size=args.grid, seed=42)

    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    np.save('data/obstacle_mask.npy', obstacle_mask)
    print(f"      Obstacle mask saved to data/obstacle_mask.npy")

    # ── Step 2: LBM Simulation ────────────────────────────────────────────
    def _mask_hash(m):
        return hashlib.md5(m.tobytes()).hexdigest()

    sim_mode = 'transient' if args.transient else 'steady'
    cache_hit = False
    if os.path.exists(args.data_cache):
        try:
            cached = np.load(args.data_cache, allow_pickle=False)
            stored_hash = str(cached.get('mask_hash', np.array(['']))[0])
            stored_mode = str(cached.get('sim_mode',  np.array(['steady']))[0])
            if stored_hash == _mask_hash(obstacle_mask) and stored_mode == sim_mode:
                u_arr = cached['u']
                v_arr = cached['v']
                print(f"\n[2/4] Loading cached LBM data from {args.data_cache} ({sim_mode})")
                print(f"      Loaded wind field: {u_arr.shape}")
                cache_hit = True
            else:
                reason = "geometry changed" if stored_hash != _mask_hash(obstacle_mask) else "sim mode changed"
                print(f"\n[2/4] Cache stale ({reason}) — regenerating LBM data")
        except Exception as e:
            print(f"\n[2/4] Cache unreadable ({e}) — regenerating")

    if not cache_hit:
        print(f"\n[2/4] Running LBM solver ({sim_mode} mode)...")
        print(f"      Grid: {args.grid}×{args.grid}  |  "
              f"Warmup: {args.warmup}  |  Collect: {args.steps}  |  "
              f"Speed: {args.speed}  |  Angle: {args.angle}°")
        from src.lbm_solver import LBMSolver
        solver = LBMSolver(obstacle_mask, inlet_speed=args.speed,
                           inlet_angle=args.angle, tau=0.7)
        u_arr, v_arr = solver.run(n_warmup=args.warmup,
                                   n_collect=args.steps,
                                   collect_every=3,
                                   transient=args.transient)
        np.savez(args.data_cache, u=u_arr, v=v_arr,
                 mask_hash=np.array([_mask_hash(obstacle_mask)]),
                 sim_mode=np.array([sim_mode]))
        print(f"      LBM data saved to {args.data_cache}")

    # ── Step 3: Training ──────────────────────────────────────────────────
    model = None
    if not args.no_train:
        print(f"\n[3/4] Training U-FNO model...")
        print(f"      Epochs: {args.epochs}  |  Batch: {args.batch}  |  "
              f"LR: {args.lr}  |  Horizon: {args.horizon}")
        from src.train import train
        model, history = train(
            u_arr, v_arr, obstacle_mask,
            save_path=args.model,
            grid_size=args.grid,
            n_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            horizon=args.horizon,
            device=device)

        # Plot training curve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        ax.plot(history['train'], color='#58a6ff', label='Train', linewidth=2)
        ax.plot(history['val'],   color='#f0883e', label='Val',   linewidth=2)
        ax.set_xlabel('Epoch', color='#8b949e')
        ax.set_ylabel('NLL Loss', color='#8b949e')
        ax.set_title('Training History', color='#e6edf3')
        ax.legend(facecolor='#21262d', labelcolor='#e6edf3')
        ax.tick_params(colors='#8b949e')
        for s in ax.spines.values(): s.set_edgecolor('#21262d')
        plt.tight_layout()
        plt.savefig('outputs/training_history.png', dpi=120, bbox_inches='tight')
        print(f"      Training curve saved to outputs/training_history.png")

    else:
        print(f"\n[3/4] Skipping training (--no-train)")

    if args.train_only:
        print("\nTrain-only mode. Done.")
        return

    # ── Step 4: Visualize ─────────────────────────────────────────────────
    print(f"\n[4/4] Launching visualization dashboard...")

    # Load best saved model
    if os.path.exists(args.model):
        from src.model import WindFNO
        ckpt = torch.load(args.model, map_location=device)
        modes = ckpt.get('modes', 20)
        model = WindFNO(in_channels=6, out_channels=4,
                        hidden=48, modes=modes, n_layers=4).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"      Loaded model from {args.model}")
    else:
        print(f"      No model found — showing persistence baseline")
        model = None

    # Training already imported matplotlib with Agg backend — can't switch to
    # interactive after the fact. Default to saving a GIF in full-pipeline mode.
    # Interactive display is available with --no-train (backend not yet locked).
    if not args.save and not args.no_train:
        args.save = 'outputs/wind_dashboard.gif'
        print(f"      Full pipeline: saving dashboard to {args.save}")

    from src.visualize import Dashboard
    lbm_to_ms = args.ref_speed / abs(args.speed)
    dash = Dashboard(u_arr, v_arr, obstacle_mask, model=model, device=device,
                     lbm_to_ms=lbm_to_ms)
    dash.run(interval=80, save_gif=args.save)


if __name__ == '__main__':
    main()
