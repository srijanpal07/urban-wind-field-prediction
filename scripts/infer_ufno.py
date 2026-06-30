"""
Inference script: predict the wind field at a given (or randomly drawn) wind condition.

If --angle or --speed are omitted, they are sampled randomly and printed so the
run is reproducible. Use --seed to fix the random draw.

Physical speed mapping: LBM speed 0.08 is treated as the reference and maps to
--ref-speed m/s. Other LBM speeds scale proportionally.

Usage:
    # Random condition:
    python infer.py --stl data/city_model.STL --model outputs/wind_fno.pth

    # Specific condition:
    python infer.py --stl data/city_model.STL --angle 45 --speed 0.08

    # Save to GIF:
    python infer.py --stl data/city_model.STL --save outputs/infer_result.gif

    # Fix random seed for reproducibility:
    python infer.py --stl data/city_model.STL --seed 7
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

import numpy as np
import torch

REF_LBM_SPEED = 0.08   # LBM speed that corresponds to --ref-speed m/s


def main():
    parser = argparse.ArgumentParser(description='Wind field inference')
    parser.add_argument('--stl',       type=str,   default=None)
    parser.add_argument('--model',     type=str,   default='outputs/wind_fno.pth')
    parser.add_argument('--angle',     type=float, default=None,
                        help='Inlet wind angle in degrees (random in [0,360) if not set)')
    parser.add_argument('--speed',     type=float, default=None,
                        help='LBM inlet speed in [0.02, 0.10] (random if not set)')
    parser.add_argument('--ref-speed', type=float, default=5.0,
                        help='Physical wind speed (m/s) corresponding to LBM speed 0.08')
    parser.add_argument('--grid',      type=int,   default=None,
                        help='Grid resolution (default: read from model checkpoint)')
    parser.add_argument('--warmup',    type=int,   default=1000)
    parser.add_argument('--steps',     type=int,   default=150)
    parser.add_argument('--save',      type=str,   default=None,
                        help='Save animation to this GIF path instead of showing interactively')
    parser.add_argument('--device',    type=str,   default='cuda')
    parser.add_argument('--seed',      type=int,   default=None,
                        help='RNG seed for random angle/speed draw')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    rng = np.random.default_rng(args.seed)
    angle = args.angle if args.angle is not None else float(rng.uniform(0, 360))
    speed = args.speed if args.speed is not None else float(rng.uniform(0.02, 0.10))

    lbm_to_ms      = args.ref_speed / REF_LBM_SPEED
    physical_inlet  = speed * lbm_to_ms

    print(f"{'='*55}")
    print(f" Urban Wind Field Inference")
    print(f" Device        : {device}")
    print(f" Angle         : {angle:.1f}°")
    print(f" LBM speed     : {speed:.4f}  →  {physical_inlet:.2f} m/s physical inlet")
    print(f"{'='*55}\n")

    # ── Load model first — need grid_size before running LBM ─────────────────
    print(f"[1/3] Loading model from {args.model}")
    model     = None
    grid_size = args.grid  # fallback if no checkpoint
    if os.path.exists(args.model):
        from src.models.ufno import WindFNO
        ckpt      = torch.load(args.model, map_location=device)
        modes     = ckpt.get('modes', 20)
        grid_size = args.grid if args.grid is not None else ckpt.get('grid_size', 256)
        model = WindFNO(in_channels=6, out_channels=4,
                        hidden=48, modes=modes, n_layers=4).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"      Loaded  (modes={modes}, grid={grid_size})")
    else:
        grid_size = args.grid if args.grid is not None else 256
        print(f"      Not found — showing persistence baseline")

    # ── Geometry ──────────────────────────────────────────────────────────────
    from src.data.geometry import stl_to_obstacle_mask, make_synthetic_city

    domain_m = None
    if args.stl and os.path.exists(args.stl):
        print(f"[2/3] Loading geometry from {args.stl}")
        obstacle_mask, bounds = stl_to_obstacle_mask(args.stl, grid_size=grid_size)
        domain_m = ((bounds['h0_max'] - bounds['h0_min']) / 1000.0,
                    (bounds['h1_max'] - bounds['h1_min']) / 1000.0)
    else:
        print("[2/3] No STL found — using synthetic city")
        obstacle_mask = make_synthetic_city(grid_size=grid_size, seed=42)

    # ── LBM simulation ────────────────────────────────────────────────────────
    print(f"[3/3] Running LBM  (warmup={args.warmup}, collect={args.steps}, grid={grid_size})...")
    from src.data.lbm_solver import LBMSolver

    solver = LBMSolver(obstacle_mask, inlet_speed=speed,
                       inlet_angle=angle, tau=0.7)
    u_arr, v_arr = solver.run(n_warmup=args.warmup, n_collect=args.steps,
                               collect_every=3, transient=True)
    print(f"      Wind field: {u_arr.shape}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    from src.viz.visualize import Dashboard

    print(f"\nLaunching dashboard  [angle={angle:.0f}°  speed={physical_inlet:.2f} m/s]")
    dash = Dashboard(u_arr, v_arr, obstacle_mask,
                     model=model, device=device,
                     lbm_to_ms=lbm_to_ms, domain_m=domain_m)
    dash.run(interval=80, save_gif=args.save)


if __name__ == '__main__':
    main()
