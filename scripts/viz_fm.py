"""
Flow-Matching Ensemble — Multi-Segment Animated Dashboard
==========================================================
The drone flies A → W1 → W2 → B. The model runs a fresh ensemble inference
at each waypoint using all observations collected so far. Three predictions
are shown, illustrating how the wind-field reconstruction and uncertainty
estimate sharpen as more observations accumulate.

Animation phases (6 total):
  1  A → W1  traversal  — drone moves, "collecting" placeholder
  2  Hold W1            — Prediction #1 shown (800 / 2400 obs)
  3  W1 → W2 traversal  — drone moves, Prediction #1 still visible
  4  Hold W2            — Prediction #2 shown (1600 / 2400 obs)
  5  W2 → B  traversal  — drone moves, Prediction #2 still visible
  6  Hold B             — Prediction #3 shown (2400 / 2400 obs, final)

Layout (2 × 3):
  [Ground Truth]   [Ensemble Mean]   [Ensemble Spread σ]
  [Drone Traj]     [|Error|]         [Segment timeline]

Usage:
  python scripts/viz_fm.py --stl data/city_model.STL
  python scripts/viz_fm.py --stl data/city_model.STL --save outputs/flow_matching/fm_dashboard.gif
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter, zoom

from src.data.geometry import stl_to_obstacle_mask, make_synthetic_city, build_geo_channels
from src.data.lbm_solver import LBMSolver
from src.data.drone_sampler import DroneSampler
from src.models.flow_matching import FlowMatchingModel

# ── colour palette ─────────────────────────────────────────────────────────────
BG, PANEL, ACCENT = '#0d1117', '#161b22', '#58a6ff'
GREEN, ORANGE, RED = '#3fb950', '#f0883e', '#f85149'
TEXT, SUBTEXT, GRID = '#e6edf3', '#8b949e', '#21262d'
CMAP_WIND, CMAP_SPREAD, CMAP_ERR = 'RdYlBu_r', 'YlOrRd', 'hot'

REF_LBM_SPEED = 0.08
_TOTAL_STEPS  = 2400
_OBS_WINDOW   = 30
_SEG_NAMES    = ['A', 'W1', 'W2', 'B']
_SEG_COLORS   = [GREEN, ORANGE, '#c084fc', ACCENT]   # A, W1, W2, B


def draw_buildings(ax, mask):
    up = 2; H, W = mask.shape
    mu = zoom(mask.astype(float), up, order=1)
    mu = gaussian_filter(mu, sigma=1.5)
    xs = np.linspace(0.5, W - 0.5, W * up)
    ys = np.linspace(0.5, H - 0.5, H * up)
    ax.contourf(xs, ys, mu, levels=[0.5, 1.5], colors=['#2d333b'], alpha=0.95, zorder=2)
    ax.contour(xs, ys, mu, levels=[0.5], colors=['#4a5568'], linewidths=0.8, zorder=2)


def speed_ms(u, v, lbm_to_ms):
    return np.sqrt(u ** 2 + v ** 2) * lbm_to_ms


def setup_wind_ax(ax, title, vmin, vmax, H, W, domain_m, cmap=CMAP_WIND):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=9, pad=3)
    ax.invert_xaxis()
    ax.tick_params(colors=SUBTEXT, labelsize=6)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    if domain_m:
        Wm, Hm = domain_m
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * Wm / W:.0f}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * Hm / H:.0f}'))
        ax.set_xlabel('Z (m)', color=SUBTEXT, fontsize=6)
        ax.set_ylabel('X (m)', color=SUBTEXT, fontsize=6)
    im = ax.imshow(np.zeros((H, W)), origin='lower', cmap=cmap,
                   norm=Normalize(vmin=vmin, vmax=vmax),
                   interpolation='bilinear', zorder=1, extent=[0, W, 0, H])
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.ax.tick_params(labelsize=6, colors=SUBTEXT)
    cb.set_label('m/s', color=SUBTEXT, fontsize=6)
    return im


def run_inference(obs_subset, sampler, obstacle_mask, model, solid_mask_t,
                  mask_t, grid_s, T_out, device, use_pp, n_samples, n_steps, rho):
    """Compute obs grid and run DPS ensemble on a partial set of observations."""
    obs_u_g, obs_v_g, obs_conf = sampler.obs_to_grid(obs_subset, grid_s, sigma=3.0)
    H, W = obstacle_mask.shape
    ys, xs = np.linspace(0, 1, H), np.linspace(0, 1, W)
    xg, yg = np.meshgrid(xs, ys)
    x_in = np.stack([obstacle_mask.astype(np.float32),
                     obs_u_g.astype(np.float32), obs_v_g.astype(np.float32),
                     obs_conf.astype(np.float32), xg.astype(np.float32),
                     yg.astype(np.float32)], axis=0)
    obs_t = torch.tensor(x_in[None], device=device)
    t0 = time.perf_counter()
    samples = model.sample(obs_t, mask_t, n_samples=n_samples, n_steps=n_steps,
                           rho=rho, device=device, solid_mask=solid_mask_t,
                           use_physics_prior=use_pp, chunk_size=1)
    elapsed = time.perf_counter() - t0
    u_s, v_s = samples[:, 0], samples[:, 1]
    u_p, v_p = FlowMatchingModel.leray_project(u_s, v_s)
    fluid = torch.tensor(~obstacle_mask, device=device).float()[None, None]
    u_p = (u_p * fluid).cpu().numpy()
    v_p = (v_p * fluid).cpu().numpy()
    return u_p, v_p, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stl',       type=str, default=None)
    parser.add_argument('--model',     type=str, default='outputs/flow_matching/fm_model.pth')
    parser.add_argument('--angle',     type=float, default=None)
    parser.add_argument('--speed',     type=float, default=None)
    parser.add_argument('--ref-speed', type=float, default=5.0)
    parser.add_argument('--warmup',    type=int, default=1000)
    parser.add_argument('--steps',     type=int, default=150)
    parser.add_argument('--n-samples', type=int, default=8)
    parser.add_argument('--n-steps',   type=int, default=20)
    parser.add_argument('--rho',       type=float, default=0.5)
    parser.add_argument('--traj-frames-per-seg', type=int, default=27,
                        help='Animation frames per A→W1, W1→W2, W2→B traversal')
    parser.add_argument('--hold-frames', type=int, default=20,
                        help='Frames to hold on each intermediate waypoint prediction')
    parser.add_argument('--hold-frames-b', type=int, default=35,
                        help='Frames to hold on the final B prediction')
    parser.add_argument('--fps',       type=int, default=12)
    parser.add_argument('--save',      type=str,
                        default='outputs/flow_matching/fm_dashboard.gif')
    parser.add_argument('--device',    type=str, default='cuda')
    parser.add_argument('--seed',      type=int, default=None)
    parser.add_argument('--no-physics-prior', action='store_true')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    lbm_to_ms = args.ref_speed / REF_LBM_SPEED
    rng = np.random.default_rng(args.seed if args.seed is not None else
                                  int(time.time()) % 100_000)

    # ── 1. Model ──────────────────────────────────────────────────────────────────
    print(f"Loading model from {args.model}")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    T_out  = ckpt.get('T_out', 10)
    grid_s = ckpt.get('grid_size', 256)
    hidden = ckpt.get('hidden', 64)
    n_lvl  = ckpt.get('n_levels', 4)
    t_emb  = ckpt.get('t_emb_dim', 256)
    use_pp = ckpt.get('use_physics_prior', True) and not args.no_physics_prior
    model  = FlowMatchingModel(T_out=T_out, hidden=hidden, n_levels=n_lvl,
                                t_emb_dim=t_emb, grid_size=grid_s).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  T_out={T_out}, grid={grid_s}, hidden={hidden}")

    # ── 2. Geometry ───────────────────────────────────────────────────────────────
    if args.stl and os.path.exists(args.stl):
        obstacle_mask, bounds = stl_to_obstacle_mask(args.stl, grid_size=grid_s)
        domain_m = (bounds['h0_max'] * 40 / 1000, bounds['h1_max'] * 40 / 1000)
    else:
        obstacle_mask = make_synthetic_city(grid_size=grid_s, seed=42)
        domain_m = None
    H, W = obstacle_mask.shape
    fluid_mask = ~obstacle_mask

    angle = args.angle if args.angle is not None else float(rng.uniform(0, 360))
    speed = args.speed if args.speed is not None else float(rng.uniform(0.04, 0.10))
    print(f"Condition: {angle:.1f}°  {speed:.4f} LBU ({speed * lbm_to_ms:.2f} m/s)")

    # ── 3. LBM ground truth ───────────────────────────────────────────────────────
    print(f"Running LBM...")
    solver = LBMSolver(obstacle_mask, inlet_speed=speed, inlet_angle=angle, tau=0.7)
    u_arr, v_arr = solver.run(n_warmup=args.warmup, n_collect=args.steps,
                               collect_every=3, transient=True)
    del solver
    if device == 'cuda':
        torch.cuda.empty_cache()

    T_lbm = u_arr.shape[0]
    t0_lbm = max(0, T_lbm - _OBS_WINDOW - T_out)
    t_seq_start = t0_lbm + _OBS_WINDOW
    t_end = min(t_seq_start + T_out, T_lbm)
    u_gt = u_arr[t_seq_start:t_end]
    v_gt = v_arr[t_seq_start:t_end]
    if u_gt.shape[0] < T_out:
        pad = T_out - u_gt.shape[0]
        u_gt = np.concatenate([u_gt, np.repeat(u_gt[-1:], pad, 0)], 0)
        v_gt = np.concatenate([v_gt, np.repeat(v_gt[-1:], pad, 0)], 0)
    speed_gt = speed_ms(u_gt, v_gt, lbm_to_ms)  # [T_out, H, W]

    # ── 4. Drone path — NO noise jitter for clean visualization ──────────────────
    print("Simulating drone path...")
    sampler = DroneSampler(grid_size=grid_s, obstacle_mask=obstacle_mask)
    waypoints = sampler.make_traverse_path(seed=int(rng.integers(0, 100_000)))
    x_path, y_path = sampler.interpolate_path(waypoints, _TOTAL_STEPS)
    # No Gaussian jitter here — jitter is for training diversity only,
    # it makes the visualization look scattered outside the actual path.
    t_idx = np.linspace(t0_lbm, t0_lbm + _OBS_WINDOW - 1, _TOTAL_STEPS).astype(int)
    obs_all = sampler.sample_field(u_arr, v_arr, x_path, y_path, t_idx)

    # Segment boundaries: A=0, W1=800, W2=1600, B=2400
    seg_steps = [0, _TOTAL_STEPS // 3, 2 * _TOTAL_STEPS // 3, _TOTAL_STEPS]
    seg_labels = [f'{n}  ({s} obs)' for n, s in zip(_SEG_NAMES[1:], seg_steps[1:])]

    mask_t = torch.tensor(build_geo_channels(obstacle_mask)[None], device=device)
    solid_mask_t = torch.tensor(obstacle_mask, dtype=torch.bool, device=device)

    # ── 5. Three inference runs ────────────────────────────────────────────────────
    preds = []  # list of (u_p, v_p, speed_mean, speed_spread, speed_err, n_obs)
    for i, n_obs in enumerate(seg_steps[1:]):
        obs_sub = {k: v[:n_obs] for k, v in obs_all.items()}
        print(f"Running inference at {_SEG_NAMES[i+1]} ({n_obs} observations)...")
        u_p, v_p, t_samp = run_inference(
            obs_sub, sampler, obstacle_mask, model, solid_mask_t, mask_t,
            grid_s, T_out, device, use_pp, args.n_samples, args.n_steps, args.rho)
        sm = speed_ms(u_p.mean(0), v_p.mean(0), lbm_to_ms)
        ss = speed_ms(u_p.std(0),  v_p.std(0),  lbm_to_ms)
        se = np.abs(sm - speed_gt)
        preds.append({'mean': sm, 'spread': ss, 'error': se,
                      'n_obs': n_obs, 't_samp': t_samp})
        print(f"  Done in {t_samp:.1f}s")

    # ── 6. Colour limits (shared across all 3 predictions) ───────────────────────
    vmax = max(float(np.percentile(speed_gt[0][fluid_mask], 98)), 0.5) * 1.05
    vmax_sp = max(max(float(np.percentile(p['spread'][0][fluid_mask], 98))
                      for p in preds), 0.1) * 1.05
    vmax_err = max(max(float(np.percentile(p['error'][0][fluid_mask], 95))
                       for p in preds), 0.1) * 1.05

    # Per-frame RMSE for each prediction (for timeline bar)
    rmse_bar = [[np.sqrt(((p['mean'][k] - speed_gt[k])[fluid_mask] ** 2).mean())
                 for k in range(T_out)]
                for p in preds]

    # ── 7. Figure ─────────────────────────────────────────────────────────────────
    print("Building animation...")
    fig = plt.figure(figsize=(18, 9), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.97, top=0.88, bottom=0.08)
    ax_gt     = fig.add_subplot(gs[0, 0])
    ax_mean   = fig.add_subplot(gs[0, 1])
    ax_spread = fig.add_subplot(gs[0, 2])
    ax_traj   = fig.add_subplot(gs[1, 0])
    ax_err    = fig.add_subplot(gs[1, 1])
    ax_seg    = fig.add_subplot(gs[1, 2])

    im_gt     = setup_wind_ax(ax_gt,     'Ground Truth (m/s)', 0, vmax, H, W, domain_m)
    im_mean   = setup_wind_ax(ax_mean,   'Ensemble Mean (m/s)', 0, vmax, H, W, domain_m)
    im_spread = setup_wind_ax(ax_spread, 'Ensemble Spread σ (m/s)', 0, vmax_sp, H, W,
                               domain_m, cmap=CMAP_SPREAD)
    im_err    = setup_wind_ax(ax_err,    '|Error| (m/s)', 0, vmax_err, H, W,
                               domain_m, cmap=CMAP_ERR)
    im_gt.set_data(speed_gt[0] * fluid_mask)   # GT fixed at first forecast frame

    for ax in (ax_gt, ax_mean, ax_spread, ax_err):
        draw_buildings(ax, obstacle_mask)

    # Trajectory panel
    ax_traj.set_facecolor(PANEL)
    ax_traj.set_title('Drone Trajectory + Observations', color=TEXT, fontsize=9, pad=3)
    ax_traj.invert_xaxis()
    ax_traj.tick_params(colors=SUBTEXT, labelsize=6)
    for s in ax_traj.spines.values():
        s.set_edgecolor(GRID)
    if domain_m:
        Wm, Hm = domain_m
        ax_traj.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * Wm / W:.0f}'))
        ax_traj.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * Hm / H:.0f}'))
        ax_traj.set_xlabel('Z (m)', color=SUBTEXT, fontsize=6)
        ax_traj.set_ylabel('X (m)', color=SUBTEXT, fontsize=6)
    draw_buildings(ax_traj, obstacle_mask)

    # Mark A, W1, W2, B on trajectory panel
    seg_x = [x_path[min(i, _TOTAL_STEPS - 1)] for i in seg_steps]
    seg_y = [y_path[min(i, _TOTAL_STEPS - 1)] for i in seg_steps]
    for xi, yi, name, col in zip(seg_x, seg_y, _SEG_NAMES, _SEG_COLORS):
        ax_traj.plot(xi, yi, 'o', color=col, ms=9, zorder=9,
                     markeredgecolor='white', markeredgewidth=1.2)
        ax_traj.text(xi + 4, yi + 4, name, color=col, fontsize=8,
                     fontweight='bold', zorder=10,
                     path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    traj_line,  = ax_traj.plot([], [], color=ACCENT, lw=1.5, alpha=0.8, zorder=4)
    drone_dot,  = ax_traj.plot([], [], 'o', color=GREEN, ms=8, zorder=8,
                                markeredgecolor='white', markeredgewidth=1)
    obs_spd = np.sqrt(obs_all['u_obs'] ** 2 + obs_all['v_obs'] ** 2) * lbm_to_ms
    obs_sc  = ax_traj.scatter([], [], c=[], cmap='coolwarm', s=5, alpha=0.5,
                               vmin=0, vmax=vmax, zorder=5)

    # Segment timeline panel
    ax_seg.set_facecolor(PANEL)
    ax_seg.set_title('Forecast Frame RMSE by Waypoint', color=TEXT, fontsize=9, pad=3)
    ax_seg.set_xlim(-0.5, T_out - 0.5)
    ax_seg.set_xlabel('Forecast frame (t+k)', color=SUBTEXT, fontsize=7)
    ax_seg.set_ylabel('RMSE (m/s)', color=SUBTEXT, fontsize=7)
    ax_seg.tick_params(colors=SUBTEXT, labelsize=7)
    for s in ax_seg.spines.values():
        s.set_edgecolor(GRID)
    bar_w = 0.25
    for pi, (rmse, col, lbl) in enumerate(zip(rmse_bar, _SEG_COLORS[1:], _SEG_NAMES[1:])):
        ax_seg.bar(np.arange(T_out) + (pi - 1) * bar_w, rmse,
                   width=bar_w, color=col, alpha=0.7, label=lbl)
    ax_seg.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
    seg_cursor = ax_seg.axvline(-1, color=RED, lw=2, zorder=5)
    seg_label  = ax_seg.text(0.5, 0.92, '', transform=ax_seg.transAxes,
                              ha='center', color=TEXT, fontsize=8)

    title_txt  = fig.suptitle('', color=TEXT, fontsize=11, y=0.96)
    status_txt = fig.text(0.5, 0.005, '', ha='center', color=SUBTEXT,
                          fontsize=8, fontfamily='monospace')

    # ── 8. Animation frames ────────────────────────────────────────────────────────
    SPF = args.traj_frames_per_seg   # traj frames per segment
    HF  = args.hold_frames           # hold frames at W1, W2
    HFB = args.hold_frames_b         # hold frames at B

    # Frame schedule:
    #  [0,         SPF)       → seg 0: A→W1 traversal
    #  [SPF,       SPF+HF)    → hold at W1, show pred[0]
    #  [SPF+HF,    2*SPF+HF)  → seg 1: W1→W2 traversal
    #  [2*SPF+HF,  2*(SPF+HF))→ hold at W2, show pred[1]
    #  [2*(SPF+HF), 3*SPF+2*HF) → seg 2: W2→B traversal
    #  [3*SPF+2*HF, end)      → hold at B, show pred[2]
    schedule = [
        (0,              SPF,           'traj', 0),
        (SPF,            SPF + HF,      'hold', 0),
        (SPF + HF,       2*SPF + HF,    'traj', 1),
        (2*SPF + HF,     2*SPF + 2*HF,  'hold', 1),
        (2*SPF + 2*HF,   3*SPF + 2*HF,  'traj', 2),
        (3*SPF + 2*HF,   3*SPF + 2*HF + HFB, 'hold', 2),
    ]
    N_FRAMES = 3 * SPF + 2 * HF + HFB

    _cur_pred_idx = [-1]   # mutable for closures

    def get_phase(frame):
        for (fstart, fend, kind, seg_idx) in schedule:
            if fstart <= frame < fend:
                return kind, seg_idx, frame - fstart, fend - fstart
        return 'hold', 2, HFB - 1, HFB

    def init():
        im_gt.set_data(speed_gt[0] * fluid_mask)
        im_mean.set_data(np.zeros((H, W)))
        im_spread.set_data(np.zeros((H, W)))
        im_err.set_data(np.zeros((H, W)))
        traj_line.set_data([], [])
        drone_dot.set_data([], [])
        obs_sc.set_offsets(np.empty((0, 2)))
        obs_sc.set_array(np.array([]))
        seg_cursor.set_xdata([-1])
        return (im_gt, im_mean, im_spread, im_err, traj_line, drone_dot,
                obs_sc, seg_cursor)

    def animate(frame):
        kind, seg_idx, local_f, local_n = get_phase(frame)

        # ── Drone position ──────────────────────────────────────────────────────
        if kind == 'traj':
            # Interpolate within current segment
            seg_s = seg_steps[seg_idx]
            seg_e = seg_steps[seg_idx + 1]
            step  = int(seg_s + (seg_e - seg_s) * local_f / max(local_n - 1, 1))
            step  = min(step, _TOTAL_STEPS - 1)
        else:
            step = seg_steps[seg_idx + 1] - 1

        # Show trajectory up to current step
        traj_line.set_data(x_path[:step + 1], y_path[:step + 1])
        drone_dot.set_data([x_path[step]], [y_path[step]])

        # Observations up to current step
        if step > 0:
            obs_sc.set_offsets(np.column_stack([
                obs_all['x'][:step], obs_all['y'][:step]]))
            obs_sc.set_array(obs_spd[:step])

        # ── Prediction panels ────────────────────────────────────────────────────
        if kind == 'hold':
            p = preds[seg_idx]
            # Cycle through T_out frames on the hold
            fore_f = local_f % T_out
            im_mean.set_data(p['mean'][fore_f] * fluid_mask)
            im_spread.set_data(p['spread'][fore_f] * fluid_mask)
            im_err.set_data(p['error'][fore_f] * fluid_mask)
            im_gt.set_data(speed_gt[fore_f] * fluid_mask)
            seg_cursor.set_xdata([fore_f])
            seg_label.set_text(f't+{fore_f}')

            wpt  = _SEG_NAMES[seg_idx + 1]
            rmse = float(np.sqrt(((p['mean'][0] - speed_gt[0])[fluid_mask] ** 2).mean()))
            spr  = float(p['spread'][0][fluid_mask].mean())
            title_txt.set_text(
                f'Flow-Matching  {angle:.1f}°, {speed * lbm_to_ms:.2f} m/s'
                f'  │  Prediction at {wpt}  ({p["n_obs"]} / {_TOTAL_STEPS} obs)')
            status_txt.set_text(
                f'RMSE = {rmse:.3f} m/s  │  Spread = {spr:.3f} m/s avg  │'
                f'  Sampling: {p["t_samp"]:.1f}s')

        else:
            # During traversal, keep showing the previous prediction if any
            pred_idx = seg_idx - 1
            if pred_idx >= 0:
                p = preds[pred_idx]
                im_mean.set_data(p['mean'][0] * fluid_mask)
                im_spread.set_data(p['spread'][0] * fluid_mask)
                im_err.set_data(p['error'][0] * fluid_mask)
                im_gt.set_data(speed_gt[0] * fluid_mask)
                seg_cursor.set_xdata([-1])
                seg_label.set_text('')
                wpt = _SEG_NAMES[pred_idx + 1]
                title_txt.set_text(
                    f'Flow-Matching  {angle:.1f}°, {speed * lbm_to_ms:.2f} m/s'
                    f'  │  Drone continuing (last pred: {wpt})')
            else:
                im_mean.set_data(np.zeros((H, W)))
                im_spread.set_data(np.zeros((H, W)))
                im_err.set_data(np.zeros((H, W)))
                im_gt.set_data(speed_gt[0] * fluid_mask)
                seg_cursor.set_xdata([-1])
                seg_label.set_text('')
                frac = (step + 1) / seg_steps[1]
                title_txt.set_text(
                    f'Flow-Matching  {angle:.1f}°, {speed * lbm_to_ms:.2f} m/s'
                    f'  │  Phase 1: Collecting observations ({frac * 100:.0f}%)')
            status_txt.set_text(
                f'Step {step + 1}/{_TOTAL_STEPS}  │  Observations so far: {step + 1}')

        return (im_gt, im_mean, im_spread, im_err, traj_line, drone_dot,
                obs_sc, seg_cursor, title_txt, status_txt)

    anim = FuncAnimation(fig, animate, frames=N_FRAMES, init_func=init,
                         interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    print(f"Saving {N_FRAMES}-frame GIF → {args.save}  (fps={args.fps})")
    t_gif = time.perf_counter()
    anim.save(args.save, writer=PillowWriter(fps=args.fps))
    print(f"Done in {time.perf_counter() - t_gif:.1f}s  →  {args.save}")
    plt.close(fig)


if __name__ == '__main__':
    main()
