"""
Wind Field Prediction - Interactive Visualization Dashboard
===========================================================
Shows side-by-side:
  Left:   LBM ground truth wind field
  Center: Model predicted wind field
  Right:  Uncertainty (sigma) field
  
With: building footprints, drone trajectory, live drone position,
      wind quiver arrows, and speed magnitude colormap.

Run: python visualize.py [--stl path/to/city.stl] [--model wind_fno.pth]
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import argparse
import os
import torch

from .geometry import stl_to_obstacle_mask, make_synthetic_city
from .lbm_solver import LBMSolver
from .drone_sampler import DroneSampler
from .model import WindFNO, prepare_input


# ─── Colour palette ────────────────────────────────────────────────────────────
BG      = '#0d1117'
PANEL   = '#161b22'
ACCENT  = '#58a6ff'
GREEN   = '#3fb950'
ORANGE  = '#f0883e'
RED     = '#f85149'
TEXT    = '#e6edf3'
SUBTEXT = '#8b949e'
GRID    = '#21262d'

CMAP_WIND  = 'RdYlBu_r'   # Wind speed magnitude
CMAP_UNCERT = 'YlOrRd'    # Uncertainty


def run_lbm(obstacle_mask, grid_size=128, n_warmup=300, n_collect=120,
            inlet_speed=0.1, inlet_angle=45.0):
    """Run LBM solver and return wind field time series."""
    solver = LBMSolver(obstacle_mask, inlet_speed=inlet_speed,
                       inlet_angle=inlet_angle, tau=0.7)
    u_arr, v_arr = solver.run(n_warmup=n_warmup, n_collect=n_collect,
                              collect_every=3)
    return u_arr, v_arr


def load_model(model_path, device):
    """Load trained WindFNO model."""
    ckpt = torch.load(model_path, map_location=device)
    model = WindFNO(in_channels=6, out_channels=4,
                    hidden=48, modes=20, n_layers=4).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def predict(model, obs_u_grid, obs_v_grid, obs_mask, geom_mask, device):
    """Run one model prediction step."""
    H, W = geom_mask.shape
    ys = np.linspace(0, 1, H)
    xs = np.linspace(0, 1, W)
    xg, yg = np.meshgrid(xs, ys)

    channels = np.stack([
        geom_mask.astype(np.float32),
        obs_u_grid.astype(np.float32),
        obs_v_grid.astype(np.float32),
        obs_mask.astype(np.float32),
        xg.astype(np.float32),
        yg.astype(np.float32)
    ], axis=0)

    x_in = torch.tensor(channels[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        u_p, v_p, su, sv = model(x_in)

    u_pred  = u_p[0, 0].cpu().numpy()
    v_pred  = v_p[0, 0].cpu().numpy()
    sigma_u = su[0, 0].cpu().numpy()
    sigma_v = sv[0, 0].cpu().numpy()
    return u_pred, v_pred, sigma_u, sigma_v


def make_quiver_grid(H, W, stride=8):
    """Return index grids for quiver downsampling."""
    ys = np.arange(stride//2, H, stride)
    xs = np.arange(stride//2, W, stride)
    xg, yg = np.meshgrid(xs, ys)
    return xg, yg, xs, ys


def speed(u, v):
    return np.sqrt(u**2 + v**2)


def mask_solid(arr, mask, fill=np.nan):
    out = arr.copy().astype(float)
    out[mask] = fill
    return out


def setup_figure():
    """Create the main dashboard figure."""
    fig = plt.figure(figsize=(18, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Title
    fig.text(0.5, 0.97, 'Urban Wind Field Prediction — Real-Time Dashboard',
             ha='center', va='top', color=TEXT, fontsize=14, fontweight='bold',
             fontfamily='monospace')
    fig.text(0.5, 0.935, 'LBM Ground Truth  |  U-FNO Prediction  |  Uncertainty',
             ha='center', va='top', color=SUBTEXT, fontsize=10, fontfamily='monospace')

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           top=0.90, bottom=0.08,
                           left=0.04, right=0.97,
                           hspace=0.35, wspace=0.15)

    # Top row: 3 wind field panels
    ax_gt    = fig.add_subplot(gs[0, 0])
    ax_pred  = fig.add_subplot(gs[0, 1])
    ax_uncert = fig.add_subplot(gs[0, 2])

    # Bottom row: error map + trajectory + metrics
    ax_err   = fig.add_subplot(gs[1, 0])
    ax_traj  = fig.add_subplot(gs[1, 1])
    ax_metrics = fig.add_subplot(gs[1, 2])

    axes = [ax_gt, ax_pred, ax_uncert, ax_err, ax_traj, ax_metrics]
    titles = ['Ground Truth (LBM)', 'Predicted (U-FNO)', 'Uncertainty σ',
              'Absolute Error |GT − Pred|', 'Drone Trajectory', 'Metrics Over Time']

    for ax, title in zip(axes[:-1], titles[:-1]):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=9, fontfamily='monospace', pad=4)
        ax.tick_params(colors=SUBTEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    ax_metrics.set_facecolor(PANEL)
    ax_metrics.set_title('Metrics Over Time', color=TEXT, fontsize=9,
                          fontfamily='monospace', pad=4)
    ax_metrics.tick_params(colors=SUBTEXT, labelsize=7)
    for spine in ax_metrics.spines.values():
        spine.set_edgecolor(GRID)

    return fig, ax_gt, ax_pred, ax_uncert, ax_err, ax_traj, ax_metrics


class Dashboard:
    """
    Live visualization dashboard.
    Animates drone flying through city, comparing LBM truth vs model prediction.
    """

    def __init__(self, u_arr, v_arr, obstacle_mask, model=None,
                 device='cpu', drone_waypoints=None, stl_path=None,
                 lbm_to_ms=1.0):
        self.u_arr = u_arr              # [T, H, W]
        self.v_arr = v_arr
        self.mask  = obstacle_mask      # [H, W] bool
        self.model = model
        self.device = device
        self.lbm_to_ms = lbm_to_ms     # LBM speed unit → m/s
        T, H, W = u_arr.shape
        self.T, self.H, self.W = T, H, W

        # Setup drone
        self.sampler = DroneSampler(grid_size=H, obstacle_mask=obstacle_mask,
                                    noise_std=0.02)
        if drone_waypoints is None:
            print("Planning street-following drone path (A*)...")
            self.waypoints = self.sampler.make_street_path(n_waypoints=10, seed=0)
            self._target_wps = getattr(self.sampler, '_last_targets', [])
        else:
            self.waypoints = drone_waypoints
            self._target_wps = drone_waypoints

        # Full drone path (one pass over all timesteps)
        self.x_path, self.y_path = self.sampler.interpolate_path(
            self.waypoints, T)

        # Cached observation confidence grid (updated each prediction call)
        self._conf_grid = np.zeros((H, W))

        # Observation buffer (rolling window)
        self.obs_window = 20
        self.obs_buffer = []

        # Metrics history
        self.mse_history  = []
        self.mae_history  = []
        self.t_history    = []

        # Quiver grid
        stride = max(4, H // 16)
        self.qxg, self.qyg, self.qxs, self.qys = make_quiver_grid(H, W, stride)

        # Global speed range for consistent colormaps
        spd = speed(u_arr, v_arr)
        self.vmin = float(np.nanpercentile(spd, 2))
        self.vmax = float(np.nanpercentile(spd, 98))

        # Setup figure
        self.fig, self.ax_gt, self.ax_pred, self.ax_uncert, \
            self.ax_err, self.ax_traj, self.ax_metrics = setup_figure()

        # Current prediction cache
        self.u_pred_cur = np.zeros((H, W))
        self.v_pred_cur = np.zeros((H, W))
        self.sigma_cur  = np.zeros((H, W))

        self._init_artists()

    def _init_artists(self):
        H, W = self.H, self.W
        blank = np.zeros((H, W))
        ext = [0, W, 0, H]

        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        from matplotlib.ticker import FuncFormatter
        s = self.lbm_to_ms
        speed_fmt = FuncFormatter(lambda x, _: f'{x * s:.1f}')
        err_fmt   = FuncFormatter(lambda x, _: f'{x * s:.2f}')

        def _style_cb(cb, fmt):
            """Apply dark-theme styling to a colorbar."""
            cb.ax.yaxis.set_major_formatter(fmt)
            cb.ax.yaxis.label.set_color(SUBTEXT)
            cb.ax.yaxis.label.set_fontsize(7)
            cb.ax.tick_params(colors=SUBTEXT, labelsize=6)

        # Ground truth panel
        self.im_gt = self.ax_gt.imshow(
            blank, extent=ext, origin='lower', cmap=CMAP_WIND,
            norm=norm, aspect='auto', interpolation='bilinear')
        self._draw_buildings(self.ax_gt)
        self.q_gt = self.ax_gt.quiver(
            self.qxg, self.qyg, blank[self.qys][:, self.qxs],
            blank[self.qys][:, self.qxs],
            color='white', alpha=0.5, scale=3, width=0.002)
        _style_cb(plt.colorbar(ScalarMappable(norm=norm, cmap=CMAP_WIND),
                               ax=self.ax_gt, fraction=0.046, pad=0.02,
                               label='Speed (m/s)'), speed_fmt)

        # Prediction panel
        self.im_pred = self.ax_pred.imshow(
            blank, extent=ext, origin='lower', cmap=CMAP_WIND,
            norm=norm, aspect='auto', interpolation='bilinear')
        self._draw_buildings(self.ax_pred)
        self.q_pred = self.ax_pred.quiver(
            self.qxg, self.qyg, blank[self.qys][:, self.qxs],
            blank[self.qys][:, self.qxs],
            color='white', alpha=0.5, scale=3, width=0.002)
        _style_cb(plt.colorbar(ScalarMappable(norm=norm, cmap=CMAP_WIND),
                               ax=self.ax_pred, fraction=0.046, pad=0.02,
                               label='Speed (m/s)'), speed_fmt)

        # Uncertainty panel (dynamic colormap — rescaled each frame)
        self.im_uncert = self.ax_uncert.imshow(
            blank, extent=ext, origin='lower', cmap=CMAP_UNCERT,
            vmin=0, vmax=0.05, aspect='auto', interpolation='bilinear')
        self._draw_buildings(self.ax_uncert)
        _style_cb(plt.colorbar(self.im_uncert, ax=self.ax_uncert,
                               fraction=0.046, pad=0.02, label='σ (m/s)'), speed_fmt)

        # Error panel (dynamic colormap — rescaled each frame)
        self.im_err = self.ax_err.imshow(
            blank, extent=ext, origin='lower', cmap='hot',
            vmin=0, vmax=0.05, aspect='auto', interpolation='bilinear')
        self._draw_buildings(self.ax_err)
        _style_cb(plt.colorbar(self.im_err, ax=self.ax_err,
                               fraction=0.046, pad=0.02, label='|Error| (m/s)'), err_fmt)

        # Trajectory panel — dark background, confidence heatmap, smooth buildings on top
        self.ax_traj.set_facecolor(BG)
        self.im_traj_conf = self.ax_traj.imshow(
            np.zeros((H, W)), extent=ext, origin='lower',
            cmap='YlOrRd', vmin=0, vmax=1, aspect='auto',
            alpha=0.65, zorder=1, interpolation='bilinear')
        self._draw_buildings(self.ax_traj)   # smooth buildings at zorder=2

        # Full planned path (faint, above buildings)
        self.ax_traj.plot(self.x_path, self.y_path,
                          color=SUBTEXT, alpha=0.3, linewidth=0.8, zorder=3)

        # Visited path (live, colored) — above buildings
        self.traj_line, = self.ax_traj.plot(
            [], [], color=ACCENT, linewidth=1.5, alpha=0.9, zorder=4)

        # Drone position marker
        self.drone_dot, = self.ax_traj.plot(
            [], [], 'o', color=GREEN, markersize=8, zorder=7,
            markeredgecolor='white', markeredgewidth=1)
        self.drone_dot_gt,  = self.ax_gt.plot(
            [], [], 'o', color=GREEN, markersize=6, zorder=7,
            markeredgecolor='white', markeredgewidth=1)
        self.drone_dot_pred, = self.ax_pred.plot(
            [], [], 'o', color=GREEN, markersize=6, zorder=7,
            markeredgecolor='white', markeredgewidth=1)

        # Observation scatter on trajectory panel
        self.obs_scatter = self.ax_traj.scatter(
            [], [], c=[], cmap='coolwarm', s=8, alpha=0.6,
            vmin=-0.15, vmax=0.15, zorder=5)

        # Target goal markers (sparse, even for A* paths)
        if self._target_wps:
            wx = [w[0] for w in self._target_wps]
            wy = [w[1] for w in self._target_wps]
            self.ax_traj.scatter(wx, wy, marker='D', color=ORANGE,
                                 s=30, zorder=8, label='Targets')
            for i, (wx_, wy_) in enumerate(zip(wx, wy)):
                self.ax_traj.text(wx_+1, wy_+1, f'T{i}', color=ORANGE,
                                  fontsize=6, fontfamily='monospace', zorder=9)

        # Metrics panel
        self.ax_metrics.set_xlabel('Timestep', color=SUBTEXT, fontsize=8)
        self.ax_metrics.set_ylabel('Error', color=SUBTEXT, fontsize=8)
        self.ax_metrics.set_facecolor(PANEL)
        self.ax_metrics.grid(color=GRID, alpha=0.5, linewidth=0.5)
        self.mse_line, = self.ax_metrics.plot([], [], color=ACCENT,
                                               linewidth=1.5, label='RMSE')
        self.mae_line, = self.ax_metrics.plot([], [], color=ORANGE,
                                               linewidth=1.5, label='MAE')
        self.ax_metrics.legend(facecolor=PANEL, edgecolor=GRID,
                                labelcolor=TEXT, fontsize=8)

        # Invert X axis on all spatial panels so that:
        #   col 0 (Z_min, LBM inlet) → displayed on the RIGHT
        #   col W-1 (Z_max, LBM outlet) → displayed on the LEFT
        # Wind flows col 0 → col W-1 in array space = RIGHT → LEFT visually.
        for _ax in (self.ax_gt, self.ax_pred, self.ax_uncert,
                    self.ax_err, self.ax_traj):
            _ax.invert_xaxis()

        # Status text
        self.status_text = self.fig.text(
            0.5, 0.02, '', ha='center', color=SUBTEXT,
            fontsize=9, fontfamily='monospace')

        # Time indicators — placed near top-left of each panel.
        # With inverted x-axis, array x = W-3 appears near the left edge.
        self.time_text_gt   = self.ax_gt.text(
            self.W-3, self.H-3, '', color=TEXT, fontsize=8, fontfamily='monospace',
            zorder=10, path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
        self.time_text_pred = self.ax_pred.text(
            self.W-3, self.H-3, '', color=TEXT, fontsize=8, fontfamily='monospace',
            zorder=10, path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    def _draw_buildings(self, ax):
        """
        Overlay building footprints as smooth vector contours.
        2× upsample + Gaussian blur removes the staircase pixelation that
        comes from rasterizing the STL to a 128² grid.
        """
        from scipy.ndimage import gaussian_filter, zoom
        H, W = self.mask.shape
        up = 2
        mask_up = zoom(self.mask.astype(float), up, order=1)  # bilinear upsample
        mask_up = gaussian_filter(mask_up, sigma=1.5)          # smooth edges
        # Cell-centre coordinates so contours align with imshow(extent=[0,W,0,H])
        xs = np.linspace(0.5, W - 0.5, W * up)
        ys = np.linspace(0.5, H - 0.5, H * up)
        ax.contourf(xs, ys, mask_up, levels=[0.5, 1.5],
                    colors=['#2d333b'], alpha=0.92, zorder=2)
        ax.contour(xs, ys, mask_up, levels=[0.5],
                   colors=['#4a5568'], linewidths=0.8, zorder=2)

    def _get_prediction(self, t):
        """Get model prediction for timestep t."""
        if self.model is None:
            # Fallback: simple persistence (last known field)
            return (self.u_arr[t], self.v_arr[t],
                    np.full((self.H, self.W), 0.02),
                    np.full((self.H, self.W), 0.02))

        # Collect observations from buffer
        if len(self.obs_buffer) == 0:
            obs_u_g = np.zeros((self.H, self.W))
            obs_v_g = np.zeros((self.H, self.W))
            conf_g  = np.zeros((self.H, self.W))
        else:
            recent = self.obs_buffer[-self.obs_window:]
            xs  = np.array([o['x'] for o in recent])
            ys  = np.array([o['y'] for o in recent])
            us  = np.array([o['u'] for o in recent])
            vs  = np.array([o['v'] for o in recent])
            obs = dict(x=xs, y=ys, u_obs=us, v_obs=vs)
            obs_u_g, obs_v_g, conf_g = self.sampler.obs_to_grid(
                obs, self.H, sigma=4.0)

        self._conf_grid = conf_g  # cache for trajectory panel overlay
        return predict(self.model, obs_u_g, obs_v_g, conf_g,
                       self.mask, self.device)

    def animate(self, t):
        """Update all panels for timestep t."""
        # Ground truth
        u_gt = mask_solid(self.u_arr[t], self.mask)
        v_gt = mask_solid(self.v_arr[t], self.mask)
        spd_gt = speed(u_gt, v_gt)

        # Add drone observation to buffer (only in fluid cells, not inside buildings)
        xi = self.x_path[t]
        yi = self.y_path[t]
        xi_i = int(np.clip(xi, 0, self.W-1))
        yi_i = int(np.clip(yi, 0, self.H-1))
        if not self.mask[yi_i, xi_i]:
            u_sample = self.u_arr[t, yi_i, xi_i] + np.random.normal(0, 0.02)
            v_sample = self.v_arr[t, yi_i, xi_i] + np.random.normal(0, 0.02)
            self.obs_buffer.append({'x': xi, 'y': yi, 'u': u_sample, 'v': v_sample})

        # Get prediction (every 5 frames to save compute)
        if t % 5 == 0 or t == 0:
            u_p, v_p, su, sv = self._get_prediction(t)
            self.u_pred_cur = mask_solid(u_p, self.mask)
            self.v_pred_cur = mask_solid(v_p, self.mask)
            self.sigma_cur  = np.sqrt(su**2 + sv**2)

        u_pred = self.u_pred_cur
        v_pred = self.v_pred_cur

        # ── Update Ground Truth panel ─────────────────────────────────────
        self.im_gt.set_data(spd_gt)
        qs = self.qys
        qx = self.qxs
        self.q_gt.set_UVC(-u_gt[np.ix_(qs, qx)], v_gt[np.ix_(qs, qx)])
        self.time_text_gt.set_text(f't={t:3d}')

        # ── Update Prediction panel ───────────────────────────────────────
        spd_pred = speed(u_pred, v_pred)
        spd_pred = np.where(np.isnan(spd_pred), np.nan, spd_pred)
        self.im_pred.set_data(spd_pred)
        self.q_pred.set_UVC(
            -np.nan_to_num(u_pred)[np.ix_(qs, qx)],
            np.nan_to_num(v_pred)[np.ix_(qs, qx)])
        self.time_text_pred.set_text(f't={t:3d} (+horizon)')

        # ── Uncertainty panel (auto-scaled) ───────────────────────────────
        sigma_disp = mask_solid(self.sigma_cur, self.mask)
        valid_s = sigma_disp[~np.isnan(sigma_disp)]
        if len(valid_s) > 0:
            self.im_uncert.set_clim(0, max(float(np.percentile(valid_s, 99)), 0.001))
        self.im_uncert.set_data(sigma_disp)

        # ── Error panel (auto-scaled) ─────────────────────────────────────
        err = np.abs(spd_gt - spd_pred)
        valid_e = err[~np.isnan(err)]
        if len(valid_e) > 0:
            self.im_err.set_clim(0, max(float(np.percentile(valid_e, 99)), 0.001))
        self.im_err.set_data(err)

        # ── Drone position ────────────────────────────────────────────────
        self.drone_dot.set_data([xi], [yi])
        self.drone_dot_gt.set_data([xi], [yi])
        self.drone_dot_pred.set_data([xi], [yi])

        # Observation confidence heatmap (shows where drone has sampled)
        conf_vis = self._conf_grid.copy()
        conf_vis[conf_vis < 0.01] = np.nan  # transparent where unsampled
        self.im_traj_conf.set_data(conf_vis)

        # Visited trajectory
        t0 = max(0, t - 40)
        self.traj_line.set_data(self.x_path[t0:t+1],
                                self.y_path[t0:t+1])

        # Observation scatter (recent obs colored by u)
        recent_n = min(50, len(self.obs_buffer))
        recent = self.obs_buffer[-recent_n:]
        if recent:
            ox = [o['x'] for o in recent]
            oy = [o['y'] for o in recent]
            ou = [o['u'] for o in recent]
            self.obs_scatter.set_offsets(np.c_[ox, oy])
            self.obs_scatter.set_array(np.array(ou))

        # ── Metrics ───────────────────────────────────────────────────────
        fluid = ~self.mask
        if fluid.any():
            gt_f   = np.nan_to_num(spd_gt)[fluid]
            pred_f = np.nan_to_num(spd_pred)[fluid]
            rmse = float(np.sqrt(np.mean((gt_f - pred_f)**2)))
            mae  = float(np.mean(np.abs(gt_f - pred_f)))
            self.mse_history.append(rmse)
            self.mae_history.append(mae)
            self.t_history.append(t)

            ts = self.t_history[-80:]
            self.mse_line.set_data(ts, self.mse_history[-80:])
            self.mae_line.set_data(ts, self.mae_history[-80:])
            self.ax_metrics.relim()
            self.ax_metrics.autoscale_view()

            self.status_text.set_text(
                f'Timestep: {t:3d}/{self.T-1}  |  '
                f'Drone: ({xi:.0f}, {yi:.0f})  |  '
                f'RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  '
                f'Mode: {"Model" if self.model else "Persistence"}')

        return []

    def run(self, interval=80, save_gif=None):
        """Launch interactive animation."""
        anim = FuncAnimation(
            self.fig, self.animate,
            frames=range(0, self.T, 1),
            interval=interval, blit=False, repeat=False)

        if save_gif:
            print(f"Saving animation to {save_gif}...")
            from matplotlib.animation import PillowWriter
            anim.save(save_gif, writer=PillowWriter(fps=12), dpi=100)
            print("Saved.")
        else:
            try:
                plt.show()
            except Exception as e:
                fallback = 'outputs/wind_dashboard.gif'
                print(f"\nInteractive display failed ({type(e).__name__}: {e})")
                print(f"Saving animation to {fallback}...")
                from matplotlib.animation import PillowWriter
                anim.save(fallback, writer=PillowWriter(fps=12), dpi=100)
                print(f"Saved to {fallback}")


def main():
    parser = argparse.ArgumentParser(description='Wind Field Prediction Dashboard')
    parser.add_argument('--stl',    type=str, default=None,
                        help='Path to city .stl file')
    parser.add_argument('--model',  type=str, default=None,
                        help='Path to trained model .pth file')
    parser.add_argument('--grid',   type=int, default=128,
                        help='Grid resolution (default 128)')
    parser.add_argument('--warmup', type=int, default=400,
                        help='LBM warmup steps')
    parser.add_argument('--steps',  type=int, default=120,
                        help='Number of timesteps to collect')
    parser.add_argument('--speed',  type=float, default=0.08,
                        help='Inlet wind speed (LB units)')
    parser.add_argument('--angle',  type=float, default=45.0,
                        help='Inlet wind angle (degrees)')
    parser.add_argument('--save',   type=str, default=None,
                        help='Save animation to GIF path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='PyTorch device')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Load or generate geometry ─────────────────────────────────────────
    if args.stl and os.path.exists(args.stl):
        print(f"Loading STL: {args.stl}")
        obstacle_mask, bounds = stl_to_obstacle_mask(
            args.stl, grid_size=args.grid)
    else:
        if args.stl:
            print(f"STL not found: {args.stl}. Using synthetic city.")
        else:
            print("No STL provided. Using synthetic city.")
        obstacle_mask = make_synthetic_city(grid_size=args.grid, seed=42)

    # ── Run LBM solver ────────────────────────────────────────────────────
    print("\nRunning LBM solver...")
    u_arr, v_arr = run_lbm(obstacle_mask, grid_size=args.grid,
                            n_warmup=args.warmup, n_collect=args.steps,
                            inlet_speed=args.speed, inlet_angle=args.angle)

    # ── Load model (optional) ─────────────────────────────────────────────
    model = None
    if args.model and os.path.exists(args.model):
        print(f"\nLoading model: {args.model}")
        model = load_model(args.model, device)
        print("Model loaded.")
    else:
        print("\nNo model provided — showing persistence baseline.")
        print("Train first with: python run_pipeline.py --train")

    # ── Launch dashboard ──────────────────────────────────────────────────
    print("\nLaunching dashboard...")
    dash = Dashboard(u_arr, v_arr, obstacle_mask, model=model, device=device)
    dash.run(interval=80, save_gif=args.save)


if __name__ == '__main__':
    main()
