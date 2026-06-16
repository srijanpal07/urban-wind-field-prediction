"""
Synthetic Drone Sampler
Simulates a drone flying a trajectory through the wind field,
sampling u/v velocity with Gaussian noise.

Drone path: straight line waypoints (A -> B -> C -> ...)
Returns sparse (x, y, u_noisy, v_noisy, t) observations.
"""

import numpy as np


class DroneSampler:
    """
    Simulates drone wind sampling along a planned trajectory.

    Parameters
    ----------
    grid_size      : int, domain grid resolution
    obstacle_mask  : bool array [H, W], True = solid (drone avoids these)
    sample_freq    : float, samples per timestep
    noise_speed_std : float, speed noise std in LBM units (≈ 0.5 m/s at lbm_to_ms=62.5)
    noise_angle_std : float, direction noise std in degrees (default ±10°)
    """

    def __init__(self, grid_size: int = 128, obstacle_mask: np.ndarray = None,
                 noise_speed_std: float = 0.008, noise_angle_std: float = 10.0):
        self.G = grid_size
        self.obstacle_mask = obstacle_mask
        self.noise_speed_std = noise_speed_std  # LBM units (~0.5 m/s at lbm_to_ms=62.5)
        self.noise_angle_std = noise_angle_std  # degrees

    def make_waypoints(self, n_waypoints: int = 6, seed: int = 0,
                       margin: float = 0.1):
        """
        Generate random waypoints in free (non-solid) space.
        Returns list of (x, y) in grid coordinates [0, G].
        """
        rng = np.random.default_rng(seed)
        G = self.G
        lo = int(margin * G)
        hi = int((1 - margin) * G)
        waypoints = []

        attempts = 0
        while len(waypoints) < n_waypoints and attempts < 10000:
            x = rng.integers(lo, hi)
            y = rng.integers(lo, hi)
            attempts += 1
            if self.obstacle_mask is not None:
                if self.obstacle_mask[y, x]:
                    continue
            waypoints.append((float(x), float(y)))

        return waypoints

    def _astar(self, start_rc, goal_rc, clearance: int = 2):
        """
        A* grid search from start_rc=(row,col) to goal_rc=(row,col).
        Returns list of (row, col) positions. Inflates obstacles by `clearance`
        cells so the drone stays away from building walls.
        """
        import heapq
        from scipy.ndimage import binary_dilation

        G = self.G
        if self.obstacle_mask is not None:
            blocked = (binary_dilation(self.obstacle_mask, iterations=clearance)
                       if clearance > 0 else self.obstacle_mask)
        else:
            blocked = np.zeros((G, G), dtype=bool)

        def h(a, b):
            return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

        counter = 0
        open_set = [(h(start_rc, goal_rc), counter, start_rc)]
        came_from = {}
        g_cost = {start_rc: 0.0}

        while open_set:
            _, _, cur = heapq.heappop(open_set)
            if cur == goal_rc:
                path, node = [], cur
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start_rc)
                path.reverse()
                return path

            r, c = cur
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r+dr, c+dc
                if not (0 <= nr < G and 0 <= nc < G) or blocked[nr, nc]:
                    continue
                step = 1.414 if (dr and dc) else 1.0
                ng = g_cost[cur] + step
                nb = (nr, nc)
                if ng < g_cost.get(nb, float('inf')):
                    g_cost[nb] = ng
                    counter += 1
                    heapq.heappush(open_set, (ng + h(nb, goal_rc), counter, nb))
                    came_from[nb] = cur

        return [start_rc, goal_rc]  # fallback: direct jump if unreachable

    def make_street_path(self, n_waypoints: int = 8, seed: int = 0,
                          clearance: int = 3, margin: float = 0.1):
        """
        Generate a survey path that navigates through free-space corridors
        between buildings using A* routing.

        Picks `n_waypoints` random free-space targets, then A*-connects them
        in sequence. Returns a flat list of (x, y) grid positions along the
        full route (can be passed directly to interpolate_path).
        """
        targets = self.make_waypoints(n_waypoints, seed, margin)
        self._last_targets = targets  # stored so Dashboard can show goal markers

        if len(targets) < 2:
            return targets

        all_pts = []
        for i in range(len(targets) - 1):
            r0, c0 = int(targets[i][1]),   int(targets[i][0])
            r1, c1 = int(targets[i+1][1]), int(targets[i+1][0])
            seg = self._astar((r0, c0), (r1, c1), clearance)
            pts = [(rc[1], rc[0]) for rc in seg]   # (col, row) → (x, y)
            all_pts.extend(pts if i == 0 else pts[1:])

        return all_pts

    def make_traverse_path(self, margin: float = 0.1, seed: int = 0,
                           clearance: int = 2):
        """
        Generate a single left-to-right traverse using A*.
        Starts at the low-column edge (array-left / display-right with invert_xaxis)
        and ends at the high-column edge, at randomly chosen free-space rows.
        No loops by construction.
        """
        rng = np.random.default_rng(seed)
        G = self.G
        lo = int(margin * G)
        hi = int((1 - margin) * G)

        def pick_free_row(col):
            if self.obstacle_mask is not None:
                free = [r for r in range(lo, hi) if not self.obstacle_mask[r, col]]
            else:
                free = list(range(lo, hi))
            candidates = free if free else list(range(lo, hi))
            return int(rng.choice(candidates))

        row_start = pick_free_row(lo)
        row_end   = pick_free_row(hi)

        path = self._astar((row_start, lo), (row_end, hi), clearance)
        waypoints = [(float(rc[1]), float(rc[0])) for rc in path]  # (col,row) → (x,y)
        self._last_targets = [(float(lo), float(row_start)),
                              (float(hi), float(row_end))]
        return waypoints

    def make_lawnmower_path(self, n_passes: int = 4, margin: float = 0.15):
        """
        Generate a lawnmower survey path (parallel horizontal sweeps).
        Good for dense coverage of the domain.
        """
        G = self.G
        lo = int(margin * G)
        hi = int((1 - margin) * G)
        waypoints = []
        xs = np.linspace(lo, hi, n_passes * 2)
        for i, x in enumerate(xs):
            if i % 2 == 0:
                waypoints.append((float(x), float(lo)))
                waypoints.append((float(x), float(hi)))
            else:
                waypoints.append((float(x), float(hi)))
                waypoints.append((float(x), float(lo)))
        return waypoints

    def interpolate_path(self, waypoints, total_steps: int):
        """
        Interpolate waypoints into a smooth trajectory of total_steps positions.
        Returns arrays x_path, y_path of shape [total_steps].
        """
        # Cumulative distance
        dists = [0.0]
        for i in range(1, len(waypoints)):
            dx = waypoints[i][0] - waypoints[i-1][0]
            dy = waypoints[i][1] - waypoints[i-1][1]
            dists.append(dists[-1] + np.sqrt(dx**2 + dy**2))
        total_dist = dists[-1]

        t_uniform = np.linspace(0, total_dist, total_steps)
        x_path = np.interp(t_uniform, dists, [w[0] for w in waypoints])
        y_path = np.interp(t_uniform, dists, [w[1] for w in waypoints])
        return x_path, y_path

    def sample_field(self, u_field: np.ndarray, v_field: np.ndarray,
                     x_path: np.ndarray, y_path: np.ndarray,
                     t_indices: np.ndarray):
        """
        Sample u/v at drone positions with bilinear interpolation + noise.

        Parameters
        ----------
        u_field, v_field : [T, H, W] wind field time series
        x_path, y_path   : [N] drone positions in grid coords
        t_indices        : [N] which timestep each position corresponds to

        Returns
        -------
        obs : dict with keys x, y, t, u_obs, v_obs, u_true, v_true
        """
        T, H, W = u_field.shape
        N = len(x_path)

        u_obs = np.zeros(N)
        v_obs = np.zeros(N)
        u_true = np.zeros(N)
        v_true = np.zeros(N)

        for i in range(N):
            xi = np.clip(x_path[i], 0, W - 1.001)
            yi = np.clip(y_path[i], 0, H - 1.001)
            ti = int(np.clip(t_indices[i], 0, T - 1))

            # Skip observations inside solid obstacles (buildings)
            if self.obstacle_mask is not None:
                ix = int(np.clip(xi, 0, W - 1))
                iy = int(np.clip(yi, 0, H - 1))
                if self.obstacle_mask[iy, ix]:
                    u_obs[i] = np.nan
                    v_obs[i] = np.nan
                    continue

            # Bilinear interpolation
            x0, y0 = int(xi), int(yi)
            x1, y1 = min(x0 + 1, W-1), min(y0 + 1, H-1)
            fx, fy = xi - x0, yi - y0

            u_i = ((1-fx)*(1-fy)*u_field[ti, y0, x0] +
                   fx*(1-fy)*u_field[ti, y0, x1] +
                   (1-fx)*fy*u_field[ti, y1, x0] +
                   fx*fy*u_field[ti, y1, x1])
            v_i = ((1-fx)*(1-fy)*v_field[ti, y0, x0] +
                   fx*(1-fy)*v_field[ti, y0, x1] +
                   (1-fx)*fy*v_field[ti, y1, x0] +
                   fx*fy*v_field[ti, y1, x1])

            u_true[i] = u_i
            v_true[i] = v_i
            speed_true = np.sqrt(u_i**2 + v_i**2)
            angle_true = np.arctan2(v_i, u_i)
            speed_noisy = max(0.0, speed_true + np.random.normal(0, self.noise_speed_std))
            angle_noisy = angle_true + np.deg2rad(np.random.normal(0, self.noise_angle_std))
            u_obs[i] = speed_noisy * np.cos(angle_noisy)
            v_obs[i] = speed_noisy * np.sin(angle_noisy)

        return dict(x=x_path, y=y_path, t=t_indices,
                    u_obs=u_obs, v_obs=v_obs,
                    u_true=u_true, v_true=v_true)

    def obs_to_grid(self, obs: dict, grid_size: int, sigma: float = 3.0):
        """
        Scatter sparse observations onto a grid using Gaussian splatting.
        Returns:
          obs_u_grid   : [H, W] interpolated u observations
          obs_v_grid   : [H, W] interpolated v observations
          obs_mask     : [H, W] float, confidence weight per cell
        """
        H = W = grid_size
        xs = np.asarray(obs['x'],     dtype=np.float32)
        ys = np.asarray(obs['y'],     dtype=np.float32)
        us = np.asarray(obs['u_obs'], dtype=np.float32)
        vs = np.asarray(obs['v_obs'], dtype=np.float32)

        valid = ~(np.isnan(us) | np.isnan(vs))
        if not valid.any():
            return np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
        xs, ys, us, vs = xs[valid], ys[valid], us[valid], vs[valid]

        u_grid = np.zeros((H, W), dtype=np.float32)
        v_grid = np.zeros((H, W), dtype=np.float32)
        w_grid = np.zeros((H, W), dtype=np.float32)

        r       = int(np.ceil(3.0 * sigma))
        inv_2s2 = np.float32(0.5 / sigma**2)
        gx_all  = np.arange(W, dtype=np.float32)
        gy_all  = np.arange(H, dtype=np.float32)

        for i in range(len(xs)):
            x0 = max(0, int(xs[i]) - r);  x1 = min(W, int(xs[i]) + r + 1)
            y0 = max(0, int(ys[i]) - r);  y1 = min(H, int(ys[i]) + r + 1)

            px = gx_all[x0:x1]   # [w_patch]
            py = gy_all[y0:y1]   # [h_patch]
            # Vectorized 2-D Gaussian over the patch — no inner Python loops
            d2 = (px[None, :] - xs[i])**2 + (py[:, None] - ys[i])**2  # [h, w]
            w  = np.exp(-d2 * inv_2s2)

            u_grid[y0:y1, x0:x1] += w * us[i]
            v_grid[y0:y1, x0:x1] += w * vs[i]
            w_grid[y0:y1, x0:x1] += w

        nz = w_grid > 1e-6
        u_grid[nz] /= w_grid[nz]
        v_grid[nz] /= w_grid[nz]
        w_norm = np.clip(w_grid / (w_grid.max() + 1e-10), 0.0, 1.0)

        return u_grid, v_grid, w_norm
