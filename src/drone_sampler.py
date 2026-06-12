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
    noise_std      : float, std of Gaussian noise on u/v (in m/s units)
    """

    def __init__(self, grid_size: int = 128, obstacle_mask: np.ndarray = None,
                 noise_std: float = 0.05):
        self.G = grid_size
        self.obstacle_mask = obstacle_mask
        self.noise_std = noise_std

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
            u_obs[i] = u_i + np.random.normal(0, self.noise_std)
            v_obs[i] = v_i + np.random.normal(0, self.noise_std)

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
        u_grid = np.zeros((H, W))
        v_grid = np.zeros((H, W))
        w_grid = np.zeros((H, W))

        for i in range(len(obs['x'])):
            xi = obs['x'][i]
            yi = obs['y'][i]
            ui = obs['u_obs'][i]
            vi = obs['v_obs'][i]
            if np.isnan(ui) or np.isnan(vi):
                continue

            # Gaussian splat radius
            x0 = max(0, int(xi - 3*sigma))
            x1 = min(W, int(xi + 3*sigma) + 1)
            y0 = max(0, int(yi - 3*sigma))
            y1 = min(H, int(yi + 3*sigma) + 1)

            for gy in range(y0, y1):
                for gx in range(x0, x1):
                    d2 = (gx - xi)**2 + (gy - yi)**2
                    w = np.exp(-d2 / (2 * sigma**2))
                    u_grid[gy, gx] += w * ui
                    v_grid[gy, gx] += w * vi
                    w_grid[gy, gx] += w

        mask = w_grid > 1e-6
        u_grid[mask] /= w_grid[mask]
        v_grid[mask] /= w_grid[mask]
        w_grid_norm = np.clip(w_grid / (w_grid.max() + 1e-10), 0, 1)

        return u_grid, v_grid, w_grid_norm
