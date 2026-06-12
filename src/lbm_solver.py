"""
2D Lattice Boltzmann Method (LBM) Solver - GPU Native
D2Q9 scheme with BGK collision operator
Handles solid obstacles (buildings) from binary mask
"""

import torch
import numpy as np


# D2Q9 lattice constants
W = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=torch.float32)
CX = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float32)
CY = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=torch.float32)
# Opposite direction indices for bounce-back
OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long)


class LBMSolver:
    """
    2D LBM solver on GPU.
    obstacle_mask: bool tensor [H, W], True = solid building
    inlet_speed:   float, inlet velocity magnitude (m/s in scaled units)
    inlet_angle:   float, degrees, 0=East, 90=North
    tau:           float, relaxation time (0.6-0.9 for stability)
    """

    def __init__(self, obstacle_mask: np.ndarray, inlet_speed: float = 0.1,
                 inlet_angle: float = 0.0, tau: float = 0.7,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"LBM running on: {self.device}")

        self.H, self.W = obstacle_mask.shape
        self.tau = tau
        self.nu = (tau - 0.5) / 3.0  # kinematic viscosity in LB units

        # Move lattice constants to device
        self.w   = W.to(self.device)
        self.cx  = CX.to(self.device)
        self.cy  = CY.to(self.device)
        self.opp = OPP.to(self.device)

        # Obstacle mask [H, W]
        self.solid = torch.tensor(obstacle_mask, dtype=torch.bool, device=self.device)

        # Inlet velocity components
        angle_rad = np.deg2rad(inlet_angle)
        self.ux_in = inlet_speed * np.cos(angle_rad)
        self.uy_in = inlet_speed * np.sin(angle_rad)

        # Initialise distribution functions [9, H, W]
        self.f = self._equilibrium(
            torch.ones(self.H, self.W, device=self.device),
            torch.full((self.H, self.W), self.ux_in, device=self.device),
            torch.full((self.H, self.W), self.uy_in, device=self.device)
        )

        # Storage for time series
        self.snapshots_u = []
        self.snapshots_v = []

    def _equilibrium(self, rho, ux, uy):
        """Compute equilibrium distribution f_eq [9, H, W]"""
        feq = torch.zeros(9, self.H, self.W, device=self.device)
        for i in range(9):
            cu = self.cx[i] * ux + self.cy[i] * uy
            feq[i] = self.w[i] * rho * (
                1.0 + 3.0 * cu
                + 4.5 * cu**2
                - 1.5 * (ux**2 + uy**2)
            )
        return feq

    def _stream(self, f):
        """Streaming step - shift distributions along lattice directions"""
        f_new = torch.zeros_like(f)
        for i in range(9):
            dx = int(self.cx[i].item())
            dy = int(self.cy[i].item())
            f_new[i] = torch.roll(torch.roll(f[i], dx, dims=1), dy, dims=0)
        return f_new

    def _macroscopic(self, f):
        """Compute macroscopic density and velocity from distributions"""
        rho = f.sum(dim=0).clamp(min=1e-4)
        ux  = (f * self.cx[:, None, None]).sum(dim=0) / rho
        uy  = (f * self.cy[:, None, None]).sum(dim=0) / rho
        ux  = ux.clamp(-0.3, 0.3)
        uy  = uy.clamp(-0.3, 0.3)
        return rho, ux, uy

    def step(self):
        """One full LBM timestep: stream → BC → collide"""
        # 1. Streaming
        f = self._stream(self.f)

        # 2. Bounce-back on solid nodes (no-slip walls)
        f_pre = self.f.clone()
        for i in range(9):
            f[i][self.solid] = f_pre[self.opp[i]][self.solid]

        # 3 & 4. Inlet/outlet BCs — side depends on flow direction
        ux_full = torch.full((self.H,), self.ux_in, device=self.device)
        uy_full = torch.full((self.H,), self.uy_in, device=self.device)
        if self.ux_in >= 0:
            # Inlet at LEFT (col=0), outlet at RIGHT — Zou-He for +x inflow
            rho_in = (f[0,:,0] + f[2,:,0] + f[4,:,0]
                      + 2.0*(f[3,:,0] + f[6,:,0] + f[7,:,0])) / (1.0 - self.ux_in)
            rho_in = rho_in.clamp(0.5, 1.5)
            f[:, :, 0]  = self._equilibrium(rho_in, ux_full, uy_full)[:, :, 0]
            f[:, :, -1] = f[:, :, -2]   # outlet: zero-gradient
        else:
            # Inlet at RIGHT (col=W-1), outlet at LEFT — Zou-He for -x inflow
            rho_in = (f[0,:,-1] + f[2,:,-1] + f[4,:,-1]
                      + 2.0*(f[1,:,-1] + f[5,:,-1] + f[8,:,-1])) / (1.0 + self.ux_in)
            rho_in = rho_in.clamp(0.5, 1.5)
            f[:, :, -1] = self._equilibrium(rho_in, ux_full, uy_full)[:, :, -1]
            f[:, :, 0]  = f[:, :, 1]    # outlet: zero-gradient

        # 5. Top/bottom: torch.roll in _stream already gives periodic BCs — no override needed

        # 6. BGK Collision
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)
        self.f = f - (f - feq) / self.tau
        self.f.clamp_(min=0.0)  # prevent negative distributions

        # Zero velocity inside solids
        self.f[:, self.solid] = self._equilibrium(
            torch.ones(self.H, self.W, device=self.device),
            torch.zeros(self.H, self.W, device=self.device),
            torch.zeros(self.H, self.W, device=self.device)
        )[:, self.solid]

        return ux.cpu().numpy(), uy.cpu().numpy()

    def run(self, n_warmup: int = 500, n_collect: int = 200, collect_every: int = 5,
            transient: bool = False, speed_variation: float = 0.25,
            variation_period: int = 40):
        """
        Run solver: warmup phase then collect snapshots.

        transient        : vary inlet speed over time (gusty wind)
        speed_variation  : ± fraction of base speed (e.g. 0.25 = ±25%)
        variation_period : timesteps per gust cycle
        Returns arrays of shape [T, H, W] for u and v.
        """
        print(f"Warming up ({n_warmup} steps)...")
        for i in range(n_warmup):
            self.step()
            if (i+1) % 100 == 0:
                print(f"  Warmup step {i+1}/{n_warmup}")

        base_speed = float(np.sqrt(self.ux_in**2 + self.uy_in**2))
        angle_rad  = float(np.arctan2(self.uy_in, self.ux_in))

        mode_str = (f" — transient mode (±{speed_variation*100:.0f}%, "
                    f"period={variation_period} steps)") if transient else ""
        print(f"Collecting {n_collect} snapshots (every {collect_every} steps){mode_str}...")

        for i in range(n_collect * collect_every):
            if transient:
                # Two-frequency gust profile for realistic variation
                factor = 1.0 + speed_variation * (
                    0.7 * np.sin(2 * np.pi * i / variation_period) +
                    0.3 * np.sin(2 * np.pi * i / (variation_period * 1.7))
                )
                factor = float(np.clip(factor, 0.2, 1.8))
                spd = base_speed * factor
                self.ux_in = spd * np.cos(angle_rad)
                self.uy_in = spd * np.sin(angle_rad)

            u, v = self.step()
            if i % collect_every == 0:
                self.snapshots_u.append(u.copy())
                self.snapshots_v.append(v.copy())

        u_arr = np.stack(self.snapshots_u, axis=0)  # [T, H, W]
        v_arr = np.stack(self.snapshots_v, axis=0)
        print(f"Done. Wind field shape: {u_arr.shape}")
        return u_arr, v_arr
