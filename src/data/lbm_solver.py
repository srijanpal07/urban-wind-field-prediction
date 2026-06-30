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

        # 3 & 4. Inlet/outlet BCs — all 4 sides, auto-selected by flow direction
        ux_in, uy_in = self.ux_in, self.uy_in

        # Left / Right BCs (x-component of inlet velocity)
        if ux_in > 0:
            # Inlet at LEFT (col=0) — Zou-He for +x inflow
            rho_in = (f[0,:,0] + f[2,:,0] + f[4,:,0]
                      + 2.0*(f[3,:,0] + f[6,:,0] + f[7,:,0])) / (1.0 - ux_in)
            rho_in = rho_in.clamp(0.5, 1.5).unsqueeze(1)      # [H, 1]
            ux_bc  = torch.full((self.H, 1), ux_in, device=self.device)
            uy_bc  = torch.full((self.H, 1), uy_in, device=self.device)
            f[:, :, 0]  = self._equilibrium(rho_in, ux_bc, uy_bc)[:, :, 0]
            f[:, :, -1] = f[:, :, -2]
        elif ux_in < 0:
            # Inlet at RIGHT (col=W-1) — Zou-He for -x inflow
            rho_in = (f[0,:,-1] + f[2,:,-1] + f[4,:,-1]
                      + 2.0*(f[1,:,-1] + f[5,:,-1] + f[8,:,-1])) / (1.0 + ux_in)
            rho_in = rho_in.clamp(0.5, 1.5).unsqueeze(1)      # [H, 1]
            ux_bc  = torch.full((self.H, 1), ux_in, device=self.device)
            uy_bc  = torch.full((self.H, 1), uy_in, device=self.device)
            f[:, :, -1] = self._equilibrium(rho_in, ux_bc, uy_bc)[:, :, -1]
            f[:, :, 0]  = f[:, :, 1]
        else:
            # No x-component: zero-gradient on both L/R sides
            f[:, :, 0]  = f[:, :, 1]
            f[:, :, -1] = f[:, :, -2]

        # Top / Bottom BCs (y-component of inlet velocity)
        if uy_in > 0:
            # Inlet at TOP (row=0) — Zou-He for +y inflow (downward in array)
            rho_in = (f[0,0,:] + f[1,0,:] + f[3,0,:]
                      + 2.0*(f[4,0,:] + f[7,0,:] + f[8,0,:])) / (1.0 - uy_in)
            rho_in = rho_in.clamp(0.5, 1.5).unsqueeze(0)      # [1, W]
            ux_bc  = torch.full((1, self.W), ux_in, device=self.device)
            uy_bc  = torch.full((1, self.W), uy_in, device=self.device)
            f[:, 0, :]  = self._equilibrium(rho_in, ux_bc, uy_bc)[:, 0, :]
            f[:, -1, :] = f[:, -2, :]
        elif uy_in < 0:
            # Inlet at BOTTOM (row=H-1) — Zou-He for -y inflow (upward in array)
            rho_in = (f[0,-1,:] + f[1,-1,:] + f[3,-1,:]
                      + 2.0*(f[2,-1,:] + f[5,-1,:] + f[6,-1,:])) / (1.0 + uy_in)
            rho_in = rho_in.clamp(0.5, 1.5).unsqueeze(0)      # [1, W]
            ux_bc  = torch.full((1, self.W), ux_in, device=self.device)
            uy_bc  = torch.full((1, self.W), uy_in, device=self.device)
            f[:, -1, :] = self._equilibrium(rho_in, ux_bc, uy_bc)[:, -1, :]
            f[:, 0, :]  = f[:, 1, :]
        # else uy_in == 0: top/bottom remain periodic from torch.roll (correct for pure x-flow)

        # 5. BGK Collision
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
            variation_period: int = 40, angle_variation: float = 15.0,
            angle_variation_period: int = 60, angle_end: float = None):
        """
        Run solver: warmup phase then collect snapshots.

        transient               : enable gusty wind (speed + direction disturbances)
        speed_variation         : ± fraction of base speed (e.g. 0.25 = ±25%)
        variation_period        : timesteps per speed-gust cycle
        angle_variation         : ± degrees of directional wobble around primary direction
        angle_variation_period  : timesteps per direction-wobble cycle (decorrelated from speed)
        angle_end               : if set, linearly rotate primary angle to this value (degrees)
        Returns arrays of shape [T, H, W] for u and v.
        """
        print(f"Warming up ({n_warmup} steps)...")
        for i in range(n_warmup):
            self.step()
            if (i+1) % 100 == 0:
                print(f"  Warmup step {i+1}/{n_warmup}")

        base_speed    = float(np.sqrt(self.ux_in**2 + self.uy_in**2))
        angle_start   = float(np.arctan2(self.uy_in, self.ux_in))
        angle_stop    = np.deg2rad(angle_end) if angle_end is not None else angle_start

        mode_parts = []
        if transient:
            mode_parts.append(f"transient (speed ±{speed_variation*100:.0f}%, dir ±{angle_variation:.0f}°)")
        if angle_end is not None:
            mode_parts.append(f"rotating {np.rad2deg(angle_start):.0f}°→{angle_end:.0f}°")
        mode_str = (" — " + ", ".join(mode_parts)) if mode_parts else ""
        print(f"Collecting {n_collect} snapshots (every {collect_every} steps){mode_str}...")

        n_total = n_collect * collect_every
        for i in range(n_total):
            t = i / max(n_total - 1, 1)          # 0 → 1 over collection window
            current_angle = angle_start + t * (angle_stop - angle_start)

            spd = base_speed
            if transient:
                # Speed gust: two overlapping sinusoids, ±speed_variation
                speed_factor = 1.0 + speed_variation * (
                    0.7 * np.sin(2 * np.pi * i / variation_period) +
                    0.3 * np.sin(2 * np.pi * i / (variation_period * 1.7))
                )
                spd = base_speed * float(np.clip(speed_factor, 0.2, 1.8))

                # Direction wobble: different period to decorrelate from speed gusts
                angle_perturb = np.deg2rad(angle_variation) * (
                    0.6 * np.sin(2 * np.pi * i / angle_variation_period) +
                    0.4 * np.sin(2 * np.pi * i / (angle_variation_period * 1.5))
                )
                current_angle += angle_perturb

            self.ux_in = float(spd * np.cos(current_angle))
            self.uy_in = float(spd * np.sin(current_angle))

            u, v = self.step()
            if i % collect_every == 0:
                self.snapshots_u.append(u.copy())
                self.snapshots_v.append(v.copy())

        u_arr = np.stack(self.snapshots_u, axis=0)  # [T, H, W]
        v_arr = np.stack(self.snapshots_v, axis=0)
        print(f"Done. Wind field shape: {u_arr.shape}")
        return u_arr, v_arr
