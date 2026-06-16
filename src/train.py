"""
Training module for WindFNO.
Trains on multi-condition LBM data with synthetic drone observations.
Call via train_model.py (multi-condition) or run_pipeline.py (single condition).
"""

import torch
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model import WindFNO, prepare_input, nll_loss
from .drone_sampler import DroneSampler


class WindDataset(Dataset):
    """
    Dataset of (input, target) pairs for wind prediction.
    Samples randomly across N wind conditions (angle × speed combinations).

    input  : [6, H, W] - geometry + sparse observations at time t
    target : [2, H, W] - dense u, v at time t+horizon
    """
    def __init__(self, u_all, v_all, obstacle_mask,
                 drone_sampler: DroneSampler, waypoints,
                 horizon: int = 10, n_samples: int = 1000,
                 obs_window: int = 15):
        # u_all, v_all: [N, T, H, W]
        self.u_all = u_all
        self.v_all = v_all
        self.mask = obstacle_mask
        self.sampler = drone_sampler
        self.waypoints = waypoints
        self.horizon = horizon
        self.n_samples = n_samples
        self.obs_window = obs_window
        self.N, self.T, self.H, self.W = u_all.shape
        self.G = self.H

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)

        # Pick a random wind condition
        cond_idx = int(rng.integers(0, self.N))
        u_series = self.u_all[cond_idx]   # [T, H, W]
        v_series = self.v_all[cond_idx]
        T = self.T

        # Random start time (ensure we have room for horizon)
        t0 = rng.integers(0, T - self.horizon - self.obs_window)
        t_target = t0 + self.obs_window + self.horizon

        # Simulate drone trajectory: 80 samples ≈ 10 Hz × 8 s traverse
        total_steps = 80
        x_path, y_path = self.sampler.interpolate_path(self.waypoints, total_steps)

        # Add small random jitter to path for diversity
        noise_scale = self.G * 0.02
        x_path = x_path + rng.normal(0, noise_scale, total_steps)
        y_path = y_path + rng.normal(0, noise_scale, total_steps)
        x_path = np.clip(x_path, 0, self.G - 1)
        y_path = np.clip(y_path, 0, self.G - 1)

        t_indices = np.linspace(t0, t0 + self.obs_window - 1, total_steps).astype(int)

        obs = self.sampler.sample_field(
            u_series, v_series, x_path, y_path, t_indices)

        obs_u_grid, obs_v_grid, obs_confidence = self.sampler.obs_to_grid(
            obs, self.G, sigma=3.0)

        # Build input tensor [6, H, W]
        H, W = self.H, self.W
        ys = np.linspace(0, 1, H)
        xs = np.linspace(0, 1, W)
        xg, yg = np.meshgrid(xs, ys)

        x_in = np.stack([
            self.mask.astype(np.float32),
            obs_u_grid.astype(np.float32),
            obs_v_grid.astype(np.float32),
            obs_confidence.astype(np.float32),
            xg.astype(np.float32),
            yg.astype(np.float32)
        ], axis=0)  # [6, H, W]

        # Target: dense wind field at t_target from this condition
        y_out = np.stack([
            u_series[t_target].astype(np.float32),
            v_series[t_target].astype(np.float32)
        ], axis=0)  # [2, H, W]
        y_out[:, self.mask] = 0.0

        return torch.tensor(x_in), torch.tensor(y_out)


def train(u_all, v_all, obstacle_mask, save_path='outputs/wind_fno.pth',
          grid_size=256, n_epochs=50, batch_size=8, lr=1e-3,
          horizon=10, device='cuda'):
    """
    Train WindFNO on multi-condition LBM data.
    u_all, v_all: [N, T, H, W] — N wind conditions stacked along axis 0.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    N, _, H, _ = u_all.shape
    grid_size = H   # always use actual data shape; ignore any passed-in value
    print(f"Conditions: {N}  |  Grid: {grid_size}×{grid_size}")

    # Setup drone sampler and path (polar noise; traverse from edge to edge)
    sampler = DroneSampler(grid_size=grid_size, obstacle_mask=obstacle_mask)
    waypoints = sampler.make_traverse_path(seed=42)

    # Scale sample count with number of conditions
    n_samples = max(600, 30 * N)
    dataset = WindDataset(u_all, v_all, obstacle_mask,
                          sampler, waypoints, horizon=horizon,
                          n_samples=n_samples, obs_window=30)

    n_val = max(60, n_samples // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4,
                              persistent_workers=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2,
                              persistent_workers=True, pin_memory=True)

    # Model
    modes = max(20, grid_size // 8)
    model = WindFNO(in_channels=6, out_channels=4,
                    hidden=48, modes=modes, n_layers=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    solid_mask = torch.tensor(obstacle_mask, dtype=torch.bool).to(device)
    best_val = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{n_epochs}",
                   unit='batch', leave=False,
                   bar_format='{l_bar}{bar:30}{r_bar}')
        for x_in, y_true in bar:
            x_in   = x_in.to(device)
            y_true = y_true.to(device)

            u_pred, v_pred, sigma_u, sigma_v = model(x_in)

            u_true = y_true[:, 0:1]
            v_true = y_true[:, 1:2]

            loss = nll_loss(u_pred, v_pred, sigma_u, sigma_v,
                            u_true, v_true, solid_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_mse  = 0.0
        with torch.no_grad():
            for x_in, y_true in val_loader:
                x_in   = x_in.to(device)
                y_true = y_true.to(device)
                u_pred, v_pred, sigma_u, sigma_v = model(x_in)
                u_true = y_true[:, 0:1]
                v_true = y_true[:, 1:2]
                loss = nll_loss(u_pred, v_pred, sigma_u, sigma_v,
                                u_true, v_true, solid_mask)
                val_loss += loss.item()
                mse = (((u_pred - u_true)**2 + (v_pred - v_true)**2) *
                       (~solid_mask).float().to(device)).mean()
                val_mse += mse.item()

        val_loss /= len(val_loader)
        val_mse  /= len(val_loader)
        scheduler.step()

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        saved = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(),
                        'history': history,
                        'grid_size': grid_size,
                        'horizon': horizon,
                        'modes': modes}, save_path)
            saved = '  ✓'

        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"MSE: {val_mse:.6f}{saved}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    return model, history
