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

# Per-worker LRU-1 cache: avoids re-decompressing the same condition file
# across consecutive __getitem__ calls in the same DataLoader worker process.
_worker_condition_cache: dict = {}


def _load_condition_cached(path: str):
    """Load u, v arrays from a compressed condition npz with per-worker caching."""
    if path not in _worker_condition_cache:
        _worker_condition_cache.clear()   # keep at most one entry per worker
        d = np.load(path)
        _worker_condition_cache[path] = (np.array(d['u']), np.array(d['v']))
    return _worker_condition_cache[path]


class WindDataset(Dataset):
    """
    Dataset of (input, target) pairs for wind prediction.
    Samples randomly across N wind conditions (angle × speed combinations).

    input  : [6, H, W] - geometry + sparse observations at time t
    target : [2, H, W] - dense u, v at time t+horizon

    Two loading modes:
      - In-memory : pass u_all/v_all as [N, T, H, W] arrays (run_pipeline.py)
      - Lazy      : pass condition_files as list of per-condition npz paths
                    (train_model.py; each file decompressed on demand per worker)

    Cache-friendly indexing:
      idx maps to cond_idx = (idx // spcc) % N so that consecutive indices in
      the same batch always target the same condition file. Combined with
      shuffle=False in the DataLoader, this reduces decompressions from ~32
      per batch (one per sample) down to ~2 (one per condition boundary),
      keeping the GPU fed with minimal I/O stalls.
      Epoch-to-epoch diversity is preserved by seeding rng with (idx + epoch*n).
    """
    def __init__(self, u_all, v_all, obstacle_mask,
                 drone_sampler: DroneSampler, waypoints,
                 horizon: int = 10, n_samples: int = 1000,
                 obs_window: int = 15, condition_files=None):
        self.condition_files = condition_files
        self.mask = obstacle_mask
        self.sampler = drone_sampler
        self.waypoints = waypoints
        self.horizon = horizon
        self.n_samples = n_samples
        self.obs_window = obs_window
        self.epoch = 0  # updated each epoch so rng varies while cond_idx stays stable

        if condition_files is not None:
            self.u_all = None
            self.v_all = None
            self.N = len(condition_files)
            d0 = np.load(condition_files[0])
            self.T, self.H, self.W = d0['u'].shape
        else:
            # u_all, v_all: [N, T, H, W]
            self.u_all = u_all
            self.v_all = v_all
            self.N, self.T, self.H, self.W = u_all.shape
        self.G = self.H
        self.spcc = max(1, self.n_samples // self.N)  # samples per condition
        # Condition order for this epoch — shuffled each epoch to prevent
        # directional bias from fixed block ordering with cosine annealing.
        self.cond_order = np.arange(self.N)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        rng_order = np.random.default_rng(epoch + 99991)
        self.cond_order = rng_order.permutation(self.N)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Block-structured: all idx in [c*spcc, (c+1)*spcc) map to the same
        # condition. cond_order shuffles which condition each block targets,
        # changing each epoch to avoid directional bias from fixed ordering.
        block = (idx // self.spcc) % self.N
        cond_idx = int(self.cond_order[block])

        # Epoch-dependent rng: same idx in different epochs → different t0/path.
        rng = np.random.default_rng(idx + self.epoch * self.n_samples)

        # Load condition data
        if self.condition_files is not None:
            u_series, v_series = _load_condition_cached(self.condition_files[cond_idx])
        else:
            u_series = self.u_all[cond_idx]   # [T, H, W]
            v_series = self.v_all[cond_idx]
        T = self.T

        # Random start time (ensure we have room for horizon)
        t0 = rng.integers(0, T - self.horizon - self.obs_window)
        t_target = t0 + self.obs_window + self.horizon

        # Vary drone path per sample so the model can't memorize condition→path
        path_seed = int(rng.integers(0, 500))
        waypoints = self.sampler.make_traverse_path(seed=path_seed)

        # Simulate drone trajectory: 80 samples ≈ 10 Hz × 8 s traverse
        total_steps = 80
        x_path, y_path = self.sampler.interpolate_path(waypoints, total_steps)

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


def train(u_all=None, v_all=None, obstacle_mask=None,
          save_path='outputs/wind_fno.pth',
          grid_size=256, n_epochs=50, batch_size=8, lr=1e-3,
          horizon=10, device='cuda', condition_files=None,
          resume_path=None):
    """
    Train WindFNO on multi-condition LBM data.

    Two modes:
      - condition_files: list of per-condition compressed npz paths (lazy loading)
      - u_all/v_all    : [N, T, H, W] arrays already in memory (run_pipeline.py)

    resume_path: if set, loads model weights + history and continues from the
    last completed epoch. n_epochs is the TOTAL target (not additional epochs).
    A fresh cosine schedule runs over the remaining epochs at the given lr.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    if condition_files is not None:
        N = len(condition_files)
        d0 = np.load(condition_files[0])
        _, H, _ = d0['u'].shape
        grid_size = H
    else:
        N, _, H, _ = u_all.shape
        grid_size = H   # always use actual data shape; ignore any passed-in value
    print(f"Conditions: {N}  |  Grid: {grid_size}×{grid_size}")

    # Setup drone sampler — path is varied per sample in WindDataset.__getitem__
    sampler = DroneSampler(grid_size=grid_size, obstacle_mask=obstacle_mask)
    waypoints = sampler.make_traverse_path(seed=0)  # default; overridden per sample

    # Scale sample count with number of conditions
    n_samples = max(600, 30 * N)
    dataset = WindDataset(u_all, v_all, obstacle_mask,
                          sampler, waypoints, horizon=horizon,
                          n_samples=n_samples, obs_window=30,
                          condition_files=condition_files)

    # Stratified split: take the last val_per_cond indices from each condition's
    # block so every condition is represented in both train and val.
    spcc = dataset.spcc
    val_per_cond = max(1, spcc // 10)
    train_idx, val_idx = [], []
    for c in range(dataset.N):
        base = c * spcc
        train_idx.extend(range(base, base + spcc - val_per_cond))
        val_idx.extend(range(base + spcc - val_per_cond, base + spcc))

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)

    n_train, n_val = len(train_ds), len(val_ds)
    print(f"Train samples: {n_train}  |  Val samples: {n_val}  "
          f"({val_per_cond}/{spcc} per condition)")

    # shuffle=False preserves block ordering → same condition per batch → cache hits.
    # set_epoch() changes rng each epoch for t0/path diversity across epochs.
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=False, num_workers=4,
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

    # Resume: load weights + history, then run a fresh cosine schedule over
    # the remaining epochs at the requested lr (acts as a warm restart).
    start_epoch = 0
    history = {'train': [], 'val': []}
    best_val = float('inf')

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        history = ckpt.get('history', {'train': [], 'val': []})
        start_epoch = len(history['train'])
        best_val = min(history['val']) if history['val'] else float('inf')
        print(f"Resumed from epoch {start_epoch}  (best val: {best_val:.4f})")
        if start_epoch >= n_epochs:
            print(f"Already at {start_epoch} epochs — increase --epochs above {start_epoch} to continue.")
            return model, history

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    remaining = max(1, n_epochs - start_epoch)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    solid_mask = torch.tensor(obstacle_mask, dtype=torch.bool).to(device)

    for epoch in range(start_epoch, n_epochs):
        # Train: advance epoch so rng varies → new t0/paths each pass
        dataset.set_epoch(epoch)

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

        # Validate — pin to epoch=0 so the same t0/paths are drawn every epoch,
        # giving a stable signal for best-checkpoint selection.
        dataset.set_epoch(0)
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
