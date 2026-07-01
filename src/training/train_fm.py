"""
Training loop for the flow-matching spatiotemporal model.

WindSequenceDataset mirrors WindDataset (src/training/train_ufno.py) but the
target is a short future *sequence* [2, T_out, H, W] starting right after the
observation window, instead of a single frame at t0+obs_window+horizon. This
matches the FlowMatchingModel's spatiotemporal U-Net, which predicts a whole
forecast horizon at once rather than one snapshot.

Everything else (lazy per-condition loading, drone trajectory simulation,
obs_window, total_steps=2400 drone samples) is unchanged from train_ufno.py.
"""

import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.flow_matching import FlowMatchingModel
from src.data.drone_sampler import DroneSampler
from src.data.geometry import build_geo_channels

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


class WindSequenceDataset(Dataset):
    """
    Dataset of (input, target_sequence) pairs for flow-matching training.
    Samples randomly across N wind conditions (angle x speed combinations).

    input  : [6, H, W]        - geometry + sparse observations at time t0..t0+obs_window
    target : [2, T_out, H, W] - dense u, v sequence for t0+obs_window .. t0+obs_window+T_out-1

    Two loading modes:
      - In-memory : pass u_all/v_all as [N, T, H, W] arrays
      - Lazy      : pass condition_files as list of per-condition npz paths
                    (each file decompressed on demand per worker)

    Cache-friendly indexing follows the same block-structured scheme as
    WindDataset in train_ufno.py.
    """

    def __init__(self, u_all=None, v_all=None, obstacle_mask=None,
                 drone_sampler: DroneSampler = None, waypoints=None,
                 T_out: int = 20, n_samples: int = 1000,
                 obs_window: int = 30, condition_files=None):
        self.condition_files = condition_files
        self.mask = obstacle_mask
        self.sampler = drone_sampler
        self.waypoints = waypoints
        self.T_out = T_out
        self.n_samples = n_samples
        self.obs_window = obs_window
        self.epoch = 0
        # Geometry conditioning (binary mask + SDF) is fixed for the whole
        # dataset (single city geometry) — compute once, not per __getitem__.
        self.geo_in = build_geo_channels(obstacle_mask) if obstacle_mask is not None else None

        if condition_files is not None:
            self.u_all = None
            self.v_all = None
            self.N = len(condition_files)
            d0 = np.load(condition_files[0])
            self.T, self.H, self.W = d0['u'].shape
        else:
            self.u_all = u_all
            self.v_all = v_all
            self.N, self.T, self.H, self.W = u_all.shape
        self.G = self.H
        self.spcc = max(1, self.n_samples // self.N)  # samples per condition
        self.cond_order = np.arange(self.N)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        rng_order = np.random.default_rng(epoch + 99991)
        self.cond_order = rng_order.permutation(self.N)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        block = (idx // self.spcc) % self.N
        cond_idx = int(self.cond_order[block])

        rng = np.random.default_rng(idx + self.epoch * self.n_samples)

        if self.condition_files is not None:
            u_series, v_series = _load_condition_cached(self.condition_files[cond_idx])
        else:
            u_series = self.u_all[cond_idx]   # [T, H, W]
            v_series = self.v_all[cond_idx]
        T = self.T

        # Random start time; need room for obs_window + T_out future frames.
        max_t0 = T - self.obs_window - self.T_out
        t0 = rng.integers(0, max(1, max_t0))
        t_seq_start = t0 + self.obs_window

        # Vary drone path per sample so the model can't memorize condition->path
        path_seed = int(rng.integers(0, 500))
        waypoints = self.sampler.make_traverse_path(seed=path_seed)

        # Simulate drone trajectory: 2400 samples ~ 10 Hz x 240 s (4-minute leg)
        total_steps = 2400
        x_path, y_path = self.sampler.interpolate_path(waypoints, total_steps)

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

        # Target sequence: dense u,v for T_out frames following the obs window.
        # Clamp the end index to avoid overrun on conditions with short T.
        t_end = min(t_seq_start + self.T_out, T)
        u_seq = u_series[t_seq_start:t_end].astype(np.float32)
        v_seq = v_series[t_seq_start:t_end].astype(np.float32)
        n_have = u_seq.shape[0]
        if n_have < self.T_out:
            # Pad by repeating the last available frame (rare edge case).
            pad = self.T_out - n_have
            u_seq = np.concatenate([u_seq, np.repeat(u_seq[-1:], pad, axis=0)], axis=0)
            v_seq = np.concatenate([v_seq, np.repeat(v_seq[-1:], pad, axis=0)], axis=0)

        y_out = np.stack([u_seq, v_seq], axis=0)  # [2, T_out, H, W]
        y_out[:, :, self.mask] = 0.0

        mask_in = self.geo_in  # [2, H, W]: binary mask + SDF, precomputed in __init__

        return (torch.tensor(x_in), torch.tensor(y_out), torch.tensor(mask_in))


def train_fm(u_all=None, v_all=None, obstacle_mask=None,
             save_path='outputs/flow_matching/fm_model.pth',
             T_out=20, n_epochs=200, batch_size=4, lr=1e-4,
             obs_window=30, device='cuda', condition_files=None,
             resume_path=None, hidden=64, n_levels=4, t_emb_dim=256,
             use_amp=True, use_checkpoint=True, use_physics_prior=True,
             lambda_div=0.1, lambda_solid=0.1, lambda_spectral=0.0,
             lambda_obs=1.0):
    """
    Train FlowMatchingModel on multi-condition LBM data.

    Two modes:
      - condition_files: list of per-condition compressed npz paths (lazy loading)
      - u_all/v_all    : [N, T, H, W] arrays already in memory

    resume_path: if set, loads model weights + history and continues from the
    last completed epoch. n_epochs is the TOTAL target (not additional epochs).
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
        grid_size = H
    print(f"Conditions: {N}  |  Grid: {grid_size}x{grid_size}  |  T_out: {T_out}")

    sampler = DroneSampler(grid_size=grid_size, obstacle_mask=obstacle_mask)
    waypoints = sampler.make_traverse_path(seed=0)

    n_samples = max(600, 30 * N)
    dataset = WindSequenceDataset(
        u_all, v_all, obstacle_mask, sampler, waypoints,
        T_out=T_out, n_samples=n_samples, obs_window=obs_window,
        condition_files=condition_files)

    spcc = dataset.spcc
    val_per_cond = max(1, spcc // 10)
    train_idx, val_idx = [], []
    for c in range(dataset.N):
        base = c * spcc
        train_idx.extend(range(base, base + spcc - val_per_cond))
        val_idx.extend(range(base + spcc - val_per_cond, base + spcc))

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    n_train, n_val = len(train_ds), len(val_ds)
    print(f"Train samples: {n_train}  |  Val samples: {n_val}  "
          f"({val_per_cond}/{spcc} per condition)")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=False, num_workers=4,
                               persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=2,
                             persistent_workers=True, pin_memory=True)

    model = FlowMatchingModel(T_out=T_out, obs_channels=6, geo_channels=8,
                               hidden=hidden, n_levels=n_levels,
                               t_emb_dim=t_emb_dim,
                               grid_size=grid_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

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
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == torch.device('cuda').type)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    amp_ctx = lambda: torch.amp.autocast('cuda', enabled=use_amp and device.type == torch.device('cuda').type)

    print(f"AMP: {use_amp}  |  Grad checkpoint: {use_checkpoint}  |  Physics prior: {use_physics_prior}  |  "
          f"lambda_div: {lambda_div}  |  lambda_solid: {lambda_solid}  |  "
          f"lambda_obs: {lambda_obs}  |  lambda_spectral: {lambda_spectral}")

    for epoch in range(start_epoch, n_epochs):
        dataset.set_epoch(epoch)

        model.train()
        train_loss = train_data = train_div = train_solid = train_spectral = train_obs = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{n_epochs}",
                   unit='batch', leave=False,
                   bar_format='{l_bar}{bar:30}{r_bar}')
        for x_in, y_seq, mask_in in bar:
            x_in = x_in.to(device)
            y_seq = y_seq.to(device)
            mask_in = mask_in.to(device)

            optimizer.zero_grad()
            with amp_ctx():
                loss_dict = model.flow_match_loss(y_seq, x_in, mask_in, solid_mask,
                                                   use_checkpoint=use_checkpoint,
                                                   use_physics_prior=use_physics_prior,
                                                   lambda_div=lambda_div, lambda_solid=lambda_solid,
                                                   lambda_spectral=lambda_spectral,
                                                   lambda_obs=lambda_obs)
            loss = loss_dict['total']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_data += loss_dict['data'].item()
            train_div += loss_dict['div'].item()
            train_solid += loss_dict['solid'].item()
            train_spectral += loss_dict['spectral'].item()
            train_obs += loss_dict['obs'].item()
            bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss /= len(train_loader)
        train_data /= len(train_loader)
        train_div /= len(train_loader)
        train_solid /= len(train_loader)
        train_spectral /= len(train_loader)
        train_obs /= len(train_loader)

        dataset.set_epoch(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_in, y_seq, mask_in in val_loader:
                x_in = x_in.to(device)
                y_seq = y_seq.to(device)
                mask_in = mask_in.to(device)
                with amp_ctx():
                    loss_dict = model.flow_match_loss(y_seq, x_in, mask_in, solid_mask,
                                                       use_physics_prior=use_physics_prior,
                                                       lambda_div=lambda_div, lambda_solid=lambda_solid,
                                                       lambda_spectral=lambda_spectral,
                                                       lambda_obs=lambda_obs)
                val_loss += loss_dict['total'].item()

        val_loss /= len(val_loader)
        scheduler.step()

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        saved = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(),
                        'history': history,
                        'T_out': T_out,
                        'grid_size': grid_size,
                        'hidden': hidden,
                        'n_levels': n_levels,
                        't_emb_dim': t_emb_dim,
                        'use_physics_prior': use_physics_prior,
                        'lambda_div': lambda_div,
                        'lambda_solid': lambda_solid,
                        'lambda_spectral': lambda_spectral,
                        'lambda_obs': lambda_obs}, save_path)
            saved = '  ✓'

        spectral_str = f" spectral={train_spectral:.4f}" if lambda_spectral > 0 else ""
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train: {train_loss:.4f} (data={train_data:.4f} div={train_div:.4f} "
              f"solid={train_solid:.4f} obs={train_obs:.4f}{spectral_str}) | "
              f"Val: {val_loss:.4f}{saved}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    return model, history
