"""
Spatiotemporal Flow-Matching model for urban wind field prediction.

Predicts a short future wind sequence u(x,y,t), v(x,y,t) conditioned on sparse
drone observations and urban geometry, using a continuous-time flow-matching
objective (rectified-flow style: linear interpolation between Gaussian noise
and the target sequence).

Input channels:
  obs_channels [B, 6, H, W] : geometry mask, obs_u, obs_v, obs_confidence,
                              x coord, y coord (same layout as WindFNO input)
  geo_channels [B, 2, H, W]: binary obstacle mask + signed distance field
                              (see src/data/geometry.build_geo_channels),
                              passed separately to the GeometryEncoder for
                              richer, smoother-gradient conditioning features

Output:
  v_theta [B, 2, T_out, H, W] : predicted velocity field (flow-matching sense,
                                 i.e. d/ds of the interpolation path)
"""

import math
from math import ceil, log2

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """s in [0,1] -> [B, dim]. Sinusoidal frequencies + 2-layer MLP."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: [B]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = s.float()[:, None] * freqs[None, :] * 1000.0  # scale s up into a useful frequency range
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class GeometryEncoder(nn.Module):
    """[B,2,H,W] (binary mask + SDF) -> [B,8,H,W]. Conv2d(2->16,GN,GELU) -> Conv2d(16->8,GN,GELU)."""

    def __init__(self, in_channels: int = 2, out_channels: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.GroupNorm(_safe_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.net(mask)


def _safe_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock2D(nn.Module):
    """
    Pre-norm residual block:
      GroupNorm -> GELU -> Conv2d(3x3) -> GroupNorm -> GELU -> Conv2d(3x3)
    + skip (1x1 conv if in_ch != out_ch)
    + timestep scale: Linear(t_emb_dim -> out_ch) as multiplicative scale
      after the first GroupNorm (scale = 1 + linear(t_emb))
    """

    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int = 256):
        super().__init__()
        self.norm1 = nn.GroupNorm(_safe_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_safe_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.GELU()
        self.t_scale = nn.Linear(t_emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W], t_emb: [B, t_emb_dim]
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Timestep scale (FiLM-style): applied after the second GroupNorm,
        # whose channel count (out_ch) matches t_scale's output.
        h = self.norm2(h)
        scale = 1.0 + self.t_scale(t_emb)[:, :, None, None]
        h = h * scale
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SpatialAttnBlock(nn.Module):
    """
    Multi-head self-attention on (H, W) tokens.
    GroupNorm -> reshape [B,C,H,W] -> [B,H*W,C] -> MHA -> reshape back -> residual.
    """

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(_safe_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class TemporalAttnBlock(nn.Module):
    """
    Multi-head self-attention on T_out tokens.
    x: [B, C, T, H, W]
    Reshape to [B*H*W, T, C] -> MHA -> reshape to [B,C,T,H,W] -> residual.
    """

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(_safe_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        # GroupNorm expects [N, C, *] - reshape to per-frame first for normalization
        h = self.norm(x.reshape(B, C, T * H, W)).reshape(B, C, T, H, W)
        # [B,C,T,H,W] -> [B,H,W,T,C] -> [B*H*W, T, C]
        h = h.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)  # [B, C, T, H, W]
        return x + h


def _per_frame(x: torch.Tensor, fn):
    """Apply a 2D op to a [B,C,T,H,W] tensor by folding T into the batch dim."""
    B, C, T, H, W = x.shape
    x2 = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    y2 = fn(x2)
    Cy = y2.shape[1]
    Hy, Wy = y2.shape[2], y2.shape[3]
    y = y2.reshape(B, T, Cy, Hy, Wy).permute(0, 2, 1, 3, 4)
    return y


class SpatiotemporalUNet(nn.Module):
    """
    U-Net backbone operating on a 5D tensor [B, C, T, H, W].

    in_channels = 2 (u,v of x_t) + 6 (obs) + 8 (geo) = 16
    """

    def __init__(self, in_channels: int, T_out: int, hidden: int = 64,
                 n_levels: int = 4, t_emb_dim: int = 256, grid_size: int = 256):
        super().__init__()
        self.T_out = T_out
        self.hidden = hidden
        self.n_levels = n_levels
        self.t_emb_dim = t_emb_dim

        self.t_embed = SinusoidalEmbedding(t_emb_dim)

        # Lift: per-frame Conv2d(in_channels, hidden, 1)
        self.lift = nn.Conv2d(in_channels, hidden, 1)

        # Spatial attention is O(H²W²) memory — cap it to ≤64×64 feature maps.
        # Attention at encoder level i is applied BEFORE the stride-2 downsample,
        # so the feature map at level i has spatial size grid_size / 2^i.
        # We want grid_size / 2^i ≤ 64  →  i ≥ log2(grid_size / 64).
        attn_start = max(0, int(ceil(log2(max(grid_size / 64, 1)))))

        # Encoder
        self.enc_res = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        ch = hidden
        self.enc_channels = []
        for i in range(n_levels):
            self.enc_res.append(ResBlock2D(ch, ch, t_emb_dim))
            self.enc_attn.append(SpatialAttnBlock(ch) if i >= attn_start else nn.Identity())
            self.enc_channels.append(ch)
            out_ch = ch * 2
            self.enc_down.append(nn.Conv2d(ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch

        # Bottleneck
        self.bottleneck_res = ResBlock2D(ch, ch, t_emb_dim)
        self.bottleneck_attn = SpatialAttnBlock(ch)
        self.bottleneck_temporal = TemporalAttnBlock(ch)
        self.bottleneck_ch = ch

        # Decoder
        self.dec_up = nn.ModuleList()
        self.dec_res = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        for i in range(n_levels):
            level = n_levels - 1 - i
            skip_ch = self.enc_channels[level]
            up_ch = ch // 2
            self.dec_up.append(nn.ConvTranspose2d(ch, up_ch, 2, stride=2))
            self.dec_res.append(ResBlock2D(up_ch + skip_ch, up_ch, t_emb_dim))
            self.dec_attn.append(SpatialAttnBlock(up_ch) if level >= attn_start else nn.Identity())
            ch = up_ch

        # Output: Conv2d(hidden, 2, 1) per-frame
        self.out_proj = nn.Conv2d(hidden, 2, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor, cond: torch.Tensor,
                use_checkpoint: bool = False) -> torch.Tensor:
        # x:    [B, 2, T, H, W]
        # s:    [B,] ODE step
        # cond: [B, 14, H, W]
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        def maybe_ckpt(fn, *args):
            if use_checkpoint:
                return ckpt_fn(fn, *args, use_reentrant=False)
            return fn(*args)

        B, _, T, H, W = x.shape

        t_emb = self.t_embed(s)  # [B, t_emb_dim]
        # Expand t_emb across T frames: each frame in the batch*T fold gets the same s
        t_emb_t = t_emb.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        cond_t = cond.unsqueeze(2).expand(-1, -1, T, -1, -1)  # [B,14,T,H,W]
        inp = torch.cat([x, cond_t], dim=1)  # [B, 16, T, H, W]

        h = _per_frame(inp, self.lift)  # [B, hidden, T, H, W]

        skips = []
        for i in range(self.n_levels):
            # Checkpoint each encoder level to trade compute for activation memory.
            enc_i = i
            h = maybe_ckpt(lambda h, te, i=enc_i: _per_frame(h, lambda t: self.enc_res[i](t, te)), h, t_emb_t)
            h = _per_frame(h, self.enc_attn[i])
            skips.append(h)
            h = _per_frame(h, self.enc_down[i])

        h = maybe_ckpt(lambda h, te: _per_frame(h, lambda t: self.bottleneck_res(t, te)), h, t_emb_t)
        h = _per_frame(h, self.bottleneck_attn)
        h = self.bottleneck_temporal(h)

        for i in range(self.n_levels):
            level = self.n_levels - 1 - i
            h = _per_frame(h, self.dec_up[i])
            skip = skips[level]
            if h.shape[-2:] != skip.shape[-2:]:
                h = _per_frame(h, lambda t, sz=skip.shape[-2:]: F.interpolate(
                    t, size=sz, mode='bilinear', align_corners=False))
            h = torch.cat([h, skip], dim=1)
            dec_i = i
            h = maybe_ckpt(lambda h, te, i=dec_i: _per_frame(h, lambda t: self.dec_res[i](t, te)), h, t_emb_t)
            h = _per_frame(h, self.dec_attn[i])

        v = _per_frame(h, self.out_proj)  # [B, 2, T, H, W]
        return v


class FlowMatchingModel(nn.Module):
    """Full flow-matching model: geometry encoder + spatiotemporal U-Net."""

    def __init__(self, T_out: int = 20, obs_channels: int = 6, geo_channels: int = 8,
                 geo_in_channels: int = 2, hidden: int = 64, n_levels: int = 4,
                 t_emb_dim: int = 256, grid_size: int = 256):
        super().__init__()
        self.T_out = T_out
        self.obs_channels = obs_channels
        self.geo_channels = geo_channels

        self.geo_encoder = GeometryEncoder(in_channels=geo_in_channels, out_channels=geo_channels)

        in_channels = 2 + obs_channels + geo_channels
        self.unet = SpatiotemporalUNet(
            in_channels=in_channels, T_out=T_out, hidden=hidden,
            n_levels=n_levels, t_emb_dim=t_emb_dim, grid_size=grid_size)

    def forward(self, x_t: torch.Tensor, s: torch.Tensor,
                obs: torch.Tensor, mask: torch.Tensor,
                use_checkpoint: bool = False) -> torch.Tensor:
        """
        x_t:  [B, 2, T_out, H, W]
        s:    [B,]
        obs:  [B, obs_channels, H, W]
        mask: [B, 1, H, W]
        """
        geo = self.geo_encoder(mask)               # [B, geo_channels, H, W]
        cond = torch.cat([obs, geo], dim=1)         # [B, obs_channels+geo_channels, H, W]
        return self.unet(x_t, s, cond, use_checkpoint=use_checkpoint)

    @staticmethod
    def physics_prior(obs: torch.Tensor, solid_mask: torch.Tensor, T_out: int,
                       noise_std: float = 0.3) -> torch.Tensor:
        """
        Divergence-free, obstacle-aware prior field used as the flow-matching
        source distribution, replacing plain Gaussian noise. Estimates a
        single confidence-weighted ambient (u, v) from the sparse drone
        observations, broadcasts it to a uniform field, zeroes it inside
        obstacles, and Leray-projects it to divergence-free — giving the
        network a physically sane starting point instead of unstructured
        noise. Gaussian noise is added on top so the prior stays a proper
        distribution (needed for ensemble diversity at inference time).

        obs:        [B, obs_channels, H, W] — channel 1=obs_u, 2=obs_v, 3=confidence
        solid_mask: [H, W] bool, True = solid
        Returns:    [B, 2, T_out, H, W]
        """
        conf = obs[:, 3]  # [B, H, W]
        denom = conf.sum(dim=(-2, -1)).clamp_min(1e-3)
        u_amb = (obs[:, 1] * conf).sum(dim=(-2, -1)) / denom
        v_amb = (obs[:, 2] * conf).sum(dim=(-2, -1)) / denom

        fluid = (~solid_mask).float()[None]  # [1, H, W]
        u0 = u_amb[:, None, None] * fluid
        v0 = v_amb[:, None, None] * fluid
        u0, v0 = FlowMatchingModel.leray_project(u0, v0)
        u0 = u0 * fluid
        v0 = v0 * fluid

        field = torch.stack([u0, v0], dim=1).unsqueeze(2)              # [B,2,1,H,W]
        field = field.expand(-1, -1, T_out, -1, -1).clone()
        field = field + noise_std * torch.randn_like(field)
        return field

    @staticmethod
    def _log_spectral_loss(u_hat: torch.Tensor, v_hat: torch.Tensor,
                            u_gt: torch.Tensor, v_gt: torch.Tensor) -> torch.Tensor:
        """
        Log-spectral L2 distance between predicted and ground-truth 2D energy
        spectra (full FFT grid, not radially binned — our flow has a dominant
        wind direction, so isotropic radial averaging would wash out real
        anisotropic structure). u_hat, v_hat, u_gt, v_gt: [B, T, H, W].

        MSE-style losses are known to bias predictions toward over-smoothed,
        low-frequency output (safe averaging reduces pointwise error without
        reproducing high-frequency turbulent detail). This penalizes energy
        mismatch directly so under-texturing shows up as loss, not just as a
        qualitative "looks blurry" impression at inference time. log1p keeps
        the loss from being dominated by the largest scales, since energy
        density spans orders of magnitude across wavenumbers.

        Note: solid cells are zeroed in both signals (dataset + lambda_solid
        penalty), so the building-edge spectral artifact this creates is
        common to both sides and the network should learn it quickly from the
        (fixed, single) geometry — the residual loss signal beyond that is
        genuine flow-physics spectral mismatch.
        """
        U, V = torch.fft.rfft2(u_hat), torch.fft.rfft2(v_hat)
        Ug, Vg = torch.fft.rfft2(u_gt), torch.fft.rfft2(v_gt)
        E = U.abs() ** 2 + V.abs() ** 2
        Eg = Ug.abs() ** 2 + Vg.abs() ** 2
        return (torch.log1p(E) - torch.log1p(Eg)).pow(2).mean()

    def flow_match_loss(self, x_target: torch.Tensor, obs: torch.Tensor,
                         mask: torch.Tensor, solid_mask: torch.Tensor = None,
                         use_checkpoint: bool = False,
                         use_physics_prior: bool = True,
                         lambda_div: float = 0.1, lambda_solid: float = 0.1,
                         lambda_spectral: float = 0.0,
                         lambda_obs: float = 1.0) -> dict:
        """
        x_target: [B, 2, T_out, H, W] ground truth future sequence
        obs:      [B, obs_channels, H, W]  channels: 0=mask, 1=obs_u, 2=obs_v,
                  3=confidence, 4=x, 5=y
        mask:     [B, geo_in_channels, H, W]
        solid_mask: optional [H, W] bool, True = solid
        lambda_div, lambda_solid: soft physics penalties on x_hat1 (see below).
        lambda_spectral: log-spectral energy-matching penalty (default 0, off).
        lambda_obs: observation-consistency penalty — the single most important
            correctness constraint. Penalises x_hat1[:,:,0] for disagreeing
            with the drone observations at locations where confidence > 0.
            Without this, the model learns "plausible wind fields given obs as
            a hint" rather than "wind fields that are consistent with the
            specific measurements the drone actually took." A model without
            lambda_obs can predict low wind exactly where the drone just
            measured high wind; this term makes that a direct training error.
            obs_conf is already normalised to [0,1]; observations cover ~3.4%
            of the grid, so lambda_obs=1.0 gives the obs term roughly equal
            weight to the data term at convergence.

        Returns a dict with 'total' (use for backward()) plus component losses.
        """
        B = x_target.shape[0]
        s = torch.rand(B, device=x_target.device)
        if use_physics_prior and solid_mask is not None:
            x_noise = self.physics_prior(obs, solid_mask, self.T_out)
        else:
            x_noise = torch.randn_like(x_target)
        s_bc = s[:, None, None, None, None]
        x_t = (1 - s_bc) * x_noise + s_bc * x_target
        v_target = x_target - x_noise
        v_pred = self.forward(x_t, s, obs, mask, use_checkpoint=use_checkpoint)

        if solid_mask is None:
            data_loss = ((v_pred - v_target) ** 2).mean()
            zero = torch.zeros((), device=data_loss.device)
            return {'total': data_loss, 'data': data_loss.detach(),
                    'div': zero, 'solid': zero, 'spectral': zero, 'obs': zero}

        fluid = (~solid_mask).float()[None, None, None]  # [1,1,1,H,W]
        data_loss = ((v_pred - v_target) ** 2 * fluid).mean()
        total = data_loss
        div_loss = torch.zeros((), device=x_target.device)
        solid_loss = torch.zeros((), device=x_target.device)
        spectral_loss = torch.zeros((), device=x_target.device)
        obs_loss = torch.zeros((), device=x_target.device)

        if lambda_div > 0 or lambda_solid > 0 or lambda_spectral > 0 or lambda_obs > 0:
            x_hat1 = (x_t + (1 - s_bc) * v_pred).float()  # one-step clean-field estimate
            u_hat, v_hat = x_hat1[:, 0], x_hat1[:, 1]      # [B, T, H, W]

            if lambda_div > 0:
                div = torch.gradient(u_hat, dim=-1)[0] + torch.gradient(v_hat, dim=-2)[0]
                fluid_bc = (~solid_mask)[None, None].expand_as(div).float()
                div_loss = (div ** 2 * fluid_bc).sum() / fluid_bc.sum().clamp(min=1)
                total = total + lambda_div * div_loss

            if lambda_spectral > 0:
                spectral_loss = self._log_spectral_loss(u_hat, v_hat,
                                                          x_target[:, 0].float(),
                                                          x_target[:, 1].float())
                total = total + lambda_spectral * spectral_loss

            if lambda_solid > 0:
                solid_bc = solid_mask[None, None].expand_as(u_hat).float()
                speed2 = u_hat ** 2 + v_hat ** 2
                solid_loss = (speed2 * solid_bc).sum() / solid_bc.sum().clamp(min=1)
                total = total + lambda_solid * solid_loss

            if lambda_obs > 0:
                # obs channels: 1=obs_u, 2=obs_v, 3=confidence (normalised [0,1])
                obs_uv   = obs[:, 1:3].float()           # [B, 2, H, W]
                obs_conf = obs[:, 3:4].float()           # [B, 1, H, W]
                # Compare against x_hat1 at the first forecast frame (t=0),
                # which corresponds to the field immediately after the obs window.
                # Under quasi-steady flow this is the same as the observed field.
                pred_uv_t0 = x_hat1[:, :, 0]            # [B, 2, H, W]
                # Normalise by the total confidence weight rather than .mean()
                # (which averages over ALL cells including the ~96.6% with zero
                # confidence, diluting the obs signal ~29× into irrelevance).
                # This measures "confidence-weighted average error at observed
                # locations" — the same units as a pointwise velocity error.
                n_obs = obs_conf.sum().clamp(min=1e-6)
                obs_loss = (obs_conf * (pred_uv_t0 - obs_uv) ** 2).sum() / n_obs
                total = total + lambda_obs * obs_loss

        return {'total': total, 'data': data_loss.detach(),
                'div': div_loss.detach(), 'solid': solid_loss.detach(),
                'spectral': spectral_loss.detach(), 'obs': obs_loss.detach()}

    def sample(self, obs: torch.Tensor, mask: torch.Tensor, n_samples: int = 20,
               n_steps: int = 20, rho: float = 0.5, device: str = 'cuda',
               solid_mask: torch.Tensor = None,
               use_physics_prior: bool = True,
               chunk_size: int = 1) -> torch.Tensor:
        """
        DPS-guided ensemble generation.
        obs: [1, obs_channels, H, W] or [B, obs_channels, H, W]
        solid_mask: [H, W] bool, True = solid. Must be passed if the model
                    was trained with use_physics_prior=True (the default) —
                    the sampling source distribution has to match training.
        chunk_size: number of samples to process simultaneously (default 1).
            At 512×512 with T_out=10, _per_frame reshapes [B,C,T,H,W] to
            [B*T, C, H, W] for convolutions — B=20*T=10 gives batch=200
            inside every conv layer, requiring ~26 GB for a single fp32
            feature map, which OOMs on 32 GB. With chunk_size=1, effective
            inner batch is T_out=10, using ~1.3 GB for the same layer.
            Increase only if you have confirmed headroom.
        Returns: [n_samples, 2, T_out, H, W]
        """
        T = self.T_out
        H, W = mask.shape[-2:]
        device_t = torch.device(device) if isinstance(device, str) else device

        obs_1 = obs.to(device_t)       # [1, obs_ch, H, W]
        mask_1 = mask.to(device_t)     # [1, geo_ch, H, W]
        solid_mask_d = solid_mask.to(device_t) if solid_mask is not None else None

        # obs for a single sample
        obs_uv_1 = obs_1[:, 1:3]   # [1, 2, H, W]
        conf_1   = obs_1[:, 3:4]   # [1, 1, H, W]

        ds = 1.0 / n_steps
        all_samples = []

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            B = end - start

            obs_b  = obs_1.expand(B, -1, -1, -1)
            mask_b = mask_1.expand(B, -1, -1, -1)
            obs_uv = obs_uv_1.expand(B, -1, -1, -1)
            conf   = conf_1.expand(B, -1, -1, -1)

            if use_physics_prior and solid_mask_d is not None:
                x = self.physics_prior(obs_b, solid_mask_d, T)
            else:
                x = torch.randn(B, 2, T, H, W, device=device_t)

            amp_enabled = device_t.type == 'cuda'
            for step in range(n_steps):
                s_val = step / n_steps
                s_t   = torch.full((B,), s_val, device=device_t)

                x_req = x.detach().requires_grad_(True)
                # AMP forward pass: ~2× faster via fp16 tensor cores.
                # Gradient computation stays in fp32 — autograd returns
                # gradients in the leaf tensor's dtype (fp32) regardless
                # of the autocast context used during the forward pass.
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    v_hat = self.forward(x_req, s_t, obs_b, mask_b)
                v_hat = v_hat.float()   # ensure fp32 before obs_loss / update
                x_hat1   = x_req + (1 - s_val) * v_hat
                pred_uv  = x_hat1[:, :, 0]
                obs_loss = (conf * (pred_uv - obs_uv) ** 2).sum()
                g = torch.autograd.grad(obs_loss, x_req)[0]

                with torch.no_grad():
                    x = x + ds * v_hat - rho * g

            all_samples.append(x.detach())

        return torch.cat(all_samples, dim=0)  # [n_samples, 2, T_out, H, W]

    @staticmethod
    def leray_project(u: torch.Tensor, v: torch.Tensor):
        """
        Project (u, v) to a divergence-free field via 2D FFT (Leray projection).
        u, v: [..., H, W] (e.g. [B, T, H, W] or [B, H, W])
        """
        shape = u.shape
        H, W = shape[-2], shape[-1]
        U = torch.fft.rfft2(u)
        V = torch.fft.rfft2(v)

        kx = torch.fft.rfftfreq(W).to(u.device) * W
        ky = torch.fft.fftfreq(H).to(u.device) * H

        n_lead = len(shape) - 2
        KY_r = ky.reshape(*([1] * n_lead), H, 1).expand(*([1] * n_lead), H, W // 2 + 1)
        KX_r = kx.reshape(*([1] * n_lead), 1, W // 2 + 1).expand(*([1] * n_lead), H, W // 2 + 1)

        K2 = KX_r ** 2 + KY_r ** 2
        K2 = K2.clone()
        K2[..., 0, 0] = 1.0

        div = KX_r * U + KY_r * V
        U_proj = U - KX_r * div / K2
        V_proj = V - KY_r * div / K2

        return torch.fft.irfft2(U_proj, s=(H, W)), torch.fft.irfft2(V_proj, s=(H, W))
