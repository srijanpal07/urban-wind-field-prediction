"""
U-FNO: U-shaped Fourier Neural Operator
For wind field reconstruction and forecasting.

Input channels (per grid cell):
  0: geometry mask (static, 1=building)
  1: observation u (Gaussian-splatted drone samples)
  2: observation v
  3: observation confidence mask
  4: x coordinate (normalized 0-1)
  5: y coordinate (normalized 0-1)

Output:
  0: predicted u(x,y)
  1: predicted v(x,y)
  2: uncertainty sigma_u
  3: uncertainty sigma_v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    """
    Fourier layer: multiply in frequency domain then IFFT back.
    Learns global flow patterns efficiently.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Fourier modes in x
        self.modes2 = modes2  # Fourier modes in y

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, H, W//2+1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2]  = self.compl_mul2d(x_ft[:, :, :m1, :m2],  self.weights1)
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(H, W))


def _group_norm(channels):
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return nn.GroupNorm(g, channels)


class FNOBlock(nn.Module):
    """Single FNO block: spectral conv + pointwise conv + activation"""
    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = _group_norm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.pointwise(x)))


class UFNO(nn.Module):
    """
    U-shaped FNO: FNO encoder-decoder with skip connections.
    Combines FNO's global frequency reasoning with U-Net's local detail.
    """
    def __init__(self, in_channels=6, out_channels=4,
                 hidden=32, modes=16, depth=3):
        super().__init__()
        self.depth = depth

        # Input projection
        self.lift = nn.Conv2d(in_channels, hidden, 1)

        # Encoder: FNO blocks + downsampling
        self.enc_blocks = nn.ModuleList()
        self.enc_down    = nn.ModuleList()
        ch = hidden
        self.enc_channels = [ch]
        for i in range(depth):
            self.enc_blocks.append(FNOBlock(ch, modes//(2**i), modes//(2**i)))
            out_ch = ch * 2
            self.enc_down.append(nn.Conv2d(ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch
            self.enc_channels.append(ch)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            FNOBlock(ch, max(4, modes//(2**depth)), max(4, modes//(2**depth))),
            FNOBlock(ch, max(4, modes//(2**depth)), max(4, modes//(2**depth)))
        )

        # Decoder: upsampling + skip + FNO blocks
        self.dec_up     = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth):
            skip_ch = self.enc_channels[depth - 1 - i]
            self.dec_up.append(nn.ConvTranspose2d(ch, ch//2, 2, stride=2))
            ch = ch//2
            self.dec_blocks.append(FNOBlock(ch + skip_ch,
                                            modes//(2**(depth-1-i)),
                                            modes//(2**(depth-1-i))))
            ch = ch + skip_ch

        # Project to final FNO channel count (clean)
        self.dec_proj = nn.ModuleList([
            nn.Conv2d(ch_in, hidden, 1)
            for ch_in in self._get_dec_channels(hidden, depth)
        ])

        # Output projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1)
        )

    def _get_dec_channels(self, hidden, depth):
        """Compute decoder channel sizes after skip concatenation."""
        ch = hidden
        enc_chs = [ch]
        for i in range(depth):
            ch *= 2
            enc_chs.append(ch)
        dec_chs = []
        for i in range(depth):
            skip = enc_chs[depth - 1 - i]
            up_ch = ch // 2
            dec_chs.append(up_ch + skip)
            ch = up_ch + skip
        return dec_chs

    def forward(self, x):
        # Lift input
        x = self.lift(x)

        # Encoder
        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.enc_down[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.depth):
            x = self.dec_up[i](x)
            skip = skips[self.depth - 1 - i]
            # Align sizes if needed
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)

        # Project to hidden then output
        x = nn.functional.adaptive_avg_pool2d(x, x.shape[2:])  # no-op, keeps shape
        # Final conv to reduce channels
        x = nn.Conv2d(x.shape[1], 32, 1).to(x.device)(x)
        x = F.gelu(x)
        x = nn.Conv2d(32, 4, 1).to(x.device)(x)
        return x


class WindFNO(nn.Module):
    """
    Clean, stable U-FNO for wind field prediction.
    Simpler than full UFNO above but effective.
    """
    def __init__(self, in_channels=6, out_channels=4,
                 hidden=48, modes=20, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden, 1)

        self.fno_layers = nn.ModuleList([
            FNOBlock(hidden, modes, modes) for _ in range(n_layers)
        ])

        # U-Net style skip at midpoint
        self.mid = n_layers // 2

        out_proj = nn.Conv2d(hidden // 2, out_channels, 1)
        nn.init.zeros_(out_proj.weight)
        nn.init.zeros_(out_proj.bias)
        self.project = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1),  # *2 for skip
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GELU(),
            out_proj
        )

    def forward(self, x):
        x = self.lift(x)
        skip = None
        for i, layer in enumerate(self.fno_layers):
            x = layer(x)
            if i == self.mid - 1:
                skip = x.clone()

        x = torch.cat([x, skip], dim=1)
        out = self.project(x)

        # Split outputs: mean u/v + log-variance for uncertainty
        u_pred     = out[:, 0:1]
        v_pred     = out[:, 1:2]
        log_var_u  = out[:, 2:3]
        log_var_v  = out[:, 3:4]

        sigma_u = torch.exp(0.5 * log_var_u.clamp(-6, 6))
        sigma_v = torch.exp(0.5 * log_var_v.clamp(-6, 6))

        return u_pred, v_pred, sigma_u, sigma_v


def prepare_input(obs_u: np.ndarray, obs_v: np.ndarray,
                  obs_mask: np.ndarray, geom_mask: np.ndarray,
                  device='cuda') -> torch.Tensor:
    """
    Assemble model input tensor from observations and geometry.
    All inputs: [H, W] numpy arrays.
    Returns: [1, 6, H, W] tensor
    """
    H, W = geom_mask.shape
    ys = np.linspace(0, 1, H)
    xs = np.linspace(0, 1, W)
    xg, yg = np.meshgrid(xs, ys)

    channels = np.stack([
        geom_mask.astype(np.float32),
        obs_u.astype(np.float32),
        obs_v.astype(np.float32),
        obs_mask.astype(np.float32),
        xg.astype(np.float32),
        yg.astype(np.float32)
    ], axis=0)  # [6, H, W]

    return torch.tensor(channels[None], dtype=torch.float32).to(device)


def nll_loss(u_pred, v_pred, sigma_u, sigma_v, u_true, v_true, solid_mask):
    """
    Negative log-likelihood loss (Gaussian) + physics regularization.
    Only computed on fluid cells (not inside buildings).
    """
    fluid_t = (~solid_mask).float().to(u_pred.device).unsqueeze(0).unsqueeze(0)

    # NLL loss: -log p(y | mu, sigma)
    nll_u = (torch.log(sigma_u + 1e-6) +
             0.5 * ((u_true - u_pred) / (sigma_u + 1e-6))**2) * fluid_t
    nll_v = (torch.log(sigma_v + 1e-6) +
             0.5 * ((v_true - v_pred) / (sigma_v + 1e-6))**2) * fluid_t

    loss = (nll_u + nll_v).mean()

    # Divergence-free regularization (incompressible flow: du/dx + dv/dy = 0)
    du_dx = u_pred[:, :, :, 1:] - u_pred[:, :, :, :-1]
    dv_dy = v_pred[:, :, 1:, :] - v_pred[:, :, :-1, :]
    min_size = min(du_dx.shape[2], dv_dy.shape[2]), min(du_dx.shape[3], dv_dy.shape[3])
    div = du_dx[:, :, :min_size[0], :min_size[1]] + dv_dy[:, :, :min_size[0], :min_size[1]]
    div_loss = (div**2).mean()

    return loss + 0.01 * div_loss
