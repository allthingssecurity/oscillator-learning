"""Conditional Implicit Kuramoto Generator (Un-0), MNIST edition.

Faithful re-implementation of the core ideas from
https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/
and https://github.com/unconv-ai/Un-0 , scaled down to 28x28 grayscale MNIST.

Pipeline (matching the paper):
    random phases  ->  coupled Kuramoto dynamics (class conditioned)
                   ->  sin/cos readout  ->  resize-conv decoder  ->  image

The only deliberate simplifications vs. the reference repo:
  * grayscale, 28x28, smaller oscillator counts (runs on a laptop / MPS);
  * a small linear projection sits between the readout and the decoder grid so
    the oscillator count and the decoder geometry can be chosen independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Kuramoto dynamics
# --------------------------------------------------------------------------- #
class ConditionalKuramotoDynamics(nn.Module):
    r"""Coupled-oscillator velocity field with one-way class conditioning.

    Main oscillators \theta and a smaller conditioning bank \phi each follow
    Kuramoto dynamics; the conditioning bank drives the main bank through a
    class-specific coupling matrix K_drive[c]:

        d\theta_i/dt = \omega_i
                     + \sum_j K_ij      sin(\theta_j - \theta_i)
                     + \sum_k Kd^c_ik   sin(\phi_k  - \theta_i)
        d\phi_k/dt   = \omega^c_k
                     + \sum_l Kc_kl     sin(\phi_l  - \phi_k)

    The sin(a - b) coupling is expanded with the angle-subtraction identity so
    it can be evaluated with two matmuls instead of an N x N broadcast:

        \sum_j K_ij sin(\theta_j - \theta_i)
            = cos(\theta_i) (sin\theta @ K^T)_i - sin(\theta_i) (cos\theta @ K^T)_i
    """

    def __init__(self, n_main: int, n_cond: int, n_classes: int, k_scale: float | None = None):
        super().__init__()
        self.n_main = n_main
        self.n_cond = n_cond
        self.n_classes = n_classes
        # 1/sqrt(N) keeps the summed coupling O(1) regardless of population size.
        self.k_scale = k_scale if k_scale is not None else 1.0 / math.sqrt(n_main)
        self.kc_scale = 1.0 / math.sqrt(n_cond)
        # Stronger conditioning->main drive so class identity actually imprints
        # on the phases within a few integration steps.
        self.kd_scale = 2.0 / math.sqrt(n_cond)

        # Natural frequencies.
        self.omega = nn.Parameter(0.1 * torch.randn(n_main))
        self.omega_cond = nn.Parameter(0.1 * torch.randn(n_cond))
        # Class-conditioned natural-frequency bias on the main oscillators: the
        # most direct Kuramoto-faithful way to make each class spin the
        # population differently (the K_drive pathway alone was too weak to
        # break the symmetry, so every class decoded to the same blurry digit).
        self.omega_class = nn.Parameter(0.1 * torch.randn(n_classes, n_main))
        # Learned per-class *initial phases* for the conditioning bank. This is
        # the strong, low-attenuation class signal: the conditioning oscillators
        # start in a class-specific configuration and imprint it on the main
        # bank through K_drive over the integration. (The main bank's random
        # init remains the "noise" that gives within-class variety.)
        self.phi0_class = nn.Parameter(math.pi * torch.randn(n_classes, n_cond))

        # Coupling matrices (diagonal zeroed in forward to forbid self-coupling).
        self.K = nn.Parameter(torch.randn(n_main, n_main) / math.sqrt(n_main))
        self.K_cond = nn.Parameter(torch.randn(n_cond, n_cond) / math.sqrt(n_cond))
        # One conditioning->main drive matrix per class.
        self.K_drive = nn.Parameter(torch.randn(n_classes, n_main, n_cond) / math.sqrt(n_cond))

    def velocity(self, theta, phi, k_drive, omega_bias):
        """Return (dtheta, dphi).

        k_drive    : per-sample conditioning->main drive matrix (B, n_main, n_cond)
        omega_bias : per-sample class natural-frequency bias    (B, n_main)
        """
        K = self.K - torch.diag_embed(torch.diagonal(self.K))
        Kc = self.K_cond - torch.diag_embed(torch.diagonal(self.K_cond))

        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        sin_p, cos_p = torch.sin(phi), torch.cos(phi)

        # Main <- main coupling.
        coup_main = cos_t * (sin_t @ K.T) - sin_t * (cos_t @ K.T)
        # Main <- conditioning drive (batched matvec per sample).
        drive_sin = torch.bmm(k_drive, sin_p.unsqueeze(-1)).squeeze(-1)
        drive_cos = torch.bmm(k_drive, cos_p.unsqueeze(-1)).squeeze(-1)
        coup_drive = cos_t * drive_sin - sin_t * drive_cos

        dtheta = self.omega + omega_bias + self.k_scale * coup_main + self.kd_scale * coup_drive

        # Conditioning <- conditioning coupling.
        coup_cond = cos_p * (sin_p @ Kc.T) - sin_p * (cos_p @ Kc.T)
        dphi = self.omega_cond + self.kc_scale * coup_cond
        return dtheta, dphi


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class ResizeConvBlock(nn.Module):
    """Nearest-neighbour 2x upsample followed by two 3x3 convs (LeakyReLU)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class ResizeConvDecoder(nn.Module):
    """Readout features -> (C, h, w) grid -> upsample cascade -> image."""

    def __init__(self, in_features: int, base_ch: int, base_hw: int,
                 num_upsamples: int, out_ch: int):
        super().__init__()
        self.base_ch = base_ch
        self.base_hw = base_hw
        self.proj = nn.Linear(in_features, base_ch * base_hw * base_hw)

        chs = [base_ch // (2 ** i) for i in range(num_upsamples + 1)]
        chs = [max(c, 16) for c in chs]
        blocks = [ResizeConvBlock(chs[i], chs[i + 1]) for i in range(num_upsamples)]
        self.blocks = nn.ModuleList(blocks)
        self.to_output = nn.Conv2d(chs[-1], out_ch, 3, padding=1)

    def forward(self, feats):
        b = feats.shape[0]
        x = self.proj(feats).view(b, self.base_ch, self.base_hw, self.base_hw)
        for blk in self.blocks:
            x = blk(x)
        x = torch.tanh(self.to_output(x))  # pixels in [-1, 1]
        return x


# --------------------------------------------------------------------------- #
# Full generator
# --------------------------------------------------------------------------- #
@dataclass
class Un0Config:
    n_main: int = 256
    n_cond: int = 64
    n_classes: int = 10
    n_steps: int = 12
    dt: float = 0.5
    solver: str = "rk4"          # "euler" or "rk4"
    img_size: int = 28
    img_ch: int = 1
    base_ch: int = 64
    base_hw: int = 7
    num_upsamples: int = 2       # 7 -> 14 -> 28
    class_dropout: float = 0.0


class ConditionalImplicitKuramotoGenerator(nn.Module):
    def __init__(self, cfg: Un0Config):
        super().__init__()
        self.cfg = cfg
        self.dynamics = ConditionalKuramotoDynamics(cfg.n_main, cfg.n_cond, cfg.n_classes)
        self.decoder = ResizeConvDecoder(
            in_features=2 * cfg.n_main,
            base_ch=cfg.base_ch,
            base_hw=cfg.base_hw,
            num_upsamples=cfg.num_upsamples,
            out_ch=cfg.img_ch,
        )

    # -- integration ------------------------------------------------------- #
    def _integrate(self, theta, phi, k_drive, omega_bias):
        dt = self.cfg.dt
        ob = omega_bias
        for _ in range(self.cfg.n_steps):
            if self.cfg.solver == "euler":
                dth, dph = self.dynamics.velocity(theta, phi, k_drive, ob)
                theta = theta + dt * dth
                phi = phi + dt * dph
            else:  # classic RK4
                k1t, k1p = self.dynamics.velocity(theta, phi, k_drive, ob)
                k2t, k2p = self.dynamics.velocity(theta + 0.5 * dt * k1t, phi + 0.5 * dt * k1p, k_drive, ob)
                k3t, k3p = self.dynamics.velocity(theta + 0.5 * dt * k2t, phi + 0.5 * dt * k2p, k_drive, ob)
                k4t, k4p = self.dynamics.velocity(theta + dt * k3t, phi + dt * k3p, k_drive, ob)
                theta = theta + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
                phi = phi + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        return theta, phi

    def forward(self, class_id: torch.Tensor, generator: torch.Generator | None = None):
        cfg = self.cfg
        b = class_id.shape[0]
        device = class_id.device

        # Main bank: random initial phases in [-pi, pi) -- the analogue of
        # diffusion noise, giving within-class sample variety.
        theta = torch.empty(b, cfg.n_main, device=device)
        theta.uniform_(-math.pi, math.pi, generator=generator)

        # Per-sample class signals.
        cls = class_id.clone()
        k_drive = self.dynamics.K_drive[cls]            # (B, n_main, n_cond)
        omega_bias = self.dynamics.omega_class[cls]     # (B, n_main)
        # Conditioning bank starts at the learned class-specific phases, plus a
        # little jitter so it is not perfectly identical across samples.
        jitter = 0.1 * torch.randn(b, cfg.n_cond, device=device, generator=generator)
        phi = self.dynamics.phi0_class[cls] + jitter    # (B, n_cond)

        if self.training and cfg.class_dropout > 0:
            mask = (torch.rand(b, device=device) < cfg.class_dropout)
            k_drive = k_drive.clone(); k_drive[mask] = 0.0
            omega_bias = omega_bias.clone(); omega_bias[mask] = 0.0

        theta, phi = self._integrate(theta, phi, k_drive, omega_bias)

        # sin/cos readout relative to the batch-mean phase (reference phase).
        theta_ref = theta.mean(dim=1, keepdim=True)
        d = theta - theta_ref
        feats = torch.cat([torch.cos(d), torch.sin(d)], dim=1)  # (B, 2*n_main)

        img = self.decoder(feats)
        return img

    @torch.no_grad()
    def sample(self, class_id, generator=None):
        was_training = self.training
        self.eval()
        img = self.forward(class_id, generator=generator)
        if was_training:
            self.train()
        return img

    def num_params(self):
        dec = sum(p.numel() for p in self.decoder.parameters())
        dyn = sum(p.numel() for p in self.dynamics.parameters())
        return {"dynamics": dyn, "decoder": dec, "total": dyn + dec}
