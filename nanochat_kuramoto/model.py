"""KuramotoGPT -- nanochat's architecture with the attention mixer replaced by
coupled Kuramoto oscillators.

We keep Karpathy nanochat's flavour (see nanochat/nanochat/gpt.py):
  * RMSNorm with no learnable params, applied pre-block and post-trunk
  * relu^2 MLP, 4x expansion, no bias
  * untied token embedding / lm_head, norm after token embedding
  * residual blocks  x = x + mix(norm(x));  x = x + mlp(norm(x))
  * softcapped logits + cross-entropy

The one swap is the token mixer.  Instead of causal self-attention, each
position carries a bank of N phase oscillators that are pulled toward the
*causal running mean-field* of all earlier positions:

    theta_init = Wphase(x)                          # content sets initial phase
    omega      = c * tanh(Womega(x)) + omega_pos    # natural frequency (+ position)
    K          = softplus(Wk(x))                    # per-oscillator coupling gain

    repeat S times (Euler):
        (r_t, psi_t) = resultant of mean over positions <= t   # causal, via cumsum
        theta <- theta + dt * ( omega + K * r_t * sin(psi_t - theta) )

    y = Wout( [cos theta, sin theta] )

The causal prefix mean makes position t depend only on positions <= t, so the
model stays autoregressive, while synchronization carries earlier-token
information forward -- the role attention plays in a normal transformer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


@dataclass
class KGPTConfig:
    vocab_size: int = 96
    sequence_len: int = 256
    n_layer: int = 4
    n_embd: int = 256
    n_osc: int = 256          # oscillators per position per layer
    osc_steps: int = 4        # Kuramoto integration steps
    osc_dt: float = 0.5
    omega_scale: float = 0.3  # cap on content-driven natural frequency
    coupling_window: int = 48 # causal coupling kernel length K(t-tau)


class KuramotoMixer(nn.Module):
    """Causal locally-coupled Kuramoto token mixer (drop-in for attention).

    Each oscillator channel couples to a learned-weighted window of the *same*
    channel at preceding positions -- a distance-dependent coupling kernel,
    K(t-tau).  The coupling field is a causal depthwise convolution of the
    (cos, sin) phase signals; the kernel is softmax-normalised over the window
    so it acts as a local order parameter (which past positions to synchronise
    with).  This gives recency and position-selectivity, which the plain global
    mean-field lacked.
    """

    def __init__(self, cfg: KGPTConfig, layer_idx: int):
        super().__init__()
        n, c = cfg.n_osc, cfg.n_embd
        self.cfg = cfg
        self.win = cfg.coupling_window
        self.to_phase = nn.Linear(c, n, bias=False)
        self.to_omega = nn.Linear(c, n, bias=False)
        self.to_k = nn.Linear(c, n, bias=False)
        self.proj = nn.Linear(2 * n, c, bias=False)
        # Per-channel causal coupling kernel K(t-tau), softmax-normalised over
        # the window in forward.  Init biased toward recent positions.
        kinit = torch.linspace(-2.0, 2.0, self.win).view(1, 1, self.win).repeat(n, 1, 1)
        self.coupling = nn.Parameter(kinit + 0.01 * torch.randn(n, 1, self.win))
        # Fixed per-oscillator positional frequencies (rotary-in-spirit).
        freqs = torch.logspace(math.log10(1e-3), math.log10(1.0), n)
        self.register_buffer("omega_pos", freqs * (math.pi / cfg.sequence_len), persistent=True)

    def _field(self, signal):
        """Causal depthwise conv of (B,T,N) signal with the coupling kernel."""
        B, T, N = signal.shape
        ker = torch.softmax(self.coupling, dim=-1)            # (N,1,win), sums to 1
        s = signal.transpose(1, 2)                            # (B,N,T)
        s = F.pad(s, (self.win - 1, 0))                       # left pad -> causal
        out = F.conv1d(s, ker, groups=N)                      # (B,N,T)
        return out.transpose(1, 2)                            # (B,T,N)

    def forward(self, x):
        cfg = self.cfg
        theta = self.to_phase(x)                                  # (B,T,N) initial phases
        omega = cfg.omega_scale * torch.tanh(self.to_omega(x)) + self.omega_pos
        k = F.softplus(self.to_k(x))                              # (B,T,N) >= 0 coupling gain

        dt = cfg.osc_dt
        for _ in range(cfg.osc_steps):
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            field_cos, field_sin = self._field(cos_t), self._field(sin_t)
            # |field| * sin(angle(field) - theta), the local-order-parameter pull.
            pull = field_sin * cos_t - field_cos * sin_t
            theta = theta + dt * (omega + k * pull)

        feats = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)  # (B,T,2N)
        return self.proj(feats)


class MLP(nn.Module):
    def __init__(self, cfg: KGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: KGPTConfig, layer_idx: int):
        super().__init__()
        self.mix = KuramotoMixer(cfg, layer_idx)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.mix(norm(x))
        x = x + self.mlp(norm(x))
        return x


class KuramotoGPT(nn.Module):
    def __init__(self, cfg: KGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.zeros_(self.lm_head.weight)
        # zero the mixer/MLP output projections so blocks start as identity
        for b in self.blocks:
            nn.init.zeros_(b.mix.proj.weight)
            nn.init.zeros_(b.mlp.c_proj.weight)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = norm(self.wte(idx))
        for b in self.blocks:
            x = b(x)
        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)          # softcap, nanochat-style
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.sequence_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
