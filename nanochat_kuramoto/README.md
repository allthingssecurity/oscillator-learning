# KuramotoGPT — nanochat with a coupled-oscillator mixer

Karpathy's [nanochat](https://github.com/karpathy/nanochat) GPT architecture with
its **self-attention token-mixer replaced by coupled Kuramoto oscillators**,
trained char-level on tinyshakespeare. Same oscillator-dynamics-as-computation
idea as the [Un-0 MNIST experiment](../README.md) one level up, applied to a
language model.

## What's kept from nanochat vs. what's swapped

nanochat's real model (`nanochat/nanochat/gpt.py`) is Hopper-specific
(Flash-Attention-3, Muon optimizer, value embeddings) and trains on 8×H100. None
of that runs on a Mac, so this is a faithful *architecture* port, not the repo's
training stack:

| Kept (nanochat flavor) | Swapped |
|---|---|
| RMSNorm (no params), pre-block + post-trunk | **attention → `KuramotoMixer`** |
| relu² MLP, 4× expansion, no bias | Muon → plain AdamW |
| untied `wte`/`lm_head`, norm after embed | FA3 → pure-torch (MPS) |
| residual blocks, softcapped logits, CE loss | dropped value-embeds / GQA / sliding-window |

## The mechanism — `KuramotoMixer`

Each position carries a bank of `N` phase oscillators. They are integrated for a
few Euler steps while coupling to a **causal, distance-weighted window** of the
*same* oscillator channel at preceding positions — a learned coupling kernel
`K(t−τ)` (a softmax-normalised causal depthwise conv), i.e. a local order
parameter:

```
θ_init = Wphase(x);   ω = c·tanh(Wω(x)) + ω_pos;   K = softplus(Wk(x))
repeat S:
    field = causalConv_K( cos θ ), causalConv_K( sin θ )     # who to sync with
    θ  +=  dt · ( ω + K · |field|·sin(∠field − θ) )
y = Wout( [cos θ, sin θ] )
```

The causal kernel keeps it autoregressive (position `t` only sees `≤ t`, verified
numerically) while synchronization carries earlier-token information forward —
the role attention plays in a transformer.

**Design note.** The first version used a *global* causal mean-field (uniform
prefix average of phases). It stalled at bigram-level loss (~2.45) — a global
average has no recency or selectivity. Switching to the learned windowed
coupling kernel broke straight through to <2.0 and into real Shakespearean
structure. (Both versions are in the git history of `model.py`.)

## Run

```bash
python -m nanochat_kuramoto.train --steps 3000 --batch 48 --block 192 --n-osc 192 --osc-steps 3
```

~18 min on an M-series Mac (MPS). Generates samples to `samples/` during
training, final 1000-char sample to `samples/final.txt`, checkpoint to `ckpt.pt`.
This is a ~3M-param char model on 1MB of text — it learns Shakespearean
*texture* (names, dialogue format, plausible words/rhythm), not facts.
