# oscillator-learning

Two experiments built on one idea: **use coupled [Kuramoto](https://en.wikipedia.org/wiki/Kuramoto_model)
oscillators as the computational substrate** of a neural network, in place of the
usual machinery. Random/initial state flows through coupled-oscillator dynamics
into an attractor, which is then read out — the generative cousin of
Hopfield/associative memory.

```
Hopfield:    corrupted input  -> energy descent       -> remembered pattern
this repo:   initial state    -> oscillator dynamics  -> useful output
```

| project | swaps out | for | result |
|---|---|---|---|
| **Un-0 / MNIST** (`un0/`) | diffusion / GAN | coupled oscillators → conv decoder | class-conditional handwritten digits |
| **KuramotoGPT / Shakespeare** (`nanochat_kuramoto/`) | self-attention | a Kuramoto oscillator token-mixer | char-level Shakespeare |

Both train in ~20–25 min on an Apple-Silicon Mac (MPS); both reach real outputs.

---

## Setup

```bash
git clone https://github.com/allthingssecurity/oscillator-learning
cd oscillator-learning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch + torchvision
```

Works on Apple MPS, CUDA, or CPU (auto-detected).

## Quick test (no training — uses the committed checkpoints)

Both trained checkpoints are in the repo, so you can generate immediately:

```bash
# 1) MNIST digits -> samples/grid_test.png
python -m un0.sample --ckpt checkpoints/un0_mnist.pt --out samples/grid_test.png

# 2) Shakespeare text -> stdout
python -m nanochat_kuramoto.sample --prompt $'KING RICHARD:\n'
```

Pre-generated samples are already in `samples/` (digit grids) and
`nanochat_kuramoto/samples/` (text), so you can eyeball results without running
anything.

---

## 1. Un-0 on MNIST — image generation with coupled oscillators

A compact, laptop-runnable re-implementation of
[**Un-0**](https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/)
([code](https://github.com/unconv-ai/Un-0)). Un-0 generates images by letting a
bank of coupled Kuramoto oscillators evolve from random phases into a
class-conditioned attractor, then decoding the settled phases into pixels.

```
 class id c ─┬─► learned cond-bank init phases φ₀[c]   (the class signal)
            ├─► per-class drive matrix K_drive[c]
            └─► per-class natural-freq bias ω_class[c]
                       │
 main bank θ ~ U[-π,π)   ◄── the "noise" (within-class variety)
                       ▼
   ┌─────────────────────────────────────┐
   │  coupled Kuramoto dynamics (RK4)     │   θ̇ᵢ = ωᵢ + ω_class[c]ᵢ
   │  main bank θ  ←drive←  cond bank φ    │          + Σⱼ Kᵢⱼ sin(θⱼ-θᵢ)
   └─────────────────────────────────────┘          + Σₖ Kd^c sin(φₖ-θᵢ)
                       ▼
        sin/cos readout (relative to mean phase)
                       ▼
        resize-conv decoder  (7×7 → 14 → 28)   →   28×28 image
```

**Conditioning, the hard part.** The blog routes class info only through the
drive matrix `K_drive[c]`. At MNIST scale that signal was too weak to survive a
dozen integration steps, so every class decoded to the same blurry mean digit.
Two faithful, Kuramoto-native additions fix it: a learned **per-class initial
phase** for the conditioning bank (`φ₀[c]`, the dominant class signal) and a
learned **per-class natural-frequency bias** (`ω_class[c]`). With those, classes
separate into crisp digits within a few hundred steps.

- **`un0/model.py`** — `ConditionalImplicitKuramotoGenerator`: oscillator
  dynamics, sin/cos readout, `ResizeConvDecoder`.
- **`un0/losses.py`** — the **conditional drifting loss**: each generated sample
  is a particle regressed toward `gen + drift`, where `drift` points to the
  soft-nearest *real* samples of its class and away from negatives. A
  distribution-matching / particle-flow objective — a *direction toward the data
  manifold*, which is what makes the dynamics behave like an attractor.
- **`un0/train_mnist.py`** — training loop, sample grids, checkpointing.
- **`un0/sample.py`** — render a labeled grid from a checkpoint.

**Train it:**

```bash
python -m un0.train_mnist --steps 6000 --batch 256
```

Downloads MNIST automatically. Outputs 10×10 class-conditional grids to
`samples/` (one row per digit) and `checkpoints/un0_mnist.pt`. Flags:
`--n-main`, `--n-cond`, `--n-steps`, `--solver {euler,rk4}`, `--tau`, `--lr`.

| | Reference Un-0 | This version |
|---|---|---|
| Dynamics | Kuramoto, main + per-class cond bank | **same** + learned `φ₀`/`ω` conditioning |
| Readout / Decoder | sin/cos + resize-conv | **same** (7→14→28) |
| Loss | drift loss over **DINOv2 + pixel** | drift over **pixels only** (paper's `pixel_weight` path) |
| Scale | 1k–16k osc, CIFAR/ImageNet, B200-hrs | 256 osc, MNIST, ~25 min on a Mac |

---

## 2. KuramotoGPT — nanochat with an oscillator mixer

Karpathy's [nanochat](https://github.com/karpathy/nanochat) GPT architecture with
its **self-attention token-mixer replaced by coupled Kuramoto oscillators**,
trained char-level on tinyshakespeare. Each position holds a bank of oscillators
that synchronize to a learned **causal, distance-weighted coupling kernel** over
earlier positions — synchronization carries information forward, the role
attention plays in a transformer (causality verified numerically).

Final: train loss **1.13**, val **1.55** in ~22 min; learns real Shakespeare
character names, play formatting, and plausible verse. Full details, the
mechanism diagram, and the "global-mean-field stalled / windowed-kernel worked"
debugging story are in **[`nanochat_kuramoto/README.md`](nanochat_kuramoto/README.md)**.

**Train it:**

```bash
# one-time: fetch the corpus (gitignored)
mkdir -p nanochat_kuramoto/data
curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
     -o nanochat_kuramoto/data/input.txt

python -m nanochat_kuramoto.train --steps 3000 --batch 48 --block 192 --n-osc 192 --osc-steps 3
```

> Note: this ports nanochat's *architecture* and trains a small model locally. It
> is **not** the real nanochat pipeline (FineWeb + 8×H100 + Flash-Attention-3 +
> Muon, none of which run on a Mac). The upstream `nanochat/` clone is gitignored.

---

## Repo layout

```
un0/                    Un-0 MNIST: model.py, losses.py, train_mnist.py, sample.py
samples/                MNIST sample grids (committed)
checkpoints/            un0_mnist.pt (committed)
nanochat_kuramoto/      KuramotoGPT: model.py, train.py, sample.py, README, ckpt.pt, samples/
requirements.txt
```
