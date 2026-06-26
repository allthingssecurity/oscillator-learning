"""Conditional drifting loss (Un-0), pixel-feature edition.

This is a faithful adaptation of `conditional_drift_loss` from
github.com/unconv-ai/Un-0 . The idea, per class c:

  * treat each generated sample as a particle in feature space;
  * compute a soft assignment to *positive* real samples of the same class and
    to *negative* samples (other-class reals + same-class generations);
  * form a drift vector  = (soft-nearest positives) - (soft-nearest negatives);
  * regress the generated features toward  (gen + drift).detach()  with an MSE.

So the supervision is a *direction to move* toward the data manifold and away
from where the generator is currently piling up mass -- a particle-flow /
distribution-matching objective, not a per-image reconstruction. This is the
"attractor toward the class manifold" behaviour described for Un-0.

We use raw pixels as the feature space (the paper's `pixel_weight` path); the
DINOv2 path is unnecessary at MNIST scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances, shape (Na, Nb)."""
    return torch.cdist(a, b, p=2).clamp_min(0) ** 2


def _drift_target(gen: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                  tau: float, neg_weight: float) -> torch.Tensor:
    """Compute (gen + drift) for one class. All inputs are (N, D) feature rows."""
    # Soft assignment from generated particles to positive reals.
    dpos = _pairwise_sq_dist(gen, pos)                       # (G, P)
    # Scale-invariant temperature: normalise by the mean distance so the
    # softmax sharpness does not depend on feature dimensionality / magnitude
    # (the repo "scales by a factor derived from the total sum of distances").
    scale_pos = dpos.mean().detach() + 1e-8
    # Doubly-normalised assignment: sqrt(row-softmax * col-softmax), as in the repo.
    logits_pos = -dpos / (tau * scale_pos)
    a_row = F.softmax(logits_pos, dim=1)
    a_col = F.softmax(logits_pos, dim=0)
    assign_pos = torch.sqrt(a_row * a_col + 1e-12)
    w_pos = assign_pos / (assign_pos.sum(dim=1, keepdim=True) + 1e-8)
    pull = w_pos @ pos                                       # (G, D) soft nearest reals

    drift = pull - gen
    if neg.shape[0] > 0 and neg_weight > 0:
        dneg = _pairwise_sq_dist(gen, neg)                  # (G, Nn)
        scale_neg = dneg.mean().detach() + 1e-8
        w_neg = F.softmax(-dneg / (tau * scale_neg), dim=1)
        push = w_neg @ neg                                  # (G, D)
        drift = drift - neg_weight * (push - gen)

    return (gen + drift).detach()


def conditional_drift_loss(x_gen: torch.Tensor, class_gen: torch.Tensor,
                           x_real: torch.Tensor, class_real: torch.Tensor,
                           tau: float = 1.0, neg_weight: float = 0.1,
                           gamma: float = 0.5) -> torch.Tensor:
    """Average per-class drifting loss.

    x_gen   : (G, C, H, W) generated images (with grad)
    x_real  : (R, C, H, W) real images (positives / negatives pool)
    *_class : (N,) integer class ids
    gamma   : fraction of other-class reals to mix into the negatives.
    """
    g = x_gen.flatten(1)
    r = x_real.flatten(1)

    losses = []
    present = torch.unique(class_gen)
    for c in present.tolist():
        gmask = class_gen == c
        gen_c = g[gmask]                                     # (G_c, D), has grad
        pos_c = r[class_real == c].detach()                 # (P_c, D)
        if pos_c.shape[0] == 0:
            continue

        # Negatives: same-class generated (detached, repel self-collapse)
        # + a gamma-fraction of other-class reals.
        neg_self = gen_c.detach()
        other = r[class_real != c].detach()
        if other.shape[0] > 0 and gamma > 0:
            k = max(1, int(gamma * other.shape[0]))
            idx = torch.randperm(other.shape[0], device=other.device)[:k]
            neg_c = torch.cat([neg_self, other[idx]], dim=0)
        else:
            neg_c = neg_self

        target = _drift_target(gen_c.detach(), pos_c, neg_c, tau, neg_weight)
        losses.append(F.mse_loss(gen_c, target))

    if not losses:
        return g.new_zeros(())
    return torch.stack(losses).mean()
