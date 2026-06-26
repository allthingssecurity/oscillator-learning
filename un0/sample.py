"""Render a labeled class-conditional sample grid from a trained Un-0 checkpoint.

    python -m un0.sample --ckpt checkpoints/un0_mnist.pt --out samples/grid_labeled.png
"""

from __future__ import annotations

import argparse

import torch
from torchvision.utils import make_grid, save_image

from un0.model import ConditionalImplicitKuramotoGenerator, Un0Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/un0_mnist.pt")
    ap.add_argument("--out", default="samples/grid_labeled.png")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = Un0Config(**ckpt["cfg"])
    model = ConditionalImplicitKuramotoGenerator(cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    g = torch.Generator(device=dev).manual_seed(args.seed)
    classes = torch.arange(10, device=dev).repeat_interleave(args.per_class)
    imgs = model.sample(classes, generator=g)
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    grid = make_grid(imgs, nrow=args.per_class, padding=2, pad_value=0.3)
    save_image(grid, args.out)
    print(f"wrote {args.out}  (rows = digits 0-9, {args.per_class} samples each)")


if __name__ == "__main__":
    main()
