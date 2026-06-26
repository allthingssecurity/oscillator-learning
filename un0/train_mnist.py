"""Train the Un-0 coupled-oscillator generator on MNIST.

Usage:
    python -m un0.train_mnist --steps 3000 --batch 256

Outputs:
    samples/sample_XXXXXX.png   periodic 10x10 class-conditional grids
    samples/final_grid.png      final grid
    checkpoints/un0_mnist.pt    trained weights
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from un0.model import ConditionalImplicitKuramotoGenerator, Un0Config
from un0.losses import conditional_drift_loss


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_grid(model, device, path, n_per_class=10):
    classes = torch.arange(10, device=device).repeat_interleave(n_per_class)
    imgs = model.sample(classes)                       # (100, 1, 28, 28) in [-1, 1]
    imgs = (imgs.clamp(-1, 1) + 1) / 2                  # -> [0, 1]
    grid = make_grid(imgs, nrow=n_per_class)
    save_image(grid, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-main", type=int, default=256)
    ap.add_argument("--n-cond", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=8, help="oscillator integration steps")
    ap.add_argument("--solver", choices=["euler", "rk4"], default="rk4")
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--neg-weight", type=float, default=0.1)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),          # -> [-1, 1]
    ])
    ds = torchvision.datasets.MNIST(args.data_root, train=True, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=2, persistent_workers=True,
    )

    cfg = Un0Config(n_main=args.n_main, n_cond=args.n_cond,
                    n_steps=args.n_steps, solver=args.solver)
    model = ConditionalImplicitKuramotoGenerator(cfg).to(device)
    print("params:", model.num_params())

    # The dynamics parameters sit behind many integration steps and get smaller
    # gradients than the decoder, so give them a higher learning rate to make
    # the class-conditioning pathway learn at a comparable pace.
    dyn_params = list(model.dynamics.parameters())
    dec_params = list(model.decoder.parameters())
    opt = torch.optim.AdamW(
        [{"params": dyn_params, "lr": args.lr * 5},
         {"params": dec_params, "lr": args.lr}],
        betas=(0.9, 0.95),
    )

    Path("samples").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    model.train()
    step = 0
    t0 = time.time()
    ema = None
    data_iter = iter(loader)
    while step < args.steps:
        try:
            x_real, y_real = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x_real, y_real = next(data_iter)
        x_real, y_real = x_real.to(device), y_real.to(device)

        # Generate one image per real sample, with matched class distribution.
        y_gen = y_real[torch.randperm(y_real.shape[0], device=device)]
        x_gen = model(y_gen)

        loss = conditional_drift_loss(
            x_gen, y_gen, x_real, y_real,
            tau=args.tau, neg_weight=args.neg_weight,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        l = loss.item()
        ema = l if ema is None else 0.98 * ema + 0.02 * l
        if step % 50 == 0:
            rate = (step + 1) / (time.time() - t0)
            print(f"step {step:5d} | loss {l:.4f} | ema {ema:.4f} | {rate:5.1f} it/s")
        if step % args.sample_every == 0:
            sample_grid(model, device, f"samples/sample_{step:06d}.png")

        step += 1

    sample_grid(model, device, "samples/final_grid.png")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "checkpoints/un0_mnist.pt")
    print(f"done in {time.time() - t0:.1f}s -> samples/final_grid.png, checkpoints/un0_mnist.pt")


if __name__ == "__main__":
    main()
