"""Train KuramotoGPT (nanochat-style, oscillator mixer) on char-level Shakespeare.

    python -m nanochat_kuramoto.train --steps 4000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nanochat_kuramoto.model import KuramotoGPT, KGPTConfig

HERE = Path(__file__).parent


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-osc", type=int, default=256)
    ap.add_argument("--osc-steps", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    # ---- char-level tokenizer ----
    text = (HERE / "data" / "input.txt").read_text()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus {len(data):,} chars | vocab {vocab_size}")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block - 1, (args.batch,))
        x = torch.stack([d[i:i + args.block] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + args.block] for i in ix])
        return x.to(device), y.to(device)

    cfg = KGPTConfig(vocab_size=vocab_size, sequence_len=args.block,
                     n_layer=args.n_layer, n_embd=args.n_embd,
                     n_osc=args.n_osc, osc_steps=args.osc_steps)
    model = KuramotoGPT(cfg).to(device)
    print(f"params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(20)
            for k in range(20):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def sample(prompt="\n", n_tokens=400, temp=0.8):
        ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(ids, n_tokens, temperature=temp)[0].tolist()
        model.train()
        return decode(out)

    Path(HERE / "samples").mkdir(exist_ok=True)
    print("\n=== sample BEFORE training ===")
    print(sample())

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            rate = (step + 1) / (time.time() - t0)
            print(f"step {step:5d} | loss {loss.item():.3f} | {rate:5.1f} it/s")
        if step > 0 and step % args.eval_every == 0:
            losses = estimate_loss()
            print(f"  [eval] train {losses['train']:.3f} | val {losses['val']:.3f}")
            txt = sample()
            (HERE / "samples" / f"sample_{step:05d}.txt").write_text(txt)
            print(f"  --- sample @ {step} ---\n{txt[:300]}\n")

    losses = estimate_loss()
    print(f"\nFINAL eval train {losses['train']:.3f} | val {losses['val']:.3f}")
    final = sample(n_tokens=1000)
    (HERE / "samples" / "final.txt").write_text(final)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "stoi": stoi, "itos": itos}, HERE / "ckpt.pt")
    print(f"done in {time.time()-t0:.0f}s\n\n=== FINAL SAMPLE ===\n{final}")


if __name__ == "__main__":
    main()
