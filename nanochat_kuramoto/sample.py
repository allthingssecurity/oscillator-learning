"""Generate text from a trained KuramotoGPT checkpoint.

    python -m nanochat_kuramoto.sample --prompt $'KING RICHARD:\n' --tokens 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nanochat_kuramoto.model import KuramotoGPT, KGPTConfig

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "ckpt.pt"))
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = KGPTConfig(**ck["cfg"])
    stoi, itos = ck["stoi"], ck["itos"]
    model = KuramotoGPT(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=dev)
    out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k)[0].tolist()
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
