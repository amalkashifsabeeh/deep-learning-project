"""Train the inpainting autoencoder.

Example:  python train.py --kind rect --epochs 8 --out runs/rect

Every training step draws new random masks, with the missing fraction picked between frac_min and
frac_max. So the model sees a different corruption of each image in every epoch.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from inpaint.baselines import mean_fill
from inpaint.data import load_fashion_mnist, train_val_split
from inpaint.masks import batch_masks
from inpaint.metrics import hole_mse
from inpaint.models import InpaintAE, inpaint_loss
from inpaint.plots import plot_loss_curves


def to_t(a, device):
    # numpy (N, 28, 28) to a float tensor (N, 1, 28, 28); the extra axis is the channel the conv layers expect
    return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(1).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="rect", choices=["rect", "random", "blocks", "mixed"])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--frac_min", type=float, default=0.1)
    p.add_argument("--frac_max", type=float, default=0.7)
    p.add_argument("--n_train", type=int, default=55000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--overfit_batch", action="store_true", help="sanity check: memorise one fixed batch")
    p.add_argument("--data", default="data")
    args = p.parse_args()

    out = Path(args.out or f"runs/{args.kind}")
    out.mkdir(parents=True, exist_ok=True)
    # one seed controls the weights (torch) and the masks and batch order (numpy), so a run can be repeated
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val = train_val_split(load_fashion_mnist("train", args.data))
    train = train[: args.n_train]  # lets me train on fewer images for quick tests
    # I fix the validation masks once, so the validation curve can be compared from epoch to epoch
    vrng = np.random.default_rng(1234)
    v_fracs = vrng.uniform(args.frac_min, args.frac_max, len(val))
    v_masks = batch_masks(args.kind, v_fracs, vrng)
    v_img_t, v_mask_t = to_t(val, device), to_t(v_masks, device)
    # the visible-pixel mean on these same masks is the line the model has to beat in the loss plot
    val_meanfill = float(hole_mse(mean_fill(val, v_masks), val, v_masks).mean())

    model = InpaintAE(args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"kind={args.kind} params={sum(p.numel() for p in model.parameters()):,} device={device}")

    hist = {"train_hole_mse": [], "val_hole_mse": [], "val_meanfill_hole_mse": val_meanfill,
            "args": vars(args), "epoch_seconds": []}
    # for the overfit check: one fixed batch with fixed masks, which a working model should be able to memorise
    fixed = None
    if args.overfit_batch:
        idx = np.arange(args.batch)
        f = rng.uniform(args.frac_min, args.frac_max, args.batch)
        fixed = (to_t(train[idx], device), to_t(batch_masks(args.kind, f, rng), device))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train)) if fixed is None else np.zeros(args.batch, dtype=int)
        losses = []
        n_steps = (len(train) // args.batch) if fixed is None else 100
        for step in range(n_steps):
            if fixed is None:
                idx = order[step * args.batch:(step + 1) * args.batch]
                # new masks at every step, so the model never sees the same corruption of an image twice
                fracs = rng.uniform(args.frac_min, args.frac_max, len(idx))
                x = to_t(train[idx], device)
                m = to_t(batch_masks(args.kind, fracs, rng), device)
            else:
                x, m = fixed
            loss, hole = inpaint_loss(model(x, m), x, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(hole))  # I log the hole MSE only, so train and validation use the same measure
        # validation in chunks of 1000 to keep memory small; the masks are the fixed ones from above
        model.eval()
        with torch.no_grad():
            vo = torch.cat([model(v_img_t[i:i + 1000], v_mask_t[i:i + 1000]) for i in range(0, len(val), 1000)])
            vh = float((((vo - v_img_t) ** 2) * v_mask_t).sum() / v_mask_t.sum())
        hist["train_hole_mse"].append(float(np.mean(losses)))
        hist["val_hole_mse"].append(vh)
        hist["epoch_seconds"].append(time.time() - t0)
        print(f"epoch {epoch:2d}  train hole-MSE {hist['train_hole_mse'][-1]:.4f}  "
              f"val hole-MSE {vh:.4f}  (visible-mean baseline {val_meanfill:.4f})  {hist['epoch_seconds'][-1]:.0f}s")

    torch.save({"state_dict": model.state_dict(), "latent": args.latent, "kind": args.kind, "args": vars(args)},
               out / "model.pt")
    json.dump(hist, open(out / "history.json", "w"), indent=1)
    plot_loss_curves(out / "history.json", out / "loss_curve.png", title=f"Training curves ({args.kind} masks)")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
