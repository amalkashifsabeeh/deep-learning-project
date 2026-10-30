"""Evaluate the trained models and the baselines on held-out Fashion-MNIST test images.

Example:  python evaluate.py --runs runs/rect runs/random runs/blocks

This writes results/metrics.csv, results/summary.md and the figures into results/.
Every method sees exactly the same masks, and every output has the visible pixels pasted back in.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inpaint.baselines import mean_fill, mean_image_fill, telea_fill
from inpaint.data import load_fashion_mnist, train_val_split
from inpaint.masks import KINDS, batch_masks
from inpaint.metrics import all_metrics
from inpaint.models import InpaintAE
from inpaint.plots import plot_cross_matrix, plot_degradation, plot_recon_grid


def load_model(run_dir):
    # the checkpoint stores the bottleneck size, so I can rebuild the same network before loading the weights
    ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu")
    m = InpaintAE(ck["latent"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return ck["kind"], m


@torch.no_grad()
def model_predict(model, images, masks, bs=1000):
    outs = []  # predicted in chunks of bs images to keep memory small
    for i in range(0, len(images), bs):
        x = torch.from_numpy(images[i:i + bs]).float().unsqueeze(1)
        m = torch.from_numpy(masks[i:i + bs]).float().unsqueeze(1)
        outs.append(model.inpaint(x, m).squeeze(1).numpy())
    return np.concatenate(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs/rect", "runs/random", "runs/blocks"])
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--fracs", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.7])
    p.add_argument("--out", default="results")
    p.add_argument("--data", default="data")
    args = p.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, _ = train_val_split(load_fashion_mnist("train", args.data))
    mean_image = train.mean(0)  # from the training images only, never the test set
    test = load_fashion_mnist("test", args.data)[: args.n_eval]
    models = dict(load_model(r) for r in args.runs if (Path(r) / "model.pt").exists())
    print("models:", list(models))

    metric_names = ["mse_hole", "psnr_hole", "ssim_hole", "mse_full", "psnr_full", "ssim_full"]
    rows = []
    for ki, kind in enumerate(KINDS):
        for frac in args.fracs:
            # the generator seed depends only on mask type and fraction, so seeds.py can recreate exactly these masks
            rng = np.random.default_rng(1000 + 100 * ki + int(round(frac * 100)))
            masks = batch_masks(kind, [frac] * len(test), rng)
            preds = {
                "mean_fill": mean_fill(test, masks),
                "mean_image": mean_image_fill(test, masks, mean_image),
                "telea": telea_fill(test, masks),
            }
            for tk, m in models.items():
                preds[f"model:{tk}"] = model_predict(m, test, masks)
            # every method is scored on the same images and the same masks, so the numbers are comparable
            for method, pred in preds.items():
                met = all_metrics(pred, test, masks)
                row = {"test_kind": kind, "frac_nominal": frac, "frac_achieved": float(masks.mean()),
                       "method": method, "n": len(test)}
                for name in metric_names:
                    v = met[name][~np.isnan(met[name])]  # drop images where SSIM on the hole is undefined
                    row[f"{name}_mean"] = float(v.mean())
                    # standard error of the mean over test images; it does not include variation between training runs
                    row[f"{name}_sem"] = float(v.std(ddof=1) / np.sqrt(len(v)))
                rows.append(row)
            print(f"{kind:7s} {frac:.2f} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)

    # a small table of hole PSNR, with the matching model next to the baselines
    lines = ["| Mask type | % missing | Visible-pixel mean | Mean image | Telea | Autoencoder (matched) |",
             "|---|---|---|---|---|---|"]
    for kind in KINDS:
        for frac in args.fracs:
            s = df[(df.test_kind == kind) & np.isclose(df.frac_nominal, frac)].set_index("method")
            g = lambda mth: f"{s.loc[mth, 'psnr_hole_mean']:.2f}" if mth in s.index else "n/a"
            lines.append(f"| {kind} | {s.frac_achieved.iloc[0] * 100:.0f} | {g('mean_fill')} | {g('mean_image')} | "
                         f"{g('telea')} | {g(f'model:{kind}')} |")
    (out / "summary.md").write_text("PSNR on missing pixels (dB, higher is better)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_degradation(df, out / "degradation_psnr_hole.png", "psnr_hole")
    plot_degradation(df, out / "degradation_mse_hole.png", "mse_hole")
    mid = min(args.fracs, key=lambda f: abs(f - 0.5))
    if len(models) > 1:
        plot_cross_matrix(df, out / "cross_mask_generalisation.png", mid)

    # example images to look at: the same 2 test images at every corruption level, for each mask type
    img_idx = [0, 7]
    for ki, kind in enumerate(KINDS):
        imgs, msks, labels = [], [], []
        rng = np.random.default_rng(7 + ki)
        for frac in args.fracs:
            for i in img_idx:
                imgs.append(test[i])
                msks.append(batch_masks(kind, [frac], rng)[0])
        imgs, msks = np.stack(imgs), np.stack(msks)
        labels = [f"{m.mean() * 100:.0f}%" for m in msks]
        rows_ = [("Telea", telea_fill(imgs, msks)), ("Mean image", mean_image_fill(imgs, msks, mean_image))]
        if kind in models:
            rows_.append(("Autoencoder", model_predict(models[kind], imgs, msks)))
        plot_recon_grid(imgs, msks, rows_, out / f"recon_{kind}.png", labels,
                        title=f"Reconstructions, {kind} masks (columns: % missing)")
    print(f"wrote results to {out}/")


if __name__ == "__main__":
    main()
