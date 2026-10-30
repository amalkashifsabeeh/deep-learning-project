"""Evaluate the matching model for several training seeds, and see how much the results move.

Example:  python seeds.py --seeds 0 1 2

Seed 0 is the model in runs/<kind>/. The other seeds are in runs/<kind>_seed<N>/ (see run_seeds.sh).
I use exactly the same test images and masks as evaluate.py, so the numbers are comparable.
This writes results/seeds_metrics.csv, results/seeds_summary.md and results/degradation_psnr_hole_seeds.png.
"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluate import load_model, model_predict
from inpaint.baselines import mean_image_fill, telea_fill
from inpaint.data import load_fashion_mnist, train_val_split
from inpaint.masks import KINDS, batch_masks
from inpaint.metrics import hole_mse, psnr_from_mse
from inpaint.plots import AQUA, BLUE, KIND_TITLE, ORANGE, _style


def run_dir(kind, seed):
    # seed 0 is the run I trained first, so it has no suffix
    return Path("runs") / (kind if seed == 0 else f"{kind}_seed{seed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--fracs", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.7])
    p.add_argument("--out", default="results")
    p.add_argument("--data", default="data")
    args = p.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, _ = train_val_split(load_fashion_mnist("train", args.data))
    mean_image = train.mean(0)
    test = load_fashion_mnist("test", args.data)[: args.n_eval]

    models = {(k, s): load_model(run_dir(k, s))[1] for k in KINDS for s in args.seeds}

    rows = []
    for ki, kind in enumerate(KINDS):
        for frac in args.fracs:
            # I use the same random generator seed as evaluate.py, so the masks are identical
            rng = np.random.default_rng(1000 + 100 * ki + int(round(frac * 100)))
            masks = batch_masks(kind, [frac] * len(test), rng)
            base = {"kind": kind, "frac_nominal": frac, "frac_achieved": float(masks.mean())}
            for name, pred in (("telea", telea_fill(test, masks)),
                               ("mean_image", mean_image_fill(test, masks, mean_image))):
                rows.append({**base, "method": name, "seed": -1,
                             "psnr_hole": float(psnr_from_mse(hole_mse(pred, test, masks)).mean())})
            for s in args.seeds:
                pred = model_predict(models[(kind, s)], test, masks)
                rows.append({**base, "method": "model", "seed": s,
                             "psnr_hole": float(psnr_from_mse(hole_mse(pred, test, masks)).mean())})
            print(f"{kind:7s} {frac:.2f} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "seeds_metrics.csv", index=False)

    # I make a table with the mean and standard deviation over the seeds, and the baselines next to it
    n = len(args.seeds)
    lines = [f"Hole PSNR (dB). Autoencoder: mean +/- standard deviation over {n} training seeds (min to max in brackets).", "",
             "| Mask type | % missing | Mean image | Telea | Autoencoder |", "|---|---|---|---|---|"]
    for kind in KINDS:
        for frac in args.fracs:
            sub = df[(df.kind == kind) & np.isclose(df.frac_nominal, frac)]
            m = sub[sub.method == "model"].psnr_hole
            get = lambda name: sub[sub.method == name].psnr_hole.iloc[0]
            lines.append(f"| {kind} | {sub.frac_achieved.iloc[0] * 100:.0f} | {get('mean_image'):.2f} | {get('telea'):.2f} | "
                         f"{m.mean():.2f} +/- {m.std(ddof=1):.2f} ({m.min():.2f} to {m.max():.2f}) |")
    # I want to know if the ranking of mask types holds in every seed (at about 50% missing)
    lines += ["", "Ranking of mask types at about 50% missing, one line per seed (hole PSNR of the matching model):", ""]
    holds = []
    for s in args.seeds:
        v = {k: df[(df.kind == k) & (df.seed == s) & np.isclose(df.frac_nominal, 0.5)].psnr_hole.iloc[0] for k in KINDS}
        ok = v["random"] > v["blocks"] > v["rect"]
        holds.append(ok)
        lines.append(f"- seed {s}: random {v['random']:.2f} > blocks {v['blocks']:.2f} > rect {v['rect']:.2f}: {'yes' if ok else 'NO'}")
    lines.append(f"\nThe order random > blocks > rect holds in {sum(holds)} of {n} seeds.")
    (out / "seeds_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # in the figure I draw the mean over the seeds as a line, and the lowest to highest seed as a band
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.5), sharey=True)
    for ax, kind in zip(axes, KINDS):
        sub = df[df.kind == kind]
        for name, color, marker, label in (("telea", ORANGE, "s", "Telea (classical)"),
                                           ("mean_image", AQUA, "^", "Mean image")):
            s_ = sub[sub.method == name].sort_values("frac_nominal")
            ax.plot(s_.frac_achieved * 100, s_.psnr_hole, color=color, lw=2, marker=marker, ms=6, label=label)
        g = sub[sub.method == "model"].groupby("frac_nominal")
        x = g.frac_achieved.first() * 100
        ax.fill_between(x, g.psnr_hole.min(), g.psnr_hole.max(), color=BLUE, alpha=0.25, lw=0)
        ax.plot(x, g.psnr_hole.mean(), color=BLUE, lw=2, marker="o", ms=6, label=f"Autoencoder (mean of {n} seeds)")
        ax.set_title(KIND_TITLE[kind])
        ax.set_xlabel("Pixels missing (%)")
    axes[0].set_ylabel("PSNR on missing pixels (dB)")
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "degradation_psnr_hole_seeds.png", dpi=200)
    print(f"wrote results to {out}/")


if __name__ == "__main__":
    main()
