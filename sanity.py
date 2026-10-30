"""Sanity checks. I run them before and after training:  python sanity.py

They look for bugs that would quietly ruin the results, such as leaking the hole contents to the
model, getting the wrong missing fraction, or using metrics that cannot tell good from bad.
The log goes to results/sanity.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inpaint.baselines import mean_fill, telea_fill
from inpaint.data import load_fashion_mnist, train_val_split
from inpaint.masks import KINDS, batch_masks
from inpaint.metrics import all_metrics, hole_mse
from inpaint.models import InpaintAE, inpaint_loss

LOG = []
FAILED = []


def report(ok, name, detail=""):
    line = f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")
    print(line)
    LOG.append(line)
    if not ok:
        FAILED.append(name)


def info(line):
    print(line)
    LOG.append(line)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(1)


def main():
    torch.manual_seed(0)
    train, _ = train_val_split(load_fashion_mnist("train"))
    test = load_fashion_mnist("test")[:500]
    rng = np.random.default_rng(0)

    # 1. I check that the mask generators give the fraction I ask for, up to rounding to whole pixels
    for kind in KINDS:
        for frac in (0.1, 0.25, 0.5, 0.7):
            got = batch_masks(kind, [frac] * 300, rng).mean()
            tol = 0.001 if kind == "random" else 0.045
            report(abs(got - frac) <= tol, f"mask fraction {kind} @ {frac:.2f}", f"achieved {got:.3f}")

    # 2. I check that the metrics can tell a perfect prediction from a bad one
    masks = batch_masks("rect", [0.3] * 500, rng)
    perfect = all_metrics(test, test, masks)
    report(np.allclose(perfect["mse_hole"], 0) and np.allclose(perfect["ssim_full"], 1.0),
           "perfect prediction gives MSE 0 and SSIM 1")
    noisy = np.clip(test + rng.normal(0, 0.3, test.shape), 0, 1).astype(np.float32)
    noisy = np.where(masks, noisy, test)
    report(all_metrics(noisy, test, masks)["psnr_hole"].mean() < perfect["psnr_hole"].mean(),
           "noisy prediction scores worse than perfect")

    # 3. I check that the baselines leave the visible pixels alone, and that Telea beats mean fill on small random holes
    m10 = batch_masks("random", [0.1] * 500, rng)
    mf, tl = mean_fill(test, m10), telea_fill(test, m10)
    report(np.array_equal(mf[~m10], test[~m10]) and np.array_equal(tl[~m10], test[~m10]),
           "baselines leave visible pixels unchanged")
    e_mf, e_tl = hole_mse(mf, test, m10).mean(), hole_mse(tl, test, m10).mean()
    report(e_tl < e_mf, "Telea beats visible-pixel mean on 10% random holes", f"{e_tl:.4f} < {e_mf:.4f}")

    # 4. I check that the model keeps the visible pixels and cannot see what is inside the hole (no leakage)
    model = InpaintAE(64).eval()
    x, m = to_t(test[:64]), to_t(batch_masks("blocks", [0.4] * 64, rng))
    out = model.inpaint(x, m)
    report(torch.equal(out * (1 - m), x * (1 - m)), "model output keeps visible pixels exactly")
    x_scrambled = x * (1 - m) + torch.rand_like(x) * m
    report(torch.allclose(model(x, m), model(x_scrambled, m)), "model output independent of hole contents (no leakage)")

    # 5. I check that the model can overfit one fixed batch, because if it cannot, something is wrong with the loss, optimiser or data
    x, m = to_t(train[:64]), to_t(batch_masks("mixed", rng.uniform(0.1, 0.7, 64), rng))
    net = InpaintAE(64)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    first = None
    for step in range(400):
        loss, hole = inpaint_loss(net(x, m), x, m)
        opt.zero_grad(); loss.backward(); opt.step()
        first = float(hole) if first is None else first
    report(float(hole) < 0.2 * first, "overfits a single batch", f"hole MSE {first:.4f} -> {float(hole):.4f}")

    # 6. I look at the results themselves. These checks only warn, and they only run after evaluate.py has been run
    csv = Path("results/metrics.csv")
    if csv.exists():
        df = pd.read_csv(csv)
        for kind in KINDS:
            s = df[(df.test_kind == kind) & (df.method == f"model:{kind}")].sort_values("frac_nominal")
            if s.empty:
                continue
            mono = bool((np.diff(s.mse_hole_mean.values) > 0).all())
            info(f"[{'ok' if mono else 'WARN'}] {kind}: model hole-MSE increases with corruption: "
                 + ", ".join(f"{v:.4f}" for v in s.mse_hole_mean))
            for _, r in s.iterrows():
                b = df[(df.test_kind == kind) & np.isclose(df.frac_nominal, r.frac_nominal) & (df.method == "mean_fill")]
                better = r.mse_hole_mean < b.mse_hole_mean.iloc[0]
                info(f"[{'ok' if better else 'WARN'}] {kind} @ {r.frac_nominal:.2f}: model "
                     f"{'beats' if better else 'does NOT beat'} visible-pixel mean fill")

    Path("results").mkdir(exist_ok=True)
    Path("results/sanity.txt").write_text("\n".join(LOG) + "\n")
    print(f"\n{len(FAILED)} failed checks" if FAILED else "\nall hard checks passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
