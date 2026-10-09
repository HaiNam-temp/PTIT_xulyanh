"""So sanh baseline tuyen tinh (Univfd-like) va VIB-head tren dac trung CLIP da cache.

Huan luyen chi tren SD15 vs that; danh gia tren (a) phan giu lai cua cung phan phoi (SD15)
va (b) cac generator CHUA THAY. Khong chon beta/epoch dua tren tap unseen (tranh ro ri).
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vib import LinearHead, VIBHead, ap_acc, predict, train_head  # noqa: E402

HERE = Path(__file__).parent
SEEDS = [0, 1, 2]
BETAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
UNSEEN_GENS = ["ADM", "BigGAN", "GLIDE", "Midjourney", "VQDM", "Wukong"]


def main():
    d = np.load(HERE / "features.npz")
    X, y, gen, split = d["feat"].astype(np.float32), d["label"], d["generator"], d["split"]
    tr_idx = np.where(split == "train")[0]
    un_real = np.where((split == "unseen") & (y == 0))[0]
    un_fake = {g: np.where((split == "unseen") & (gen == g))[0] for g in UNSEEN_GENS}

    models = {"Linear (Univfd-like)": lambda: (LinearHead(), 0.0)}
    for b in BETAS:
        models[f"VIB beta={b:g}"] = (lambda b=b: (VIBHead(), b))

    cols = ["In-dist (SD15)"] + UNSEEN_GENS + ["Unseen mean"]
    res = {m: {"AP": [], "ACC": []} for m in models}

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(tr_idx)
        cut = int(0.8 * len(perm))
        fit, held = perm[:cut], perm[cut:]
        for name, mk in models.items():
            net, beta = mk()
            loss = train_head(net, X[fit], y[fit], beta=beta, seed=seed)
            ap_row, acc_row = [], []
            p = predict(net, X[held])
            a, c = ap_acc(y[held], p)
            ap_row.append(a); acc_row.append(c)
            for g in UNSEEN_GENS:
                # can bang 1:1 (chance AP = ACC = 50%): lay so anh that bang so anh gia cua generator
                real_sub = rng.choice(un_real, len(un_fake[g]), replace=False)
                idx = np.concatenate([real_sub, un_fake[g]])
                a, c = ap_acc(y[idx], predict(net, X[idx]))
                ap_row.append(a); acc_row.append(c)
            ap_row.append(float(np.mean(ap_row[1:]))); acc_row.append(float(np.mean(acc_row[1:])))
            res[name]["AP"].append(ap_row); res[name]["ACC"].append(acc_row)
            print(f"seed {seed} {name:22s} loss={loss:.4f} AP in-dist={ap_row[0]:.3f} unseen mean={ap_row[-1]:.3f}", flush=True)

    # ---- bang ket qua (mean +- std qua cac seed) ----
    out = {}
    for metric in ["AP", "ACC"]:
        print(f"\n=== {metric} (%), mean +- std qua {len(SEEDS)} seed ===")
        print(f"{'model':24s}" + "".join(f"{c[:11]:>14s}" for c in cols))
        for m in models:
            arr = np.array(res[m][metric]) * 100
            mean, std = arr.mean(0), arr.std(0)
            print(f"{m:24s}" + "".join(f"{a:8.1f}+-{s:3.1f}" for a, s in zip(mean, std)))
            out.setdefault(m, {})[metric] = {"mean": mean.tolist(), "std": std.tolist()}
    (HERE / "results.json").write_text(json.dumps({"columns": cols, "results": out}, indent=1))

    # ---- bieu do ----
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    w = 0.8 / len(models)
    for ax, metric in zip(axes, ["AP", "ACC"]):
        for i, m in enumerate(models):
            mean = np.array(out[m][metric]["mean"]); std = np.array(out[m][metric]["std"])
            ax.bar(np.arange(len(cols)) + i * w, mean, w, yerr=std, label=m, capsize=2)
        ax.set_ylabel(f"{metric} (%)")
        ax.axhline(50, color="gray", lw=0.5, ls="--")
        ax.set_ylim(30, 102)  # duong gach = muc ngau nhien (50%) vi tap danh gia da can bang
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_xticks(np.arange(len(cols)) + 0.4 - w / 2)
    axes[1].set_xticklabels(cols, rotation=20)
    axes[0].set_title("Huan luyen tren SD15 vs that -> kiem thu tren generator chua thay")
    plt.tight_layout()
    plt.savefig(HERE / "results.png", dpi=130)
    print("saved results.png, results.json")


if __name__ == "__main__":
    main()
