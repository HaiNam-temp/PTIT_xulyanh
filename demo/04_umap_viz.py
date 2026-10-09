"""UMAP kieu Hinh 3 cua bai bao: dac trung CLIP tho vs latent `mu` cua VIB-head.

Mau theo (loai anh x that/gia). Loai anh la nhan zero-shot CLIP (dong vat / vat nhan tao),
khong phai nhan that. Kem theo silhouette score: tach theo loai anh (cang thap cang tot, vi la
thong tin nhieu) va tach theo that/gia (cang cao cang tot).

Dung: python 04_umap_viz.py [beta]   (mac dinh 0.01)
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import umap
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from vib import VIBHead, train_head  # noqa: E402

HERE = Path(__file__).parent
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2


def main():
    d = np.load(HERE / "features.npz")
    X, y, split = d["feat"].astype(np.float32), d["label"], d["split"]
    tr = split == "train"

    # Nhan zero-shot argmax bi lech (~90% animal) -> dung hieu so diem roi chia theo trung vi
    # tren tap train de hai nhom can bang. Day van chi la nhan xap xi, khong phai nhan that.
    sys.path.insert(0, str(HERE.parent / "VIBAIGCDetect" / "models"))
    import clip
    cm, _ = clip.load("ViT-L/14", device="cpu")
    _tf = cm.transformer.forward
    cm.transformer.forward = lambda x: _tf(x)[1]   # repo tra ve tuple (out, x), chi lay x
    with torch.no_grad():
        txt = cm.encode_text(clip.tokenize(["a photo of an animal", "a photo of a man-made object"])).float()
        txt = txt / txt.norm(dim=-1, keepdim=True)
    Xn = torch.as_tensor(X) / torch.as_tensor(X).norm(dim=-1, keepdim=True)
    score = (Xn @ txt.T).numpy()
    diff = score[:, 1] - score[:, 0]          # lon = giong "vat nhan tao"
    cat = (diff > np.median(diff[tr])).astype(int)
    del cm

    net = VIBHead()
    train_head(net, X[tr], y[tr], beta=BETA, seed=0)
    net.eval()
    with torch.no_grad():
        mu = net.encode(torch.as_tensor(X))[0].numpy()

    # chi ve tap train (cung phan phoi) de moi nhom deu co du diem
    Xs, Ms, ys, cs = X[tr], mu[tr], y[tr], cat[tr]
    group = cs * 2 + ys   # 0:A-real 1:A-fake 2:B-real 3:B-fake
    names = ["animal real", "animal fake", "object real", "object fake"]
    colors = ["#4c78a8", "#aec7e8", "#e45756", "#ffb3b0"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rows = []
    for ax, (title, F) in zip(axes, [("CLIP ViT-L/14 raw features", Xs), (f"VIB-head latent mu (beta={BETA:g})", Ms)]):
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(F)
        for g in range(4):
            m = group == g
            ax.scatter(emb[m, 0], emb[m, 1], s=12, c=colors[g], label=f"{names[g]} (n={m.sum()})")
        s_cat = silhouette_score(F, cs)
        s_real = silhouette_score(F, ys)
        rows.append((title, s_cat, s_real))
        ax.set_title(f"{title}\nsilhouette: by category={s_cat:.3f}  by real/fake={s_real:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(HERE / "umap.png", dpi=130)
    for t, a, b in rows:
        print(f"{t:40s} silhouette(category)={a:.3f}  silhouette(real/fake)={b:.3f}")
    print("saved umap.png")


if __name__ == "__main__":
    main()
