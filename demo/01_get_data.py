"""Tai du lieu nho tu Tiny-GenImage (HuggingFace) va chuan hoa anh.

- train : anh that vs anh gia SD15 (cung ho Stable Diffusion voi SD v1.4 trong bai bao)
- unseen: anh gia tu generator KHAC (ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong) + anh that giu rieng

Moi anh deu duoc resize canh ngan -> 256, center-crop 224, luu JPEG q=95 de tranh
lo nhan qua dinh dang/kich thuoc (GenImage goc: that = JPEG, gia = PNG).
"""
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "TheKernel01/Tiny-GenImage"
OUT = Path(__file__).parent / "data"
SEED = 0
N_TRAIN = 200          # moi lop (that / SD15)
N_UNSEEN_FAKE = 80     # moi generator la
N_UNSEEN_REAL = 160
# chi so trong cot `generator` cua dataset
GEN = {0: "Real", 1: "ADM", 2: "BigGAN", 3: "GLIDE", 4: "Midjourney", 6: "SD15", 7: "VQDM", 8: "Wukong"}
TRAIN_FILES = ["data/train-00000-of-00014.parquet", "data/train-00001-of-00014.parquet"]
UNSEEN_FILE = "data/validation-00000-of-00004.parquet"


def load_shard(fname):
    path = hf_hub_download(REPO, fname, repo_type="dataset")
    t = pq.read_table(path, columns=["image", "generator"])
    imgs = t.column("image").to_pylist()   # [{'bytes':..., 'path':...}]
    gens = t.column("generator").to_pylist()
    return imgs, gens


def normalize(raw):
    img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return img.crop((l, t, l + 224, t + 224))


def collect(files):
    by_gen = {}
    for f in files:
        imgs, gens = load_shard(f)
        for im, g in zip(imgs, gens):
            by_gen.setdefault(GEN[g], []).append(im)
    return by_gen


def save(items, folder, prefix, meta, split, label, gen):
    folder.mkdir(parents=True, exist_ok=True)
    for i, raw in enumerate(items):
        p = folder / f"{prefix}_{i:04d}.jpg"
        normalize(raw).save(p, quality=95)
        meta.append(dict(path=str(p.relative_to(OUT)), split=split, label=label, generator=gen))


def main():
    random.seed(SEED)
    meta = []

    tr = collect(TRAIN_FILES)
    for k in tr:
        random.shuffle(tr[k])
    print({k: len(v) for k, v in tr.items()})
    save(tr["SD15"][:N_TRAIN], OUT / "train/fake", "sd15", meta, "train", 1, "SD15")
    save(tr["Real"][:N_TRAIN], OUT / "train/real", "real", meta, "train", 0, "Real")

    un = collect([UNSEEN_FILE])
    for k in un:
        random.shuffle(un[k])
    print({k: len(v) for k, v in un.items()})
    save(un["Real"][:N_UNSEEN_REAL], OUT / "unseen/real", "real", meta, "unseen", 0, "Real")
    for g in ["ADM", "BigGAN", "GLIDE", "Midjourney", "VQDM", "Wukong"]:
        save(un[g][:N_UNSEEN_FAKE], OUT / f"unseen/{g}", g.lower(), meta, "unseen", 1, g)

    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", len(meta), "images ->", OUT)


if __name__ == "__main__":
    main()
