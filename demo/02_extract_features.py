"""Trich dac trung CLIP ViT-L/14 (dong bang) mot lan cho toan bo anh -> features.npz

Dung lai thu vien CLIP trong repo tac gia (VIBAIGCDetect/models/clip) va tien xu ly nhu main.py
(Normalize theo mean/std cua CLIP). Dong thoi gan nhan "loai anh" (dong vat / vat do con nguoi
tao ra) bang zero-shot CLIP de dung cho UMAP kieu Hinh 3 cua bai bao.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "VIBAIGCDetect" / "models"))
import clip  # noqa: E402  (ban CLIP cua repo)

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
PROMPTS = ["a photo of an animal", "a photo of a man-made object"]


def main():
    torch.manual_seed(0)
    meta = json.loads((HERE / "data/meta.json").read_text())
    model, _ = clip.load("ViT-L/14", device="cpu")
    model.eval()
    # Repo sua Transformer de tra ve (out, x) -> encode_text cua nhanh text bi loi; chi lay x.
    # model.transformer (text) la module rieng, khong anh huong model.visual.transformer.
    _text_forward = model.transformer.forward
    model.transformer.forward = lambda x: _text_forward(x)[1]
    tf =T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    with torch.no_grad():
        txt = model.encode_text(clip.tokenize(PROMPTS)).float()
        txt = txt / txt.norm(dim=-1, keepdim=True)

    feats, cats = [], []
    bs, t0 = 32, time.time()
    with torch.no_grad():
        for i in range(0, len(meta), bs):
            batch = meta[i:i + bs]
            x = torch.stack([tf(Image.open(HERE / "data" / m["path"]).convert("RGB")) for m in batch])
            f = model.encode_image(x).float()
            fn = f / f.norm(dim=-1, keepdim=True)
            cats.append((fn @ txt.T).argmax(1).numpy())
            feats.append(f.numpy())
            print(f"{i + len(batch)}/{len(meta)}  {time.time() - t0:.0f}s", flush=True)

    feats = np.concatenate(feats)
    np.savez(
        HERE / "features.npz",
        feat=feats,
        label=np.array([m["label"] for m in meta]),
        generator=np.array([m["generator"] for m in meta]),
        split=np.array([m["split"] for m in meta]),
        category=np.concatenate(cats),
    )
    print("features", feats.shape, "-> features.npz")


if __name__ == "__main__":
    main()
