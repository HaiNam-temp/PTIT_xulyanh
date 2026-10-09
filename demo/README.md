# Demo mô phỏng VIB-Net (CVPR 2025) — chạy trên CPU

Ý tưởng: bộ phân loại trên đặc trưng CLIP đóng băng, có/không có nút thắt thông tin biến phân (VIB),
huấn luyện chỉ trên **SD15 vs ảnh thật**, kiểm thử trên các generator **chưa thấy**.

```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python 01_get_data.py          # tải ~1.4GB parquet từ HF (TheKernel01/Tiny-GenImage), xuất 1040 ảnh 224x224
python 02_extract_features.py  # CLIP ViT-L/14 (repo tác giả), CPU -> features.npz
python 03_train_compare.py     # Linear vs VIB (beta quét) -> bảng AP/ACC + results.png
python 04_umap_viz.py [beta]   # UMAP kiểu Hình 3 + silhouette -> umap.png
```

Lưu ý: chạy các script từ thư mục khác `demo/` hoặc thư mục `VIBAIGCDetect/` đều được, nhưng
đừng đặt file lạ cạnh script (Python nạp module từ thư mục của script).

## Kết quả lần chạy này (CPU, 3 seed, tập đánh giá cân bằng 1:1, mức ngẫu nhiên = 50%)
| | In-dist SD15 | ADM | BigGAN | GLIDE | Midjourney | VQDM | Wukong | Unseen mean |
|---|---|---|---|---|---|---|---|---|
| AP Linear | 94.3 | 47.8 | 38.9 | 63.6 | 86.4 | 48.0 | 90.0 | 62.5 |
| AP VIB β=0 | 94.8 | 49.3 | 42.3 | 66.9 | 87.9 | 51.5 | 92.5 | 65.1 |
| AP VIB β=1e-2 | 95.5 | 49.7 | 41.8 | 64.8 | 84.1 | 52.0 | 90.2 | 63.8 |

- VIB-head nhỉnh hơn Linear khoảng 1–3 điểm AP trung bình trên unseen, nhưng **β>0 không tốt hơn β=0**
  và độ chênh nằm gần mức sai số (±1–3). Nên không chứng minh được tác dụng của nút thắt KL.
- ADM, VQDM, BigGAN gần/dưới mức ngẫu nhiên với mọi mô hình: demo nhỏ này **không tái hiện** được mức
  tổng quát hoá của bài (AP ~95).
- `umap.png`: sau khi huấn luyện, `mu` tách thật/giả rõ (silhouette 0.78) nhưng đây là **trên chính tập train**,
  không phải bằng chứng tổng quát hoá. Nhãn "loại ảnh" xấp xỉ nên trên đặc trưng CLIP thô cũng không thấy
  tách cụm theo loại rõ như Hình 3 của bài.

## Giới hạn (đọc trước khi kết luận)
- Tiny-GenImage không có SD v1.4 nên dùng SD15 (cùng họ). Wukong cũng là diffusion họ SD → "unseen" nhưng gần họ.
  Chỉ BigGAN là GAN thật sự khác họ.
- Dữ liệu nhỏ (400 ảnh train, 80 ảnh/generator lạ), 3 seed → sai số lớn, chỉ xem xu hướng.
- Ảnh được chuẩn hoá (resize 256 → crop 224 → JPEG q95) để giảm rò rỉ nhãn qua định dạng/kích thước,
  nhưng không loại bỏ hoàn toàn.
- "Loại ảnh" trong UMAP là nhãn zero-shot CLIP (animal / man-made object), không phải nhãn thật.
- Lúc đánh giá dùng `mu` (không lấy mẫu) nên khác `clip_models.py` (lấy mẫu ngẫu nhiên).
- Loss VIB = BCE + β·KL(N(mu,σ²)‖N(0,I)): phần KL/β do ta viết theo bài, repo gốc không có code huấn luyện.
