# Phát hiện ảnh do AI tạo ra bằng VIB-Net — demo mô phỏng bài báo CVPR 2025

Bài tập môn **Xử lý ảnh (PTIT)**. Repo này tóm tắt và chạy thử lại ý tưởng chính của bài:

> **Towards Universal AI-Generated Image Detection by Variational Information Bottleneck Network**
> Haifeng Zhang và cộng sự, CVPR 2025 — [bài báo](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Towards_Universal_AI-Generated_Image_Detection_by_Variational_Information_Bottleneck_Network_CVPR_2025_paper.pdf), [code gốc](https://github.com/oceanzhf/VIBAIGCDetect)

👉 Muốn trình bày demo? Đọc **[KICH_BAN_DEMO.md](KICH_BAN_DEMO.md)**.

---

## 1. Bài báo nói gì? (giải thích ngắn)

**Bài toán:** cho một bức ảnh, đoán xem đó là ảnh **thật** hay ảnh do **AI tạo ra** (Stable Diffusion, Midjourney, GAN…).

**Khó khăn:** máy phát hiện thường chỉ giỏi với loại AI nó đã được học. Gặp một loại AI mới là đoán sai nhiều.

**Vì sao?** Các máy phát hiện hiện nay dùng **CLIP** (một mô hình lớn hiểu nội dung ảnh) để lấy đặc trưng. Nhưng đặc trưng CLIP chứa quá nhiều thông tin về **nội dung** (con mèo, cái xe, cái cây…), không liên quan gì đến thật hay giả. Máy dễ học "mẹo" sai, ví dụ *"ảnh có cái cây thì là ảnh giả"*.

**Giải pháp của bài — VIB (Variational Information Bottleneck, "nút thắt thông tin"):**
ép đặc trưng CLIP đi qua một "cổ chai" nhỏ. Cổ chai chỉ cho đi qua phần thông tin **cần thiết để phân biệt thật/giả**, và bỏ phần nội dung thừa.

```
Ảnh ──► CLIP (đóng băng) ──► đặc trưng 768 chiều ──► [ VIB: nén còn 256 chiều ] ──► Thật / Giả
                                                       giữ thông tin thật/giả,
                                                       bỏ thông tin nội dung
```

Hàm mất mát: `Loss = BCE (phân loại đúng) + β × KL (phạt nếu giữ quá nhiều thông tin)`

**Kết quả trong bài:** thử trên 17 loại AI, cao hơn phương pháp tốt nhất trước đó khoảng **+5.5% mAP** và **+9.3% độ chính xác**.

---

## 2. Repo này có gì?

```
.
├── README.md                ← file này
├── KICH_BAN_DEMO.md         ← kịch bản trình bày demo
├── 26. Zhang_..._paper.pdf  ← bài báo gốc
├── VIBAIGCDetect/           ← code gốc của tác giả (submodule, chỉ có code test)
└── demo/                    ← demo do mình viết
    ├── VIBNet_demo.ipynb    ← notebook chạy trên máy (CPU), đã có sẵn kết quả
    ├── VIBNet_kaggle.ipynb  ← notebook chạy trên Kaggle (GPU), dữ liệu lớn hơn
    ├── 01_get_data.py       ← tải dữ liệu
    ├── 02_extract_features.py ← trích đặc trưng CLIP
    ├── 03_train_compare.py  ← huấn luyện, so sánh Linear vs VIB
    ├── 04_umap_viz.py       ← vẽ UMAP
    ├── vib.py               ← mô hình VIB-head + hàm mất mát
    └── results.png, umap.png ← kết quả
```

**Code gốc của tác giả** chỉ có phần **kiểm thử**, không có code huấn luyện và không có hàm mất mát KL. Phần huấn luyện trong `demo/` do mình viết lại theo công thức trong bài. Kiến trúc mô hình giống hệt file `VIBAIGCDetect/models/clip_models.py`.

---

## 3. Demo làm gì?

**Câu hỏi:** chỉ cho máy học ảnh **thật** và ảnh **Stable Diffusion 1.5**, máy có nhận ra ảnh của các AI **chưa từng gặp** không? VIB có giúp không?

| | Dữ liệu |
|---|---|
| Huấn luyện | ảnh thật + ảnh SD15 |
| Kiểm thử ("đề thi lạ") | ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong |

Dữ liệu lấy từ [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) (bản thu nhỏ của GenImage mà bài dùng).

So sánh hai "bộ não" đặt sau CLIP:
- **Linear**: một lớp tuyến tính đơn giản (kiểu phương pháp Univfd, CVPR 2023).
- **VIB**: khối nút thắt thông tin của bài báo, thử nhiều mức β.

---

## 4. Cách chạy

### Cách A — Kaggle (khuyên dùng, có GPU)
1. Lên [kaggle.com](https://www.kaggle.com) → **Create → New Notebook → File → Import Notebook** → chọn `demo/VIBNet_kaggle.ipynb`.
2. Bên phải: **Accelerator = GPU T4**, **Internet = On**.
3. Bấm **Run All**. Notebook tự cài thư viện, tự tải code tác giả và dữ liệu (~8GB).

### Cách B — Máy cá nhân (CPU)
```bash
git clone --recursive https://github.com/HaiNam-temp/PTIT_xulyanh
cd PTIT_xulyanh/demo
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```
Sau đó mở `VIBNet_demo.ipynb` bằng VS Code hoặc Jupyter → **Run All**. Dữ liệu và đặc trưng đã có sẵn trong repo, chạy khoảng 2 phút.

> Lỡ clone không có `--recursive` (thư mục `VIBAIGCDetect` bị rỗng)? Chạy `git submodule update --init`.

Hoặc chạy từng bước bằng script:
```bash
python 01_get_data.py          # tải dữ liệu
python 02_extract_features.py  # trích đặc trưng CLIP (~18 phút trên CPU)
python 03_train_compare.py     # huấn luyện + so sánh → results.png
python 04_umap_viz.py          # UMAP → umap.png
```

---

## 5. Kết quả (bản CPU, dữ liệu nhỏ)

AP (%) — càng cao càng tốt, **50% = đoán bừa**:

| Mô hình | SD15 (đã học) | ADM | BigGAN | GLIDE | Midjourney | VQDM | Wukong | **TB đề lạ** |
|---|---|---|---|---|---|---|---|---|
| Linear | 94.3 | 47.8 | 38.9 | 63.6 | 86.4 | 48.0 | 90.0 | **62.5** |
| VIB β=0 | 94.8 | 49.3 | 42.3 | 66.9 | 87.9 | 51.5 | 92.5 | **65.1** |
| VIB β=0.01 | 95.5 | 49.7 | 41.8 | 64.8 | 84.1 | 52.0 | 90.2 | **63.8** |

**Nhận xét thẳng thắn:**
- ✅ Loại AI đã học (SD15) và loại gần giống (Wukong, Midjourney): phát hiện tốt.
- ❌ Loại AI khác hẳn (ADM, VQDM, BigGAN): gần như đoán bừa. Đây đúng là vấn đề bài báo muốn giải quyết.
- ➖ VIB nhỉnh hơn Linear 1–3 điểm, nhưng mức chênh xấp xỉ sai số, và β > 0 không tốt hơn β = 0. Demo nhỏ này **chưa chứng minh được** tác dụng của nút thắt thông tin.
- Lý do chính: dữ liệu ít hơn bài khoảng **80 lần** và không dùng augmentation. Bản Kaggle khắc phục một phần (nhiều dữ liệu hơn, có augmentation blur/JPEG).

---

## 6. Giới hạn

- Tiny-GenImage không có SD v1.4 nên dùng SD15 (cùng họ). Wukong cũng thuộc họ SD.
- Không dùng checkpoint đã huấn luyện của tác giả.
- Bài không công bố đủ chi tiết (β, số epoch…) nên các siêu tham số là tự chọn.
- Nhãn "loại ảnh" (động vật / đồ vật) trong hình UMAP là nhãn ước lượng bằng CLIP, không phải nhãn thật.
