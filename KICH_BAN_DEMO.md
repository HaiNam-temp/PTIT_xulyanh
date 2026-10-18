# Kịch bản trình bày demo VIB-Net (10–15 phút)

Kịch bản đi theo notebook **`demo/VIBNet_demo.ipynb`**. Notebook bản Kaggle có cùng các mục.

Mỗi bước gồm 3 phần:
- 🖥️ **Làm gì**: thao tác trên màn hình
- 🗣️ **Nói gì**: lời thoại gợi ý, có thể nói theo ý mình
- 💡 **Ý chính**: điều người nghe cần nhớ

---

## Chuẩn bị trước buổi demo (5 phút)
- [ ] Mở notebook, bấm **Run All** một lần trước. Lần đầu nạp CLIP chậm, chạy trước thì lúc trình bày sẽ nhanh.
- [ ] Mở sẵn bài báo PDF ở **Hình 3** (trang 4) để so sánh.
- [ ] Phóng to chữ notebook (Ctrl + `+`) để người ngồi cuối phòng đọc được.
- [ ] Nếu mất mạng: notebook đã lưu sẵn kết quả, chỉ cần cuộn xuống trình bày, không cần chạy lại.

---

## Bước 1 — Mở đầu: vấn đề là gì? (1–2 phút)

🖥️ Mục đầu của notebook (phần giới thiệu).

🗣️ *"Hiện nay AI vẽ ảnh rất giống thật. Đã có nhiều công cụ phát hiện ảnh giả, nhưng chúng có một điểm yếu: chỉ giỏi với loại AI đã được học. Ví dụ học trên Stable Diffusion thì gặp Midjourney hay GAN là bắt đầu đoán sai."*

🗣️ *"Bài báo CVPR 2025 này giải thích lý do: các công cụ dùng CLIP để 'nhìn' ảnh, nhưng CLIP chú ý quá nhiều vào **nội dung**, như con mèo hay cái xe, chứ không chú ý vào **dấu vết của AI**. Nên máy học mẹo sai, kiểu 'ảnh có cái cây là ảnh giả'."*

🗣️ *"Giải pháp của bài là thêm một 'nút thắt thông tin', gọi là VIB. Nó ép thông tin đi qua một cổ chai nhỏ, chỉ giữ phần cần để phân biệt thật/giả."*

💡 **Vấn đề = tổng quát hoá kém. Giải pháp = lọc bớt thông tin nội dung bằng VIB.**

---

## Bước 2 — Dữ liệu: "học một đề, thi đề khác" (1–2 phút)

🖥️ Mục 2: bảng số lượng ảnh và lưới ảnh mẫu.

🗣️ *"Em cho máy học 200 ảnh thật và 200 ảnh do Stable Diffusion 1.5 tạo. Sau đó em 'cho thi' bằng ảnh của 6 loại AI **chưa từng học**: ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong."*

🗣️ (chỉ vào lưới ảnh) *"Mọi người thử đoán xem ảnh nào thật, ảnh nào giả? Bằng mắt thường rất khó."*

💡 **Học trên 1 loại AI → kiểm tra trên 6 loại AI lạ. Đây đúng là tình huống thực tế.**

---

## Bước 3 — Mô hình hoạt động thế nào (2–3 phút)

🖥️ Mục 3–4: công thức loss và bảng so sánh với code tác giả.

🗣️ *"Quy trình gồm 3 bước: ảnh đi vào CLIP để lấy 768 con số mô tả ảnh. CLIP được giữ nguyên, không huấn luyện lại. Sau đó các con số đi qua một 'bộ não' nhỏ để quyết định thật hay giả."*

🗣️ *"Em so sánh 2 loại bộ não:*
- *Linear: chỉ một phép nhân đơn giản, đây là phương pháp cũ.*
- *VIB: nén 768 số xuống còn 256, có thêm nhiễu ngẫu nhiên, và bị phạt nếu giữ quá nhiều thông tin. Đây là phương pháp của bài."*

🗣️ *"Hàm mất mát có 2 phần: phần thứ nhất bắt phân loại đúng, phần thứ hai (β × KL) bắt nén thông tin. β càng lớn thì nén càng mạnh."*

🗣️ *"Bảng này cho thấy mô hình em viết **giống hệt** code của tác giả. Chỉ khác là tác giả không công bố code huấn luyện, nên phần KL em tự viết theo công thức trong bài."*

💡 **CLIP (đóng băng) → VIB (nén) → Thật/Giả. Loss = phân loại đúng + β × phạt giữ thông tin.**

---

## Bước 4 — Huấn luyện thử (1 phút)

🖥️ Mục 5: chạy cell, chỉ vào thời gian và biểu đồ loss.

🗣️ *"Huấn luyện chỉ mất vài giây, vì CLIP đã trích đặc trưng sẵn và chỉ huấn luyện bộ não nhỏ phía sau."*

🗣️ *"Trên ảnh cùng loại đã học (SD15), cả hai mô hình đều đạt khoảng 96%. Vậy là máy đã học tốt bài cũ."*

💡 **Bài đã học: cả hai đều giỏi. Câu hỏi thật sự là gặp đề lạ thì sao.**

---

## Bước 5 — Kết quả chính ⭐ (3–4 phút, phần quan trọng nhất)

🖥️ Mục 6: biểu đồ cột. Chỉ vào **đường gạch ngang 50%**.

🗣️ *"Đường gạch ngang là mức 50%, tức là **đoán bừa**, giống tung đồng xu."*

Đọc biểu đồ từ trái sang phải:

1. 🗣️ *"Cột SD15 đạt khoảng 95%: bài đã học thì làm tốt."*
2. 🗣️ *"Midjourney, Wukong đạt 85–92%: các AI này giống Stable Diffusion nên máy vẫn nhận ra."*
3. 🗣️ *"ADM, VQDM và nhất là BigGAN, một loại GAN khác hẳn, chỉ đạt khoảng 40–50%, tức là **đoán bừa**. Đây chính là vấn đề bài báo nói tới: gặp loại AI khác họ là máy thất bại."*
4. 🗣️ *"So sánh Linear (cột xanh dương) với VIB (các cột còn lại): VIB nhỉnh hơn khoảng 1–3 điểm. Nhưng thanh sai số (vạch đen) cũng cỡ đó, và β lớn hơn 0 không tốt hơn β = 0. Nên em **chưa thể khẳng định** VIB giúp được trong thí nghiệm nhỏ này."*

💡 **Tái hiện được vấn đề (đề lạ → đoán bừa). Chưa tái hiện được lời giải (VIB chỉ nhỉnh nhẹ).**

---

## Bước 6 — Hình UMAP (1–2 phút)

🖥️ Mục 7: hai hình chấm màu. Mở bài báo **Hình 3** bên cạnh.

🗣️ *"UMAP là cách 'trải' 768 chiều xuống mặt phẳng 2 chiều để nhìn bằng mắt. Mỗi chấm là một ảnh, màu xanh là động vật, màu đỏ là đồ vật, màu đậm là ảnh thật, màu nhạt là ảnh giả."*

🗣️ (hình trái) *"Đặc trưng CLIP gốc: thật và giả trộn lẫn vào nhau."*

🗣️ (hình phải) *"Sau khi qua VIB: tách thành 2 cụm thật và giả rất rõ, giống Hình 3 trong bài."*

🗣️ ⚠️ *"Nhưng em phải nói rõ: hình này vẽ trên chính dữ liệu đã dùng để huấn luyện, nên tách rõ là chuyện dễ hiểu. Nó minh hoạ ý tưởng của bài, chứ không chứng minh máy sẽ giỏi với ảnh lạ."*

💡 **Hình minh hoạ đúng ý tưởng, nhưng không phải bằng chứng.**

---

## Bước 7 — Thử từng ảnh 🎯 (1–2 phút, phần gây ấn tượng)

🖥️ Mục 8: 6 ảnh, mỗi ảnh kèm xác suất "giả".

🗣️ *"Giờ thử từng ảnh. Con số P(giả) càng gần 1 thì máy càng chắc là ảnh AI."*

🗣️ (chỉ ảnh Wukong) *"Ảnh Wukong: VIB cho 0.97, máy rất chắc đây là ảnh giả, và đúng."*

🗣️ (chỉ ảnh BigGAN) *"Ảnh BigGAN: chỉ 0.05, máy tưởng là ảnh thật, và **sai**. Đúng với kết quả ở biểu đồ: máy không nhận ra GAN."*

🖥️ *(Nếu có thời gian)* Đổi biến `paths` sang một ảnh khán giả đưa, rồi chạy lại cell.

💡 **Ví dụ cụ thể khớp với biểu đồ: giỏi với AI họ SD, bó tay với GAN.**

---

## Bước 8 — Kết luận (1 phút)

🖥️ Mục 9.

🗣️ *"Tóm lại, demo của em:*
- ✅ *Chạy lại được toàn bộ quy trình của bài: CLIP đóng băng + VIB.*
- ✅ *Thấy rõ vấn đề mà bài nêu: gặp loại AI khác họ là máy đoán bừa.*
- ❌ *Chưa đạt kết quả ~95% như bài, và chưa thấy rõ tác dụng của VIB."*

🗣️ *"Lý do: em dùng ít dữ liệu hơn bài khoảng 80 lần, chạy trên CPU, không dùng augmentation. Bản Kaggle của em dùng GPU, nhiều dữ liệu hơn và có augmentation như bài, để kiểm tra tiếp."*

💡 **Trung thực về kết quả. Hiểu được vì sao chưa đạt.**

---

## Câu hỏi có thể gặp và gợi ý trả lời

**❓ Tại sao không dùng code của tác giả để huấn luyện?**
→ Repo tác giả chỉ có code kiểm thử, không có hàm mất mát KL và không có vòng lặp huấn luyện. Em viết lại theo công thức (6) trong bài, kiến trúc giữ giống hệt code tác giả.

**❓ AP là gì? Sao không chỉ dùng độ chính xác?**
→ AP (Average Precision) đo khả năng xếp ảnh giả lên trên ảnh thật, **không phụ thuộc ngưỡng** 0.5. Độ chính xác (ACC) thì phụ thuộc ngưỡng. Bài báo dùng cả hai, em cũng vậy.

**❓ Tại sao 50% là đoán bừa?**
→ Tập kiểm thử có số ảnh thật bằng số ảnh giả (1:1). Tung đồng xu thì đúng 50%.

**❓ β là gì?**
→ Mức độ "nén". β = 0 là không phạt, mô hình giữ bao nhiêu thông tin cũng được. β lớn thì bị ép bỏ nhiều thông tin. Bài không công bố β nên em thử 5 giá trị.

**❓ Sao không thấy VIB tốt hơn rõ rệt như bài?**
→ Dữ liệu quá ít (400 ảnh so với 32.000 ảnh), không có augmentation (bài nói rất quan trọng), và mỗi kết quả có sai số ±1–3 điểm, lớn ngang mức chênh lệch.

**❓ CLIP là gì?**
→ Mô hình của OpenAI, học từ 400 triệu cặp ảnh và câu mô tả. Nó biến mỗi ảnh thành một vector số mô tả nội dung ảnh.
