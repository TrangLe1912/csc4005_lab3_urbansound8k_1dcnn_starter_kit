# CSC4005 Lab 3 Report – UrbanSound8K with 1D-CNN

## 1. Thông tin sinh viên

- Họ tên:
- Mã sinh viên:
- Lớp:
- Link GitHub repo:
- Link W&B run/project:

---

## 2. Mục tiêu thí nghiệm

Mô tả ngắn gọn mục tiêu của lab:

- phân loại âm thanh môi trường trên UrbanSound8K,
- sử dụng MFCC/log-mel làm chuỗi đặc trưng theo thời gian,
- xây dựng và huấn luyện 1D-CNN,
- theo dõi thí nghiệm bằng W&B,
- phân tích lỗi bằng confusion matrix.

---

## 3. Dữ liệu và tiền xử lý

### 3.1. Dataset

- Dataset:
- Số lớp:
- Các lớp:
- Fold dùng để train:
- Fold dùng để validation:
- Fold dùng để test:

### 3.2. Tiền xử lý audio

Điền cấu hình đã dùng:

| Thành phần | Giá trị |
|---|---|
| Sample rate |  |
| Duration |  |
| Feature type |  |
| n_mfcc / n_mels |  |
| n_fft |  |
| hop_length |  |
| Augmentation |  |

Giải thích ngắn: vì sao cần đưa audio về cùng sample rate và cùng độ dài?

---

## 4. Mô hình 1D-CNN

Mô tả kiến trúc mô hình:

```text
Input feature sequence
→ Conv1D block 1
→ Conv1D block 2
→ Conv1D block 3
→ Global Average Pooling
→ Dense classifier
→ Softmax
```

Bảng cấu hình:

| Thành phần | Giá trị |
|---|---|
| model_name |  |
| hidden_channels |  |
| dropout |  |
| optimizer |  |
| learning rate |  |
| weight decay |  |
| batch size |  |
| epochs |  |
| patience |  |

---

## 5. Kết quả thực nghiệm

### 5.1. Kết quả chính

| Metric | Giá trị |
|---|---:|
| Best validation accuracy |  |
| Test accuracy |  |
| Average epoch time |  |
| Total parameters |  |
| Trainable parameters |  |

### 5.2. Learning curves

Chèn hình `curves.png`.

Nhận xét:

- Train loss/val loss có giảm đều không?
- Có dấu hiệu overfitting không?
- Early stopping có xảy ra không?

### 5.3. Confusion matrix

Chèn hình `confusion_matrix.png`.

Nhận xét:

- Những lớp nào dễ phân loại?
- Những lớp nào dễ bị nhầm?
- Có thể do đặc trưng âm thanh, độ dài clip, nhiễu nền, hay mất cân bằng dữ liệu?

---

## 6. W&B tracking

Dán link W&B:

```text
https://wandb.ai/...
```

Ảnh chụp hoặc mô tả dashboard cần có:

- learning curves,
- final metrics,
- configuration,
- confusion matrix image.

---

## 7. Phân tích và thảo luận

Trả lời ngắn các câu hỏi:

1. Vì sao dùng 1D-CNN thay vì MLP cho chuỗi đặc trưng audio?
2. Kernel 1D trong bài này đang trượt theo chiều nào?
3. MFCC giúp mô hình học dễ hơn raw waveform ở điểm nào?
4. Mô hình hiện tại còn hạn chế gì?
5. Có thể cải thiện kết quả bằng cách nào?

---

## 8. Bài mở rộng nếu có

Nếu làm raw waveform hoặc log-mel, điền bảng sau:

| Pipeline | Feature/Input | Test accuracy | Nhận xét |
|---|---|---:|---|
| Baseline | MFCC + 1D-CNN |  |  |
| Extension 1 | log-mel + 1D-CNN |  |  |
| Extension 2 | raw waveform + 1D-CNN |  |  |

---

## 9. Kết luận

Tóm tắt 3–5 ý chính học được từ lab.
