# CSC4005 Lab 3 – UrbanSound8K với 1D-CNN

## 1. Bối cảnh

Trong các lab trước, sinh viên đã làm quen với MLP/CNN trên dữ liệu dạng bảng hoặc ảnh. Lab này chuyển sang dữ liệu âm thanh. Thay vì xem audio là một file khó xử lý, ta đưa audio về chuỗi đặc trưng theo thời gian, sau đó dùng 1D-CNN để học các pattern cục bộ.

## 2. Mục tiêu học tập

Sau lab này, sinh viên có thể:

1. đọc metadata và file âm thanh của UrbanSound8K,
2. chuẩn hóa audio về cùng sample rate và cùng độ dài,
3. trích xuất MFCC hoặc log-mel spectrogram,
4. xây dựng mô hình 1D-CNN cho phân loại âm thanh,
5. dùng W&B để log quá trình train,
6. phân tích confusion matrix để nhận diện lỗi mô hình.

## 3. Luồng xử lý chính

```text
UrbanSound8K wav files
→ resample + mono + pad/crop
→ MFCC sequence
→ 1D-CNN
→ classifier 10 lớp
→ W&B + confusion matrix
```

## 4. Vì sao cấu hình chính dùng MFCC?

Raw waveform có số chiều lớn và mô hình cần học cả đặc trưng tần số từ đầu. Điều này thường khó hơn với laptop cá nhân và thời lượng lab ngắn.

MFCC/log-mel là biểu diễn đã tóm tắt thông tin phổ âm thanh theo thời gian. Vì vậy, mô hình 1D-CNN nhỏ vẫn có thể học được pattern cục bộ ổn định hơn.

## 5. Nhiệm vụ bắt buộc

### Task 1 – Setup

```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
wandb login
```

### Task 2 – Chạy debug

```bash
python -m src.train \
  --config configs/fast_debug.json \
  --data_dir /duong_dan/UrbanSound8K
```

### Task 3 – Chạy baseline chính

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --run_name mssv_mfcc_1dcnn_baseline
```

### Task 4 – Phân tích kết quả

Mở thư mục:

```text
outputs/<run_name>/
```

và phân tích:

- `curves.png`,
- `confusion_matrix.png`,
- `metrics.json`,
- W&B dashboard.

## 6. Câu hỏi gợi ý cho báo cáo

1. Input của 1D-CNN có shape như thế nào?
2. Chiều thời gian nằm ở đâu trong tensor?
3. Conv1D học pattern cục bộ theo chiều nào?
4. Lớp âm thanh nào dễ bị nhầm nhất?
5. Có dấu hiệu overfitting không?
6. Nếu tăng số epoch hoặc tăng dữ liệu, kết quả có chắc chắn tốt hơn không? Vì sao?

## 7. Bài mở rộng

### Mở rộng A – log-mel + 1D-CNN

Chạy lại baseline nhưng đổi feature:

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --feature_type logmel \
  --run_name mssv_logmel_1dcnn
```

### Mở rộng B – raw waveform + 1D-CNN

```bash
python -m src.train \
  --config configs/extension_raw_waveform.json \
  --data_dir /duong_dan/UrbanSound8K \
  --run_name mssv_raw_waveform_extension
```

Yêu cầu khi làm mở rộng:

- vẫn phải log W&B,
- so sánh với baseline MFCC,
- không chỉ báo accuracy mà phải giải thích nguyên nhân khác biệt.
