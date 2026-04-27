# Hướng dẫn W&B cho CSC4005 Lab 3

W&B là yêu cầu bắt buộc trong Lab 3. Sinh viên cần dùng W&B để theo dõi quá trình train và nộp link run/project trong báo cáo.

## 1. Đăng nhập

```bash
wandb login
```

Dán API key từ tài khoản W&B khi terminal yêu cầu.

## 2. Chạy baseline có W&B

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir /duong_dan/UrbanSound8K
```

Trong config, các trường mặc định là:

```json
{
  "use_wandb": true,
  "wandb_mode": "online",
  "project": "csc4005-lab3-urbansound-1dcnn"
}
```

## 3. Khi mạng yếu: chạy offline

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --wandb_mode offline
```

Sau khi có mạng:

```bash
wandb sync wandb/offline-run-...
```

## 4. Các thông tin cần có trên W&B

Mỗi run nên có:

- config thí nghiệm,
- train_loss, val_loss,
- train_acc, val_acc,
- learning rate,
- epoch time,
- best_val_acc,
- test_acc,
- curves image,
- confusion matrix image.

## 5. Cách đặt tên run

Gợi ý:

```text
mssv_mfcc_1dcnn_baseline
mssv_logmel_1dcnn
mssv_raw_waveform_extension
```

Ví dụ:

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --run_name 23123456_mfcc_1dcnn_baseline
```
