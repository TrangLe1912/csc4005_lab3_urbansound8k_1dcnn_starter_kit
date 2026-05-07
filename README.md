# CSC4005 Lab 3 – Environmental Sound Classification with 1D-CNN


Case study: **UrbanSound8K** – phân loại 10 loại âm thanh môi trường.

Trọng tâm của lab:
- tiền xử lý audio về cùng sample rate và độ dài,
- trích xuất đặc trưng **MFCC/log-mel theo thời gian**,
- huấn luyện **1D-CNN** trên chuỗi đặc trưng,
- log thí nghiệm bằng **Weights & Biases (W&B)**,
- phân tích learning curves và confusion matrix.

Phần **raw waveform 1D-CNN** đã được chuẩn bị trong repo nhưng được đặt là **bài mở rộng**, không phải cấu hình chính trên lớp.

---

## 1. Cấu trúc repo

```text
csc4005_lab3_urbansound8k_1dcnn_starter/
├── README.md
├── REPORT_TEMPLATE.md
├── requirements.txt
├── configs/
│   ├── baseline_mfcc_1dcnn.json
│   ├── fast_debug.json
│   └── extension_raw_waveform.json
├── docs/
│   ├── RUBRIC.md
│   ├── WANDB_GUIDE.md
│   └── LAB_GUIDE_LAB3.md
├── notebooks/
│   └── lab3_demo.ipynb
├── outputs/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── ci/
    ├── check_structure.py
    └── smoke_train.py
```

---

## 2. Cài đặt môi trường

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra lệnh chạy:

```bash
python -m src.train --help
```

---

## 3. Dữ liệu UrbanSound8K

Repo này **không chứa dữ liệu**. Sinh viên cần tải UrbanSound8K và truyền đường dẫn qua `--data_dir`.

Cấu trúc dữ liệu được hỗ trợ:

```text
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   └── ...
└── metadata/
    └── UrbanSound8K.csv
```

Có thể truyền vào:

```bash
--data_dir /duong_dan/UrbanSound8K
```

hoặc file zip:

```bash
--data_dir /duong_dan/UrbanSound8K.zip
```

Starter kit sẽ tự tìm file `UrbanSound8K.csv` và các file `.wav` tương ứng trong thư mục `audio/fold*/`.

---

## 4. Cấu hình chính khuyến nghị cho lớp học

Cấu hình chính dùng **MFCC + 1D-CNN** để lab chạy ổn trên CPU/laptop cá nhân:

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir /duong_dan/UrbanSound8K
```

Cấu hình này dùng:

- sample rate: `16000 Hz`
- duration: `4.0 giây`
- feature: `MFCC`, `n_mfcc=40`
- split theo fold:
  - train: fold 1–8
  - validation: fold 9
  - test: fold 10
- giới hạn mẫu theo lớp để lab chạy ổn:
  - `max_train_per_class=120`
  - `max_eval_per_class=50`
- W&B: bật mặc định

---

## 5. Chạy nhanh để kiểm tra pipeline

Khi mới setup môi trường, nên chạy cấu hình debug trước:

```bash
python -m src.train \
  --config configs/fast_debug.json \
  --data_dir /duong_dan/UrbanSound8K
```

Cấu hình này chỉ dùng một phần nhỏ dữ liệu để kiểm tra:

- code đọc được dữ liệu,
- feature cache hoạt động,
- model train được,
- W&B log được.

---

## 6. W&B là yêu cầu bắt buộc của lab

Trước khi chạy train chính:

```bash
wandb login
```

Khi chạy xong, sinh viên cần nộp link W&B run hoặc project dashboard trong báo cáo.

Có thể chạy offline khi mạng yếu:

```bash
python -m src.train \
  --config configs/baseline_mfcc_1dcnn.json \
  --data_dir /duong_dan/UrbanSound8K \
  --wandb_mode offline
```

Sau đó đồng bộ lại:

```bash
wandb sync wandb/offline-run-...
```

---

## 7. Output sau khi train

Mỗi run tạo thư mục:

```text
outputs/<run_name>/
```

bao gồm:

- `best_model.pt`
- `history.csv`
- `curves.png`
- `confusion_matrix.png`
- `metrics.json`
- `used_config.json`

W&B cần log tối thiểu:

- `train_loss`
- `val_loss`
- `train_acc`
- `val_acc`
- `lr`
- `epoch_time_sec`
- `best_val_acc`
- `test_acc`
- `confusion_matrix_image`
- `curves_image`

---

## 8. Bài mở rộng: 1D-CNN trực tiếp trên raw waveform

Phần raw waveform đã có cấu hình riêng:

```bash
python -m src.train \
  --config configs/extension_raw_waveform.json \
  --data_dir /duong_dan/UrbanSound8K
```

Sinh viên khá/giỏi có thể so sánh:

| Pipeline | Biểu diễn đầu vào | Mục tiêu |
|---|---|---|
| MFCC + 1D-CNN | chuỗi đặc trưng MFCC | cấu hình chính, ổn định |
| log-mel + 1D-CNN | chuỗi đặc trưng phổ mel | biến thể để thử nghiệm |
| raw waveform + 1D-CNN | tín hiệu âm thanh gốc | bài mở rộng, khó hơn |

Không yêu cầu raw waveform phải tốt hơn MFCC. Điều quan trọng là sinh viên giải thích được vì sao kết quả khác nhau.

---

## 9. Checklist nộp bài

- [ ] Chạy được ít nhất 1 run `MFCC + 1D-CNN`
- [ ] Có link W&B dashboard/run
- [ ] Có `curves.png`
- [ ] Có `confusion_matrix.png`
- [ ] Có bảng metric: train/val/test accuracy
- [ ] Có nhận xét lớp nào dễ nhầm lẫn
- [ ] Có giải thích ngắn: vì sao 1D-CNN phù hợp với chuỗi đặc trưng audio
- [ ] Bài mở rộng, nếu làm: so sánh thêm raw waveform hoặc log-mel

---

## 10. Lệnh kiểm tra CI local

```bash
python ci/check_structure.py
python ci/smoke_train.py
```

`smoke_train.py` tự tạo dữ liệu audio giả lập nhỏ theo format UrbanSound8K để kiểm tra pipeline, không cần tải dataset thật.
