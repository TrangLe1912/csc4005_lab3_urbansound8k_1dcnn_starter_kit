# Rubric chấm điểm CSC4005 Lab 3  
## UrbanSound8K Classification with 1D-CNN

Lab này đánh giá khả năng xử lý dữ liệu âm thanh môi trường và xây dựng mô hình **1D-CNN** cho bài toán phân loại âm thanh trên **UrbanSound8K**. Trọng tâm chính là pipeline ổn định với **MFCC/log-mel theo thời gian**, không yêu cầu raw waveform là baseline bắt buộc.



---

## Quy ước chấm chung

- Tổng điểm: **10 điểm**.
- Sinh viên phải nộp đúng hạn theo yêu cầu trên LMS/GitHub Classroom.
- Mọi kết quả thực nghiệm cần có **bằng chứng tái lập**: mã nguồn, cấu hình chạy, output, hình ảnh learning curves/confusion matrix, và link W&B.
- Điểm mô hình không chỉ dựa vào accuracy tuyệt đối. Giảng viên ưu tiên đánh giá: pipeline đúng, thí nghiệm có kiểm soát, phân tích lỗi có lý, và báo cáo có khả năng giải thích.
- Nếu không chạy được toàn bộ lab, sinh viên vẫn có thể được chấm phần đã hoàn thành nếu trình bày rõ lỗi, log lỗi, hướng xử lý, và phần nào đã kiểm chứng.

## Mức đánh giá chung

| Mức | Ý nghĩa |
|---|---|
| Xuất sắc | Hoàn thành đúng yêu cầu, có phân tích sâu, minh chứng rõ, kết quả tái lập tốt |
| Đạt tốt | Hoàn thành phần lớn yêu cầu, có kết quả và phân tích hợp lý |
| Đạt tối thiểu | Chạy được một phần chính, còn thiếu phân tích hoặc thiếu minh chứng |
| Chưa đạt | Không chạy được pipeline chính, thiếu file quan trọng, hoặc không có bằng chứng thực nghiệm |


## Bảng tiêu chí chấm điểm

| Tiêu chí | Điểm | Xuất sắc | Đạt tốt | Đạt tối thiểu | Chưa đạt |
|---|---:|---|---|---|---|
| 1. Cấu trúc repo và khả năng chạy lại | 1.0 | Repo đúng cấu trúc starter kit, có config, docs, output; chạy lại được bằng lệnh trong README; không đưa UrbanSound8K vào repo | Repo đúng phần lớn cấu trúc, chạy lại được sau chỉnh sửa nhỏ | Có file chính nhưng thiếu docs/config hoặc đường dẫn chưa rõ | Repo thiếu nhiều file, không thể chạy lại |
| 2. Chuẩn bị dữ liệu UrbanSound8K | 1.5 | Đọc đúng metadata, map đúng classID/class name, xử lý fold hợp lý, kiểm tra số lượng mẫu/lớp, không rò rỉ dữ liệu | Đọc được dữ liệu và chia train/val/test hợp lý | Đọc được dữ liệu nhưng chưa kiểm tra fold/class distribution | Không đọc được dữ liệu hoặc gán nhãn sai |
| 3. Trích xuất đặc trưng audio | 2.0 | Trích xuất MFCC/log-mel ổn định; xử lý sampling rate, mono, duration, padding/truncation; chuẩn hóa feature; giải thích được vì sao dùng đặc trưng này cho baseline | Trích xuất feature đúng và train được | Có feature nhưng xử lý duration/normalization còn sơ sài | Không trích xuất được feature hoặc feature sai định dạng |
| 4. Mô hình 1D-CNN | 2.0 | Xây dựng 1D-CNN hợp lý cho chuỗi đặc trưng; có BatchNorm/Dropout/Pooling; giải thích được kernel, channel, pooling theo thời gian | Train được 1D-CNN và có kết quả hợp lý | Có model nhưng cấu hình yếu, dễ overfitting hoặc thiếu giải thích | Không train được 1D-CNN |
| 5. W&B logging | 1.5 | Có dashboard W&B đầy đủ: config, metric theo epoch, learning curves, confusion matrix, feature mode, model params, link rõ ràng | Có log W&B các metric chính | Có dùng W&B nhưng thiếu metric hoặc thiếu link rõ ràng | Không dùng W&B |
| 6. Đánh giá và phân tích lỗi | 1.0 | Có learning curves, confusion matrix, classification report; phân tích lớp dễ nhầm như siren/car horn, drilling/jackhammer; đề xuất cải thiện hợp lý | Có biểu đồ và nhận xét kết quả | Có kết quả nhưng phân tích còn chung chung | Không có phân tích |
| 7. Bài mở rộng raw waveform | 0.5 | Có thử raw waveform 1D-CNN, so sánh với MFCC/log-mel và nêu vì sao khó hơn | Có chạy thử raw waveform nhưng phân tích ít | Có nêu ý tưởng nhưng chưa chạy được | Không đề cập |
| 8. Báo cáo và câu hỏi tự kiểm tra | 0.5 | Báo cáo rõ ràng, có minh chứng, trả lời câu hỏi bằng lập luận riêng | Báo cáo đủ ý chính | Báo cáo thiếu minh chứng hoặc câu trả lời sơ sài | Không nộp báo cáo |

## Checklist bằng chứng cần nộp

- [ ] Link GitHub repository hoặc GitHub Classroom submission.
- [ ] Link W&B project/run.
- [ ] Config đã dùng cho baseline MFCC/log-mel + 1D-CNN.
- [ ] File `metrics.json`, `history.csv`, `curves.png`, `confusion_matrix.png`.
- [ ] Báo cáo có bảng thông tin: feature mode, sampling rate, duration, số MFCC/mel bins, model params.
- [ ] Nhận xét ít nhất 2 cặp lớp dễ nhầm.
- [ ] Nếu làm mở rộng raw waveform: có run riêng và so sánh với baseline.

## Gợi ý trừ điểm

| Lỗi | Mức trừ gợi ý |
|---|---:|
| Đưa thư mục UrbanSound8K hoặc audio gốc lên GitHub | -1.0 đến -2.0 |
| Không dùng fold/metadata đúng của UrbanSound8K | -0.5 đến -1.5 |
| Không xử lý độ dài audio thống nhất | -0.5 đến -1.0 |
| Không có W&B link | -1.0 đến -1.5 |
| Chỉ báo accuracy, không có confusion matrix | -0.5 đến -1.0 |
| Nhầm 1D-CNN trên chuỗi đặc trưng với 2D-CNN trên ảnh spectrogram mà không giải thích | -0.5 đến -1.0 |

## Điểm cộng khuyến khích

| Nội dung | Điểm cộng tối đa |
|---|---:|
| Có so sánh MFCC và log-mel trên cùng split | +0.5 |
| Có audio augmentation đơn giản: noise, time shift, gain | +0.3 |
| Có phân tích mẫu dự đoán sai kèm tên file audio | +0.3 |
| Có cải thiện script để cache feature giúp chạy nhanh hơn | +0.2 |
