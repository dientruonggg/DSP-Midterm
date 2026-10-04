# Ghi chú hiểu bài và biện luận khi trình bày

## Câu mở đầu ngắn

“Input của hệ thống là tín hiệu tiếng nói theo thời gian. Output là hai biên bắt đầu và kết thúc vùng speech. Nhóm biến tín hiệu dài thành các frame 20 ms, tính đặc trưng cho từng frame, phân lớp speech/silence, rồi dùng điều kiện khoảng lặng tối thiểu 200 ms để làm sạch output.”

## Vì sao dùng STE?

- Âm hữu thanh có dao động tuần hoàn và thường mang năng lượng lớn.
- Khoảng lặng có năng lượng nhỏ.
- Vì mức âm của mỗi file khác nhau nên dùng `STE / max(STE)` để đưa về cùng thang đo.
- Điểm yếu: các phụ âm vô thanh như /s/, /f/, /t/ có thể yếu gần giống nhiễu nền.

## Vì sao thêm FIR `h[n] = [1, -0.97]`?

Hệ lọc cho `y[n] = x[n] - 0.97x[n-1]`. Nó làm giảm thành phần biến thiên chậm/tần số thấp và nhấn mạnh thay đổi nhanh/tần số cao. Vì âm vô thanh thường giàu thành phần cao tần, đặc trưng sau lọc giúp phân biệt nó với khoảng lặng tốt hơn. Đây là pre-emphasis, không phải “lọc sạch mọi nhiễu”.

## Vì sao dùng ZCR và tỉ lệ năng lượng cao tần?

- Voiced: năng lượng lớn, F0 rõ, ZCR thường thấp hơn.
- Unvoiced: năng lượng yếu hơn, không có F0 ổn định, ZCR và năng lượng cao tần thường lớn hơn.
- Silence/noise: năng lượng thấp; ZCR có thể cao nên vẫn cần energy floor.

## Giải thích từng task

### TT1

Baseline dùng một ngưỡng STE chung, tìm bằng binary search trên training. Bản mở rộng dùng bốn đặc trưng và logistic loss. Đây là bản tốt nhất về biên: MAE trung bình 6.25 ms.

### TT2

Baseline đếm histogram một chiều của STE. Bản mở rộng dùng `H[energy, ZCR]`: mỗi ô biểu diễn một kiểu frame. Nó giữ âm vô thanh tốt hơn (UV recall 0.891 → 0.947), nhưng biên ngoài lệch thêm vài hop nên MAE 17.50 → 20.00 ms.

### TT3

Baseline giả sử STE của speech và silence theo hai Gaussian một chiều. Bản mở rộng dùng Gaussian đa biến. Kết quả đa biến không tốt hơn vì chỉ có bốn file train, trong khi covariance có nhiều tham số. Đây là minh chứng “model phức tạp hơn chưa chắc tốt hơn”.

## Nếu thầy hỏi “tại sao không tối ưu tiếp trên test?”

Không được chọn tham số theo bốn file test vì như vậy số đo không còn phản ánh khả năng tổng quát. Nhóm giữ riêng `studio_M1` để validation, chốt cách làm, sau đó refit bằng toàn bộ bốn file training và chỉ cuối cùng mới đo test.

## Nếu thầy hỏi “F0 có tham gia quyết định không?”

Không. F0 được ước lượng bằng tự tương quan và chỉ vẽ để giải thích vùng voiced. Biên VAD không dùng nhãn test hoặc F0 ground truth. Điều này tránh cho hệ thống phụ thuộc vào pitch, vì âm vô thanh vốn không có F0 ổn định nhưng vẫn phải được giữ là speech.

## Nếu thầy hỏi “vì sao cần 4 metric?”

- MAE: sai số trung bình của hai biên, đúng trọng tâm đề.
- RMSE: phạt mạnh lỗi biên lớn.
- Frame F1: kiểm tra toàn vùng speech, tránh chỉ nhìn hai biên ngoài.
- UV recall: kiểm tra riêng failure mode của energy-only VAD.

## Cách nói về hình

- Màu xám: waveform input.
- Màu cam: normalized STE.
- Màu xanh lá ở variant 2: decision score đã chuẩn hóa để minh họa.
- Đỏ nét đứt: ground truth từ LAB.
- Xanh dương chấm: biên output của thuật toán.
- Tím: F0 ước lượng; đoạn vô thanh/khoảng lặng thường bị đứt vì không có tuần hoàn ổn định.

## Ba câu kết nên nhớ

1. “The best boundary result is TT1-2 with 6.25 ms mean MAE.”
2. “The joint histogram preserves more unvoiced frames, but slightly shifts the outer boundaries.”
3. “With only four training recordings, the simple Gaussian is more reliable than full covariance.”

## Slide thuyết trình & Kịch bản 3 phút

- **Link Slide Canva của nhóm:** [Canva Slide VAD Midterm](https://canva.link/jhh1gex7ybd87yk)

### Phân bổ cấu trúc 4 slide & căn giờ cho từng thành viên (3 phút):

| Slide | Nội dung hiển thị | Lời thoại mẫu (Script thuyết trình) | Thời lượng |
|---|---|---|:---:|
| **Slide 1: Cover** | Tiêu đề nhiệm vụ (TT1/TT2/TT3), Tên SV, MSSV | *“Kính thưa Thầy, em tên là [Tên], hôm nay em xin báo cáo nhiệm vụ [TT1/TT2/TT3] trong đề tài Phân đoạn tiếng nói và khoảng lặng.”* | **10 giây** |
| **Slide 2: Giải pháp lõi (Tìm ngưỡng)** | Chèn **1 ảnh duy nhất** về giải pháp trên tập Train (Ảnh Histogram cho TT2, Ảnh phân bố Gauss `gaussian_distribution_slide.png` cho TT3) và giá trị ngưỡng $T$ | *“Để tìm ngưỡng dùng chung cho 4 file huấn luyện, em phân tích phân bố đặc trưng của khoảng lặng và tiếng nói. Điểm cắt tối ưu đạt được tại $T = [giá\_trị]$. Đồ thị cho thấy vùng phân tách rõ rệt giữa hai trạng thái.”* | **45 - 50 giây** |
| **Slide 3: Kết quả thực nghiệm** | • Bảng số liệu MAE 4 file kiểm thử<br>• 1-2 figure đẹp nhất (Waveform, STE, vạch đỏ Ground truth, vạch xanh Predicted) | *“Áp dụng trên 4 file kiểm thử, thuật toán đạt sai số biên trung bình MAE là [X] ms. Ở môi trường studio ít nhiễu sai số chỉ từ 5-10 ms; file phone_F2 sai số lớn hơn một chút do năng lượng đuôi âm vô thanh giảm dần sát mức nhiễu nền.”* | **60 - 70 giây** |
| **Slide 4: Kết luận & Đánh giá** | 3 gạch đầu dòng ngắn về ưu/nhược điểm chính của phương pháp | *“Tóm lại, thuật toán có ưu điểm là [tính thích nghi / tự động học tham số / chạy nhanh]. Hạn chế là [nhạy với ngưỡng / giả thiết xấp xỉ]. Em xin kết thúc phần slide và chuyển sang 1 phút demo chương trình.”* | **20 giây** |

⏱️ **Tổng thời gian nói:** Khoảng 2 phút 20 giây - 2 phút 30 giây (an toàn tuyệt đối, không lo bị cắt chuông 3 phút).


