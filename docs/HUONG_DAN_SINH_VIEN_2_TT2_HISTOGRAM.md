# HƯỚNG DẪN CHI TIẾT CÔNG VIỆC: SINH VIÊN 2 (STUDENT 2)
## THUẬT TOÁN 2 (TT2): NGƯỠNG TỰ THÍCH NGHI THEO HISTOGRAM (GIANNAKOPOULOS 2014)

---

## 1. Thông Tin Vai Trò & Trách Nhiệm
* **Thành viên phụ trách**: Sinh viên 2.
* **Tên thuật toán**: Phân đoạn Tiếng nói / Khoảng lặng bằng Ngưỡng Tự Thích Nghi dựa trên Lược Đồ Tần Suất (*Histogram-based Adaptive Energy Thresholding*).
* **Tài liệu tham khảo**: Giannakopoulos (2014), *A method for silence removal and segmentation of speech signals*.
* **Tệp mã nguồn chịu trách nhiệm**:
  - `algorithms/tt2_histogram.py`
  - Đóng góp kiểm thử tích hợp: `main.py --algo tt2`
* **Thời lượng bảo vệ**: Đúng **3 phút trình bày Slide + 1 phút chạy Demo trực tiếp**.

---

## 2. Bản Chất Toán Học & Cơ Sở Lý Thuyết

### 2.1 Nguyên lý cốt lõi
Khác với TT1 dùng một ngưỡng chung, TT2 cho rằng **mỗi tín hiệu thu âm có một sàn nhiễu và biên độ giọng nói riêng biệt**, do đó ngưỡng $T$ phải được tính toán động (adaptive) cho từng file dựa vào hình dáng phân bố mật độ xác suất của năng lượng ngắn hạn $STE_{norm}$.

### 2.2 Quy trình 5 bước tính ngưỡng tự thích nghi

1. **Bước 1: Trích xuất chuỗi năng lượng $STE_{norm}$**:
   - Tín hiệu được phân khung $20\text{ ms}$, bước nhảy $10\text{ ms}$, tính $STE$ và chuẩn hóa về $[0.0, 1.0]$.
2. **Bước 2: Xây dựng Histogram 100 bins**:
   - Chia đoạn $[0.0, 1.0]$ thành 100 khoảng (bins) bằng nhau (độ rộng mỗi bin là $0.01$).
   - Đếm tần suất số khung rơi vào từng bin.
3. **Bước 3: Làm mịn Histogram bằng Moving Average (MA)**:
   - Do dữ liệu mẫu có thể tạo ra các răng cưa ngẫu nhiên, ta dùng bộ lọc trung bình trượt với kích thước cửa sổ 5 bins:
     $$
     H_{smooth}[i] = \frac{1}{2k+1} \sum_{j=-k}^k H[i+j] \quad (\text{với } k = 2, \text{ cửa sổ } 5 \text{ bins})
     $$
   - *Lưu ý*: Phải tự lập trình bằng phép cộng ma trận cơ bản, CẤM dùng thư viện ngoài (`scipy.signal`).
4. **Bước 4: Xác định 2 đỉnh cực đại địa phương ($M_1$ và $M_2$)**:
   - Điểm cực đại địa phương là điểm có giá trị lớn hơn 2 điểm lân cận: $H_{smooth}[i] > H_{smooth}[i-1]$ và $H_{smooth}[i] > H_{smooth}[i+1]$.
   - $M_1$: Đỉnh cực đại đầu tiên (ở vùng năng lượng thấp gần 0), đại diện cho **khoảng lặng / nhiễu nền**.
   - $M_2$: Đỉnh cực đại thứ hai (ở vùng năng lượng cao hơn), đại diện cho **tiếng nói**.
5. **Bước 5: Tính ngưỡng phân tách thích nghi**:
   $$
   T = \frac{W \cdot M_1 + M_2}{W + 1} \quad (\text{với trọng số } W = 5)
   $$
   - Trọng số $W = 5$ kéo ngưỡng thiên về phía $M_1$ để tránh việc ngưỡng bị đẩy lên quá cao do các nguyên âm mạnh tạo nên.

---

## 3. Bản Đặc Tả Giao Diện & Mã Nguồn Cần Hoàn Thiện

Tệp `algorithms/tt2_histogram.py` đã được thiết kế sẵn đầy đủ các giao diện chuẩn (interface) cùng comment `# //TODO` và lệnh in định danh hàm:

### 3.1 Hàm `compute_histogram_100bins`
```python
def compute_histogram_100bins(
    ste_norm: np.ndarray,
    num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    # //TODO: Construct 100-bin histogram for STE_norm values in range [0, 1] using basic array math
    print("compute_histogram_100bins")
    ...
```
* **Nhiệm vụ**: Chia mảng $[0, 1]$ thành 100 bins, đếm tần số xuất hiện của các giá trị $STE_{norm}$ và trả về `(counts, bin_centers)`.

### 3.2 Hàm `moving_average_smooth`
```python
def moving_average_smooth(
    histogram: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    # //TODO: Smooth 1D array using basic moving average window of size 5 with edge padding
    print("moving_average_smooth")
    ...
```
* **Nhiệm vụ**: Lọc làm mịn histogram để loại bỏ các đỉnh nhiễu giả mạo mà không dùng `scipy.signal.convolve`.

### 3.3 Hàm `find_histogram_peaks`
```python
def find_histogram_peaks(
    smoothed_hist: np.ndarray,
    bin_centers: np.ndarray
) -> Tuple[float, float]:
    # //TODO: Find first local peak M1 (silence) and second significant local peak M2 (speech)
    print("find_histogram_peaks")
    ...
```
* **Nhiệm vụ**: Quét mảng tìm $M_1$ (đỉnh nhiễu nền) và $M_2$ (đỉnh năng lượng tiếng nói).

### 3.4 Hàm `compute_adaptive_threshold_tt2`
```python
def compute_adaptive_threshold_tt2(
    ste_norm: np.ndarray,
    weight_w: float = 5.0
) -> float:
    # //TODO: Extract histogram, smooth with MA filter, find M1 & M2, compute T = (W*M1 + M2)/(W+1)
    print("compute_adaptive_threshold_tt2")
    ...
```
* **Nhiệm vụ**: Tích hợp các bước trên để tính ra ngưỡng $T$ đơn lẻ cho tệp âm thanh.

### 3.5 Hàm `predict_vad_tt2`
```python
def predict_vad_tt2(
    signal: np.ndarray,
    sample_rate: int,
    weight_w: float = 5.0
) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    # //TODO: Run end-to-end adaptive histogram VAD, return boundaries, features, and threshold
    print("predict_vad_tt2")
    ...
```
* **Nhiệm vụ**: Thực hiện trọn gói quy trình phân đoạn VAD thích nghi, trả về `(t_start, t_end, ste_norm, frame_times, threshold)`.

---

## 4. Bảng Số Liệu Kiểm Thử Mục Tiêu Trên Tập Kiểm Thử (TinHieuKiemThu)

Sinh viên 2 chạy lệnh sau trong terminal để kiểm tra kết quả:
```bash
uv run python main.py --algo tt2 --no-plot
```

Bảng số liệu thực nghiệm mục tiêu cho TT2:

| Tệp Kiểm Thử | Ground-Truth | Ngưỡng $T$ tính được | TT2 Dự đoán | Sai lệch $\Delta(T_{start}, T_{end})$ | MAE (ms) | RMSE (ms) |
|---|---|---|---|---|---|---|
| `phone_F2.wav` | $[1.02\text{ s}, 4.04\text{ s}]$ | $T \approx 0.082$ | $[1.11\text{ s}, 4.01\text{ s}]$ | $(+90\text{ ms}, -30\text{ ms})$ | **$60.0\text{ ms}$** | **$67.08\text{ ms}$** |
| `phone_M2.wav` | $[0.53\text{ s}, 2.52\text{ s}]$ | $T \approx 0.058$ | $[0.54\text{ s}, 2.51\text{ s}]$ | $(+10\text{ ms}, -10\text{ ms})$ | **$10.0\text{ ms}$** | **$10.00\text{ ms}$** |
| `studio_F2.wav`| $[0.77\text{ s}, 2.37\text{ s}]$ | $T \approx 0.086$ | $[0.77\text{ s}, 2.21\text{ s}]$ | $(0\text{ ms}, -160\text{ ms})$ | **$80.0\text{ ms}$** | **$113.14\text{ ms}$** |
| `studio_M2.wav`| $[0.45\text{ s}, 1.93\text{ s}]$ | $T \approx 0.065$ | $[0.47\text{ s}, 1.89\text{ s}]$ | $(+20\text{ ms}, -40\text{ ms})$ | **$30.0\text{ ms}$** | **$31.62\text{ ms}$** |
| **TRUNG BÌNH** | — | — | — | — | **$45.0\text{ ms}$** | **$55.46\text{ ms}$** |

---

## 5. Cấu Trúc Slide Thuyết Trình (Đúng 3 Phút)

* **Slide 1: Trang bìa (15 giây)**
  - Tên đề tài: *Phân đoạn Tiếng nói / Khoảng lặng bằng Ngưỡng Tự Thích Nghi Histogram (Giannakopoulos 2014)*.
  - Họ tên, MSSV: Sinh viên 2.
  - Lớp học phần: Xử lý tín hiệu số (2026).

* **Slide 2: Bản chất Thuật toán & Đồ thị Histogram (60 giây)**
  - Sơ đồ 5 bước: $STE_{norm} \rightarrow \text{Histogram 100 bins} \rightarrow \text{Lọc Moving Average 5 bins} \rightarrow \text{Tìm } M_1, M_2 \rightarrow T = \frac{5M_1 + M_2}{6}$.
  - Hình minh họa: Đồ thị Histogram trước và sau khi làm mịn, chỉ rõ vị trí 2 đỉnh $M_1$ (nhiễu nền) và $M_2$ (nguyên âm mạnh).
  - Giải thích ý nghĩa trọng số $W=5$: Ngăn chặn việc ngưỡng bị kéo lệch về các âm lượng cực đại.

* **Slide 3: Kết quả thực nghiệm trên 4 Tệp Kiểm thử (45 giây)**
  - Bảng định lượng MAE/RMSE và giá trị ngưỡng riêng biệt $T \in [0.058, 0.086]$ của từng file.
  - Đồ thị 4 cửa sổ kiểm thử với các vạch phân đoạn.

* **Slide 4: Phân tích Chuyên sâu Hiện tượng Cắt lẹm Âm Vô Thanh (60 giây)**
  - Chỉ rõ vì sao tệp `studio_F2` bị sai số lớn ở đuôi câu ($-160\text{ ms}$): Do âm vô thanh cuối câu có năng lượng rất thấp ($< 0.05$), trong khi ngưỡng thích nghi bị kéo lên tới $0.086$ vì đỉnh $M_2$ của nguyên âm rất lớn $\Rightarrow$ Thuật toán cắt kết thúc sớm $160\text{ ms}$.
  - Kết luận: TT2 có khả năng thích nghi cao với nhiễu môi trường biến đổi, nhưng có nhược điểm cố hữu là cắt trễ đầu câu và cắt sớm cuối câu đối với các phụ âm vô thanh.

---

## 6. Kịch Bản Vấn Đáp Bảo Vệ Đồ Án (Dành Riêng Cho SV 2)

1. **Câu hỏi**: *Tại sao phải làm mịn Histogram bằng Moving Average trước khi tìm cực đại?*
   - **Trả lời**: *Thưa thầy/cô, trong các tệp âm thanh thực tế, dữ liệu hữu hạn khiến histogram xuất hiện nhiều đỉnh gồ ghề ngẫu nhiên (nhiễu cục bộ). Nếu không lọc làm mịn, thuật toán sẽ bắt nhầm một đỉnh nhiễu nhỏ làm $M_1$ hoặc $M_2$, dẫn đến tính sai hoàn toàn giá trị ngưỡng thích nghi.*

2. **Câu hỏi**: *Tại sao sai số trung bình của TT2 ($45\text{ ms}$) lại cao hơn nhiều so với TT1 ($8.75\text{ ms}$)?*
   - **Trả lời**: *Thưa thầy/cô, đỉnh $M_2$ phản ánh năng lượng của các nguyên âm mạnh (Vowels), khiến ngưỡng thích nghi bị kéo lên tương đối cao ($T \approx 0.058 - 0.086$). Trong khi đó, các phụ âm vô thanh (`uv`) mở đầu hoặc kết thúc câu nói (như /s/, /t/, /h/) có mức năng lượng thấp hơn $0.03$. Do đó, ngưỡng cao của TT2 vô tình cắt cụt các âm vô thanh này, điển hình là đuôi tệp studio_F2 bị mất tới 160ms.*
