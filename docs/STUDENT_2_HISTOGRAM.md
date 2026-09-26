# BẢN ĐẶC TẢ NHIỆM VỤ CHI TIẾT: SINH VIÊN 2 (STUDENT 2)
## THUẬT TOÁN 2 (TT2): NGƯỠNG TỰ THÍCH NGHI THEO HISTOGRAM (GIANNAKOPOULOS 2014)
### Voice Activity Detection (VAD) with Adaptive Histogram-Based Energy Thresholding

---

## 1. THÔNG TIN VAI TRÒ & TRÁCH NHIỆM (ROLE INFORMATION & SCOPE)

* **Thành viên đảm nhiệm**: **Sinh viên 2** (Student 2).
* **Tên thuật toán**: Phân đoạn Tiếng nói / Khoảng lặng bằng Ngưỡng Tự Thích Nghi Dựa Trên Lược Đồ Tần Suất Năng Lượng (*Adaptive Histogram-Based Energy Thresholding*).
* **Căn cứ khoa học**: Giannakopoulos (2014), *A method for silence removal and segmentation of speech signals*.
* **Tệp mã nguồn chịu trách nhiệm độc quyền**:
  - `algorithms/tt2_histogram.py`: Cài đặt toàn bộ 5 bước tính histogram, làm mịn Moving Average, tìm đỉnh $M_1/M_2$, tính ngưỡng thích nghi và dự đoán VAD.
  - Tích hợp kiểm thử dòng lệnh: `uv run python main.py --algo tt2`
* **Các mô-đun lõi phối hợp sử dụng (`core/`)**:
  - `core/features.py`: Hàm `extract_ste_features`.
  - `core/postprocess.py`: Hàm `apply_threshold`, `remove_short_silences_200ms`, `extract_speech_boundaries`.
  - `core/metrics.py`: Hàm `calculate_mae_rmse`, `evaluate_file_performance`.
* **Khung thời gian báo cáo**: Đúng **3 phút trình bày Slide + 1 phút chạy Demo trực tiếp trên terminal**.

---

## 2. BẢN CHẤT BÀI TOÁN & CƠ SỞ LÝ THUYẾT (CORE PRINCIPLES & THEORETICAL BACKGROUND)

### 2.1 Triết lý Ngưỡng Tự Thích Nghi (Adaptive Thresholding Philosophy)
Trong khi Thuật toán 1 (TT1) áp dụng một ngưỡng toàn cục cố định duy nhất cho mọi tệp âm thanh, Thuật toán 2 (TT2) dựa trên một nhận thức thực tế:
- **Mỗi bản thu âm có một đặc trưng âm học riêng biệt**: Tùy thuộc vào khoảng cách micro, độ nhạy thiết bị thu, mức tạp âm môi trường và âm lượng phát âm của người nói.
- Do đó, việc áp một ngưỡng cố định có thể hoạt động tốt trên tập dữ liệu đã biết nhưng sẽ thất bại khi gặp một bản thu có mức khuếch đại hoặc nhiễu sàn hoàn toàn khác biệt.
- TT2 giải quyết vấn đề này bằng cách **tính toán động một giá trị ngưỡng $T$ riêng biệt cho từng tệp âm thanh**, trích xuất trực tiếp từ hình dạng phân bố tần suất của dãy năng lượng ngắn hạn $STE_{norm}$.

### 2.2 Hiện tượng Phân Bố Lưỡng Đỉnh (Bimodal Energy Distribution)
Khi biểu diễn tần suất xuất hiện của các giá trị năng lượng $STE_{norm}$ trên một tệp âm thanh chứa câu nói, đồ thị luôn thể hiện xu hướng phân bố lưỡng đỉnh (Bimodal Distribution):
1. **Cụm khoảng lặng / nhiễu nền**: Chiếm số lượng lớn khung ở vùng năng lượng rất thấp gần $0.0$, tạo nên đỉnh cực đại thứ nhất ($M_1$).
2. **Cụm tiếng nói**: Chiếm các khung có năng lượng dao động lớn từ $0.15$ đến $0.60$ (chủ yếu là các nguyên âm có tính cộng hưởng buồng thanh quản), tạo nên vùng tập trung cực đại thứ hai ($M_2$).
3. **Mục tiêu của TT2**: Tự động phát hiện vị trí của hai đỉnh $M_1$ và $M_2$ trên lược đồ tần suất để thiết lập một ranh giới năng lượng ngăn cách giữa hai phân bố này.

### 2.3 Vai trò của Bộ lọc Làm mịn Trung bình trượt (Moving Average Smoothing)
Do dữ liệu của một tệp âm thanh có độ dài hữu hạn (chỉ vài trăm khung), biểu đồ tần suất thô (raw histogram) thường xuất hiện các gai nhọn ngẫu nhiên (spurious local spikes). Nếu tìm cực đại trực tiếp trên histogram thô, thuật toán sẽ bắt nhầm các đỉnh nhiễu cục bộ thay vì đỉnh thực của phân bố. Do đó, bước **làm mịn bằng bộ lọc trung bình trượt 5 bins** là bắt buộc để khôi phục đường bao phân bố trơn tru.

### 2.4 Vai trò của Trọng số $W = 5.0$
Công thức tính ngưỡng của Giannakopoulos sử dụng trung bình có trọng số:
$$
T = \frac{W \cdot M_1 + M_2}{W + 1} = \frac{5M_1 + M_2}{6}
$$
- Đỉnh $M_2$ của tiếng nói phản ánh năng lượng của các nguyên âm mạnh, thường nằm ở mức khá cao ($0.20 - 0.40$).
- Nếu lấy trung bình cộng đối xứng ($W = 1 \Rightarrow T = \frac{M_1 + M_2}{2}$), ngưỡng sẽ rơi vào khoảng $0.10 - 0.20$. Mức ngưỡng này quá cao, sẽ cắt mất toàn bộ các phụ âm hữu thanh yếu và âm vô thanh.
- Việc đặt trọng số $W = 5$ giúp **kéo ngưỡng dịch mạnh về phía đỉnh khoảng lặng $M_1$**, hạ thấp ranh giới để bảo vệ các âm thanh yếu của giọng nói.

---

## 3. QUY TRÌNH 5 BƯỚC & CÔNG THỨC TOÁN HỌC (5-STEP PROCESS & MATHEMATICAL FORMULATIONS)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DÃY NĂNG LƯỢNG CHUẨN HÓA STE_norm                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 1 & 2: Xây dựng Histogram 100 bins trong đoạn [0.0, 1.0]           │
│             Độ rộng mỗi bin: Δ = 0.01; Đếm tần số xuất hiện            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 3: Làm mịn Histogram bằng Moving Average cửa sổ 5 bins             │
│         H_smooth[i] = (1/5) * sum_{j=-2}^{2} H[i+j]                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 4: Phát hiện 2 cực đại địa phương:                                │
│         - M1: Đỉnh khoảng lặng / nhiễu nền (vùng năng lượng thấp)       │
│         - M2: Đỉnh tiếng nói (vùng nguyên âm năng lượng cao)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 5: Tính ngưỡng thích nghi riêng cho tệp:                          │
│         T = (5 * M1 + M2) / 6                                          │
└────────────────────────────────────────────────────────────────────────┘
```

### Bước 1: Trích xuất Dãy Năng Lượng $STE_{norm}$
Tín hiệu được phân khung với kích thước $20\text{ ms}$ ($N = 0.02 f_s$), bước nhảy $10\text{ ms}$ ($H = 0.01 f_s$). Năng lượng mỗi khung được tính bằng tổng bình phương và chuẩn hóa cực đại:
$$
STE[m] = \sum_{n=0}^{N-1} x^2[m \cdot H + n], \quad STE_{norm}[m] = \frac{STE[m]}{\max_k STE[k]} \in [0.0, 1.0]
$$

### Bước 2: Xây dựng Histogram 100 Bins trong $[0.0, 1.0]$
Chia đoạn giá trị năng lượng $[0.0, 1.0]$ thành $B = 100$ khoảng đều nhau:
- **Độ rộng mỗi bin**: $\Delta = \frac{1.0 - 0.0}{100} = 0.01$.
- **Biên của bin thứ $b$** ($b = 0, 1, \dots, 99$):
  $$
  \text{Bin } b = [e_b, e_{b+1}) = [0.01 \cdot b, \; 0.01 \cdot (b + 1))
  $$
- **Tâm của bin thứ $b$**:
  $$
  c_b = \frac{e_b + e_{b+1}}{2} = 0.005 + 0.01 \cdot b
  $$
- Đếm số lượng khung rơi vào từng bin:
  $$
  H[b] = \sum_{m=0}^{M-1} \mathbb{I}\left(STE_{norm}[m] \in \text{Bin } b\right)
  $$
  *(Riêng giá trị tại biên $1.0$ được gộp vào bin cuối cùng $b = 99$).*

### Bước 3: Làm Mịn Histogram bằng Bộ Lọc Trung Bình Trượt (Moving Average)
Lọc làm mịn với kích thước cửa sổ 5 bins ($k = 2$ phần tử mỗi bên):
$$
H_{smooth}[i] = \frac{1}{2k+1} \sum_{j=-k}^k H[i+j] = \frac{1}{5} \sum_{j=-2}^2 H[i+j]
$$
- **Xử lý điều kiện biên**: Tại các vị trí đầu mút $i < 2$ hoặc $i > 97$, chỉ lấy trung bình cộng trên các phần tử hợp lệ trong phạm vi $[ \max(0, i-2), \min(99, i+2) ]$ để không làm méo mó tổng năng lượng.
- **Yêu cầu kỹ thuật**: Tự lập trình bằng vòng lặp và mảng NumPy cơ bản, **nghiêm cấm dùng `scipy.signal.convolve` hay `scipy.ndimage`**.

### Bước 4: Phát hiện Hai Cực Đại Địa Phương ($M_1$ và $M_2$)
Một điểm $i$ được định nghĩa là cực đại địa phương nếu giá trị của nó lớn hơn cả hai điểm lân cận liền kề:
$$
H_{smooth}[i] > H_{smooth}[i-1] \quad \text{và} \quad H_{smooth}[i] > H_{smooth}[i+1]
$$
- **Đỉnh khoảng lặng $M_1$**: Là cực đại địa phương đầu tiên tìm thấy tính từ chỉ số $0$ (vùng năng lượng thấp nhất), đại diện cho nhiễu nền:
  $$
  M_1 = c_{i_{M1}}
  $$
- **Đỉnh tiếng nói $M_2$**: Là cực đại địa phương có tần suất xuất hiện cao nhất trong số các đỉnh còn lại nằm ở vùng năng lượng cao hơn ($i > i_{M1} + 2$):
  $$
  M_2 = c_{i_{M2}} \quad \text{với } i_{M2} = \arg\max_{j \in \text{peaks}, j > i_{M1}+2} H_{smooth}[j]
  $$
- **Cơ chế dự phòng (Fallback)**:
  - Nếu không có đỉnh nào (dữ liệu phẳng): Gán mặc định $M_1 = 0.001$, $M_2 = 0.200$.
  - Nếu chỉ có 1 đỉnh duy nhất: Chọn điểm có giá trị lớn thứ hai cách đỉnh thứ nhất ít nhất 10 bins.

### Bước 5: Tính Ngưỡng Thích Nghi (Adaptive Threshold Calculation)
Ngưỡng phân đoạn năng lượng của tệp âm thanh được tính theo công thức:
$$
T = \frac{5 \cdot M_1 + M_2}{6}
$$
Sau đó, ngưỡng $T$ này được chuyển vào bộ lọc hậu xử lý:
1. So sánh $d[m] = 1$ nếu $STE_{norm}[m] \ge T$, ngược lại $0$.
2. Gộp các khoảng lặng $< 200\text{ ms}$ nằm giữa hai đoạn tiếng nói thành tiếng nói.
3. Trích xuất biên bắt đầu $\hat{T}_{start}$ và kết thúc $\hat{T}_{end}$.

---

## 4. BẢNG KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG & PHÂN TÍCH HIỆN TƯỢNG CẮT LẸM ÂM VÔ THANH

Sinh viên 2 thực thi lệnh kiểm thử trên tập `TinHieuKiemThu`:
```bash
uv run python main.py --algo tt2 --no-plot
```

### 4.1 Bảng số liệu đối sánh trên 4 tệp kiểm thử:

| Tệp Kiểm Thử | Ground-Truth $[T_s, T_e]$ | Đỉnh $M_1$ | Đỉnh $M_2$ | Ngưỡng $T$ Tính Được | TT2 Dự Đoán $[\hat{T}_s, \hat{T}_e]$ | Sai Lệch $(\Delta T_s, \Delta T_e)$ | MAE (ms) | RMSE (ms) |
|---|---|---|---|---|---|---|---|---|
| `phone_F2.wav` | $[1.02\text{ s}, 4.04\text{ s}]$ | $0.005$ | $0.465$ | **$0.0817$** | $[1.11\text{ s}, 4.01\text{ s}]$ | $(+90\text{ ms}, -30\text{ ms})$ | **$60.0\text{ ms}$** | **$67.08\text{ ms}$** |
| `phone_M2.wav` | $[0.53\text{ s}, 2.52\text{ s}]$ | $0.005$ | $0.325$ | **$0.0583$** | $[0.54\text{ s}, 2.51\text{ s}]$ | $(+10\text{ ms}, -10\text{ ms})$ | **$10.0\text{ ms}$** | **$10.00\text{ ms}$** |
| `studio_F2.wav`| $[0.77\text{ s}, 2.37\text{ s}]$ | $0.005$ | $0.495$ | **$0.0867$** | $[0.77\text{ s}, 2.21\text{ s}]$ | $(0\text{ ms}, -160\text{ ms})$ | **$80.0\text{ ms}$** | **$113.14\text{ ms}$** |
| `studio_M2.wav`| $[0.45\text{ s}, 1.93\text{ s}]$ | $0.005$ | $0.365$ | **$0.0650$** | $[0.47\text{ s}, 1.89\text{ s}]$ | $(+20\text{ ms}, -40\text{ ms})$ | **$30.0\text{ ms}$** | **$31.62\text{ ms}$** |
| **TRUNG BÌNH** | — | — | — | **$0.0729$** | — | — | **$45.00\text{ ms}$** | **$55.46\text{ ms}$** |

### 4.2 Phân tích Chuyên sâu Hiện tượng Cắt lẹm Âm Vô Thanh (Unvoiced Truncation Analysis)
Một câu hỏi bảo vệ đồ án mang tính quyết định: **Tại sao sai số trung bình của TT2 ($\text{MAE} = 45.00\text{ ms}$) lại cao hơn rõ rệt so với TT1 ($\text{MAE} = 8.75\text{ ms}$)?**

Sinh viên 2 cần nắm vững 3 luận điểm khoa học sau để trả lời:
1. **Bản chất của ngưỡng thích nghi bị kéo cao**:
   - Các ngưỡng thích nghi tính ra cho 4 file kiểm thử đều dao động trong khoảng $T \in [0.058, 0.086]$.
   - Mức ngưỡng này **cao gấp 23 đến 34 lần** so với ngưỡng tối ưu toàn cục $T_{opt} = 0.0025$ của TT1!
   - Nguyên nhân: Đỉnh nguyên âm $M_2$ có năng lượng quá lớn ($0.32 - 0.50$). Cho dù đã áp dụng trọng số chia 6, giá trị $T$ vẫn bị kéo lên trên $0.05$.
2. **Hậu quả cắt lẹm âm vô thanh (`uv`) ở đuôi câu (Điển hình trên `studio_F2`)**:
   - Ở tệp `studio_F2`, câu nói kết thúc bằng một âm vô thanh kéo dài từ $2.21\text{ s}$ đến $2.37\text{ s}$ ($160\text{ ms}$). Mức năng lượng của đoạn âm vô thanh này chỉ đạt $0.02 - 0.04$.
   - Vì ngưỡng của TT2 đối với `studio_F2` là $T = 0.0867$, toàn bộ đoạn âm vô thanh đuôi câu có năng lượng $< 0.0867$ bị coi là khoảng lặng!
   - Do đó, thuật toán cắt kết thúc câu nói sớm $160\text{ ms}$, đẩy sai số MAE của riêng file này lên mức $80.0\text{ ms}$.
3. **Hậu quả nhận diện trễ ở đầu câu (Điển hình trên `phone_F2`)**:
   - Ở tệp `phone_F2`, phụ âm mở đầu câu là âm xát vô thanh có năng lượng yếu từ $1.02\text{ s}$ đến $1.11\text{ s}$.
   - Ngưỡng $T = 0.0817$ không thể bắt được đoạn âm xát này, dẫn đến việc thuật toán chỉ phát hiện tiếng nói khi bước vào nguyên âm chính tại $1.11\text{ s}$, gây trễ $+90\text{ ms}$ ở đầu câu.

---

## 5. DANH SÁCH CÔNG VIỆC TỪNG BƯỚC CHO SINH VIÊN 2 (STEP-BY-STEP TODO CHECKLIST)

### Checklist triển khai code trong `algorithms/tt2_histogram.py`:
- [ ] **Bước 1**: Mở tệp `algorithms/tt2_histogram.py`, rà soát các hàm thành phần và kiểu dữ liệu:
  - `compute_histogram_100bins(ste_norm: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]`
  - `moving_average_smooth(histogram: np.ndarray, window_size: int) -> np.ndarray`
  - `find_histogram_peaks(smoothed_hist: np.ndarray, bin_centers: np.ndarray) -> Tuple[float, float]`
  - `compute_adaptive_threshold_tt2(ste_norm: np.ndarray, weight_w: float) -> float`
  - `predict_vad_tt2(signal: np.ndarray, sample_rate: int, weight_w: float) -> Tuple[float, float, np.ndarray, np.ndarray, float]`
- [ ] **Bước 2**: Giữ nguyên toàn bộ vết in định danh hàm: `print("ten_ham")` và chú thích `# //TODO`.
- [ ] **Bước 3**: Trong `compute_histogram_100bins`:
  - Khởi tạo 101 mốc biên bằng `np.linspace(0.0, 1.0, 101)`.
  - Phân loại bằng `np.digitize(ste_norm, bin_edges) - 1`.
  - Cắt clip chỉ số trong $[0, 99]$ để đưa mẫu $1.0$ vào bin 99.
  - Tính tâm bin và trả về `(counts, bin_centers)`.
- [ ] **Bước 4**: Trong `moving_average_smooth`:
  - Dùng vòng lặp cơ bản tính trung bình trượt trên cửa sổ $[-2, +2]$.
  - Kiểm tra nghiêm ngặt: Tuyệt đối không import `scipy.signal`.
- [ ] **Bước 5**: Trong `find_histogram_peaks`:
  - Quét tìm cực đại địa phương thỏa mãn $H[i] > H[i-1]$ và $H[i] > H[i+1]$.
  - Gán $M_1$ là đỉnh đầu tiên, $M_2$ là đỉnh cao nhất ở vùng năng lượng cao.
  - Cài đặt đầy đủ 2 cơ chế fallback khi số đỉnh $< 2$.
- [ ] **Bước 6**: Chạy kiểm thử tự động:
  ```bash
  uv run pytest tests/test_vad_pipeline.py -k "tt2"
  uv run python main.py --algo tt2 --no-plot
  ```
  Xác nhận kết quả hiển thị khớp với Bảng số liệu mục 4.

---

## 6. HƯỚNG DẪN TRÌNH BÀY SLIDE & BẢO VỆ ĐỒ ÁN (PRESENTATION GUIDE)

### 6.1 Quy định hình thức nghiêm ngặt
* **Thời lượng tối đa**: Đúng **3 phút trình bày Slide + 1 phút chạy Demo trực tiếp**.
* **Định dạng chuẩn**:
  - $\le 7$ dòng chữ mỗi slide.
  - $\le 10$ từ mỗi dòng.
  - Cỡ chữ $\ge 18\text{ pt}$, độ tương phản màu cao.
  - **CẤM**: Không trình bày lý thuyết giáo trình chung chung; chỉ tập trung vào phân bố Histogram, công thức tính ngưỡng thích nghi, số liệu MAE/RMSE và cơ chế cắt lẹm âm vô thanh.

### 6.2 Cấu trúc 4 Slide chi tiết của Sinh viên 2

#### Slide 1: Trang bìa (Thời lượng: 15 giây)
* **Tiêu đề**: Phân Đoạn Tiếng Nói / Khoảng Lặng Bằng Ngưỡng Tự Thích Nghi Histogram
* **Thuật toán**: Giannakopoulos (2014) Adaptive Histogram Thresholding
* **Người thực hiện**: [Họ và tên Sinh viên 2] — MSSV: [Mã số SV 2]
* **Lớp học phần**: Xử Lý Tín Hiệu Số (XLTHS - 2026)

#### Slide 2: Bản chất Thuật toán & Đồ thị Histogram Làm Mịn (Thời lượng: 60 giây)
* **Dòng 1**: Triết lý thích nghi: Mỗi file âm thanh có sàn nhiễu riêng biệt.
* **Dòng 2**: Quy trình 5 bước: $STE_{norm} \rightarrow$ Histogram 100 bins $\rightarrow$ Lọc MA 5 bins $\rightarrow$ Tìm $M_1, M_2 \rightarrow$ Tính $T$.
* **Dòng 3**: Lọc Moving Average: $H_{smooth}[i] = \frac{1}{5} \sum_{j=-2}^2 H[i+j]$ (loại bỏ gai nhiễu giả).
* **Dòng 4**: Đỉnh $M_1$: Đại diện khoảng lặng/nhiễu ($M_1 \approx 0.005$).
* **Dòng 5**: Đỉnh $M_2$: Đại diện nguyên âm tiếng nói ($M_2 \approx 0.32 - 0.50$).
* **Dòng 6**: **Công thức ngưỡng: $T = \frac{5M_1 + M_2}{6}$ ($W=5$ kéo ngưỡng về phía $M_1$)**.
* *(Hình ảnh minh họa slide)*: Đồ thị so sánh histogram thô vs histogram sau làm mịn, chỉ rõ 2 mũi tên $M_1$ và $M_2$.

#### Slide 3: Kết quả Thực nghiệm trên 4 Tệp Kiểm thử (Thời lượng: 45 giây)
* **Dòng 1**: Kiểm thử độc lập trên 4 tệp kiểm thử `TinHieuKiemThu`.
* **Dòng 2**: Ngưỡng thích nghi dao động theo từng file: $T \in [0.058, 0.086]$.
* **Dòng 3**: `phone_M2`: Kết quả tốt $\text{MAE} = 10.0\text{ ms}$, $\text{RMSE} = 10.0\text{ ms}$.
* **Dòng 4**: `studio_M2`: $\text{MAE} = 30.0\text{ ms}$ (lệch nhẹ hai đầu).
* **Dòng 5**: `phone_F2`: $\text{MAE} = 60.0\text{ ms}$ (trễ $+90\text{ ms}$ do âm vô thanh đầu câu).
* **Dòng 6**: `studio_F2`: $\text{MAE} = 80.0\text{ ms}$ (mất $-160\text{ ms}$ âm vô thanh đuôi câu).
* **Dòng 7**: **Trung bình tập kiểm thử: $\text{MAE} = 45.00\text{ ms}$, $\text{RMSE} = 55.46\text{ ms}$**.
* *(Hình ảnh minh họa slide)*: Bảng số liệu kết quả và đồ thị 4 cửa sổ kiểm thử.

#### Slide 4: Phân tích Hiện tượng Cắt lẹm Âm Vô Thanh & Đánh giá (Thời lượng: 60 giây)
* **Dòng 1**: Nguyên nhân sai số $\text{MAE} = 45\text{ ms}$: Ngưỡng thích nghi ($0.058 - 0.086$) bị kéo quá cao.
* **Dòng 2**: Đỉnh $M_2$ bị chi phối bởi nguyên âm mạnh, kéo giá trị ngưỡng vượt mức âm gió.
* **Dòng 3**: Âm vô thanh (`uv`) có năng lượng thấp ($< 0.04$) bị nhận nhầm thành khoảng lặng.
* **Dòng 4**: Hậu quả: Cắt trễ phụ âm mở đầu và cắt cụt phụ âm kết thúc.
* **Dòng 5**: Hướng khắc phục: Tăng trọng số $W \ge 10$ hoặc kết hợp thêm Zero Crossing Rate (ZCR).
* **Dòng 6**: Đánh giá: Rất mạnh mẽ khi nhiễu biến đổi lớn, nhưng cần cải tiến cho âm vô thanh.

### 6.3 Kịch bản Chạy Demo Trực Tiếp (1 phút)
1. Mở cửa sổ Terminal tại `/home/bim/Projects/DSP_MidTerm`.
2. Gõ lệnh:
   ```bash
   uv run python main.py --algo tt2
   ```
3. Thuyết minh với Hội đồng chấm thi:
   - *"Thưa thầy cô, em đang chạy thuật toán TT2 trên 4 file kiểm thử. Trên terminal, chương trình in ra ngưỡng thích nghi riêng biệt cho từng file: ví dụ phone_F2 là T=0.0817, phone_M2 là T=0.0583, studio_F2 là T=0.0867."*
   - *"Trên 4 cửa sổ đồ thị hiển thị ở 4 góc màn hình, thầy cô có thể quan sát thấy ở tệp studio_F2 (góc dưới-trái), vạch màu xanh kết thúc sớm hơn vạch đỏ 160 ms. Đây chính là minh chứng trực quan cho hiện tượng cắt lẹm âm vô thanh đuôi câu khi ngưỡng thích nghi bị nguyên âm kéo lên quá cao."*

---

## 7. KỊCH BẢN VẤN ĐÁP BẢO VỆ ĐỒ ÁN (ORAL DEFENSE Q&A)

### Câu hỏi 1: Tại sao phải làm mịn Histogram bằng Moving Average trước khi tìm cực đại? Nếu không làm mịn thì xảy ra lỗi gì?
* **Trả lời trọng tâm**:
  > *"Thưa Thầy/Cô, do độ dài của mỗi tệp âm thanh chỉ từ 2 đến 4 giây (khoảng 200 - 450 khung), số lượng mẫu phân vào 100 bins là hữu hạn. Điều này tạo ra hiện tượng răng cưa ngẫu nhiên và các đỉnh nhọn giả mạo (spurious local spikes) trên biểu đồ tần suất thô. Nếu không dùng bộ lọc trung bình trượt 5 bins để làm mịn, điều kiện cực đại địa phương ($H[i] > H[i-1]$ và $H[i] > H[i+1]$) sẽ bắt nhầm một gai nhiễu ngẫu nhiên làm đỉnh $M_1$ hoặc $M_2$. Khi đó, vị trí đỉnh bị lệch hoàn toàn, kéo theo ngưỡng thích nghi $T$ bị tính sai lệch hàng chục lần, dẫn đến sự đổ vỡ toàn diện của thuật toán phân đoạn."*

### Câu hỏi 2: Tại sao sai số trung bình của TT2 ($45.0\text{ ms}$) lại lớn gấp hơn 5 lần so với TT1 ($8.75\text{ ms}$)?
* **Trả lời trọng tâm**:
  > *"Thưa Thầy/Cô, nguyên nhân cốt lõi nằm ở sự chênh lệch năng lượng giữa nguyên âm và âm vô thanh. Đỉnh $M_2$ trên histogram được hình thành bởi các nguyên âm (Voiced) có năng lượng cực đại, thường nằm ở mức $0.32 - 0.50$. Dù công thức của Giannakopoulos đã dùng trọng số $W=5$ để giảm bớt ảnh hưởng của $M_2$, ngưỡng thích nghi tính ra vẫn dao động trong khoảng $0.058 - 0.086$ (cao gấp 30 lần ngưỡng $0.0025$ của TT1). Trong khi đó, các âm vô thanh (`uv`) ở đầu câu hoặc cuối câu (như /s/, /t/, /h/) có mức năng lượng rất thấp, chỉ từ $0.01$ đến $0.04$. Do ngưỡng $T > 0.058$, thuật toán TT2 coi toàn bộ các âm vô thanh này là khoảng lặng, dẫn đến việc cắt trễ đầu câu và cắt cụt đuôi câu (điển hình mất tới $160\text{ ms}$ ở đuôi tệp `studio_F2`)."*

### Câu hỏi 3: Làm thế nào để khắc phục hiện tượng cắt lẹm âm vô thanh mà vẫn giữ nguyên triết lý thích nghi của TT2?
* **Trả lời trọng tâm**:
  > *"Thưa Thầy/Cô, để khắc phục hiện tượng này trong kỹ thuật xử lý tín hiệu tiếng nói thực tế, nhóm đề xuất hai giải pháp:
  > 1. **Về mặt năng lượng**: Ta có thể tăng mạnh trọng số $W$ từ $5$ lên $12$ hoặc $15$ ($T = \frac{15M_1 + M_2}{16}$), giúp dìm ngưỡng $T$ xuống sát $M_1$ hơn nữa (về vùng $0.01 - 0.02$).
  > 2. **Về mặt đặc trưng phối hợp**: Kết hợp thêm đặc trưng Tốc độ qua điểm không (Zero Crossing Rate - ZCR). Vì các âm vô thanh có năng lượng thấp nhưng ZCR lại rất cao (do tính ngẫu nhiên của âm gió), cơ chế ngưỡng kép (Dual-threshold: kết hợp năng lượng cho nguyên âm và ZCR cho âm vô thanh) sẽ cứu vãn được hoàn toàn các đoạn âm xát ở hai đầu câu nói."*

### Câu hỏi 4: Trong tình huống nào thì thuật toán thích nghi TT2 thể hiện sự vượt trội so với ngưỡng cố định TT1?
* **Trả lời trọng tâm**:
  > *"Thuật toán TT2 sẽ vượt trội hoàn toàn khi hệ thống phải xử lý các luồng âm thanh có độ lợi khuếch đại (gain) hoặc môi trường nhiễu biến động liên tục và khó đoán định trước (ví dụ người nói tiến lại gần micro rồi lùi ra xa, hoặc tín hiệu thu từ nhiều dòng điện thoại khác nhau có SNR dao động mạnh từ $10\text{ dB}$ đến $40\text{ dB}$). Khi đó, ngưỡng cố định của TT1 sẽ bị 'mù' và gãy đổ do không thể khái quát hóa, trong khi TT2 luôn tự co giãn theo phân bố năng lượng nội tại của từng tệp âm thanh để tự tìm ra ranh giới tương đối."*
