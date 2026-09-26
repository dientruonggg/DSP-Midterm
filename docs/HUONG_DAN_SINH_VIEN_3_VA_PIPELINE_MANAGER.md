# HƯỚNG DẪN TOÀN DIỆN: SINH VIÊN 3 & TRƯỞNG NHÓM (PIPELINE MANAGER)
## KIẾN TRÚC HỆ THỐNG DÙNG CHUNG & THUẬT TOÁN 3 (TT3: GAUSSIAN BAYES)

---

## 1. Thông Tin Vai Trò Kép (Dual Role)
* **Thành viên phụ trách**: Sinh viên 3 (Kiêm Trưởng nhóm kiến trúc / Pipeline Manager).
* **Trách nhiệm kép**:
  1. **Pipeline Manager (Trưởng nhóm)**:
     - Thiết kế và duy trì lõi dùng chung (`core/`): Đọc WAV, phân tích nhãn LAB, trích xuất đặc trưng STE chuẩn hóa, bộ lọc loại bỏ khoảng lặng ảo $< 200\text{ ms}$, tính độ đo sai số MAE/RMSE (ms).
     - Xây dựng chương trình điều phối trung tâm `main.py`: Tự động duyệt 4 tệp kiểm thử, sắp xếp đồng thời **4 Figure tại 4 góc màn hình** theo đúng quy định thi cử.
     - Quản lý tích hợp Git, chuẩn hóa môi trường bằng `uv`, tổng hợp bảng Benchmark so sánh cả 3 thuật toán.
  2. **Nghiên cứu thuật toán TT3**:
     - Cài đặt tệp `algorithms/tt3_gaussian.py`: Thống kê phân bố Gauss trên tập huấn luyện, giải phương trình xác suất Bayes để tìm ngưỡng tối ưu lý thuyết $T_{Bayes} \approx 0.002864$.
* **Thời lượng bảo vệ**: Đúng **3 phút trình bày Slide + 1 phút chạy Demo trực tiếp**.

---

## 2. Phần A: Trách Nhiệm Pipeline Manager (Lõi Dùng Chung & Điều Phối)

### 2.1 Kiến trúc mô-đun chuẩn hóa
Toàn bộ mã nguồn dự án được tổ chức phân tầng rõ ràng để tránh xung đột Git khi làm việc nhóm:

```
DSP_MidTerm/
├── core/                   # [Pipeline Manager duy trì] Lõi dùng chung
│   ├── __init__.py
│   ├── io_utils.py         # Đọc WAV bằng wave chuẩn, đọc LAB, tách Ground-Truth
│   ├── features.py         # Tự code Framing (20ms/10ms), tính STE và STE_norm
│   ├── postprocess.py      # Áp ngưỡng, nối khoảng lặng ảo < 200ms (20 khung)
│   └── metrics.py          # Tính MAE, RMSE biên thời gian (đơn vị ms)
│
├── algorithms/             # Mô-đun thuật toán của 3 thành viên
│   ├── __init__.py
│   ├── tt1_hodgkinson.py   # [SV 1] Tối ưu hóa ngưỡng toàn cục cố định (T_opt = 0.0025)
│   ├── tt2_histogram.py    # [SV 2] Ngưỡng tự thích nghi theo Histogram (W = 5)
│   └── tt3_gaussian.py     # [SV 3 & PM] Thống kê Gauss & Ngưỡng xác suất Bayes (T = 0.00286)
│
├── main.py                 # [Pipeline Manager duy trì] Script thực thi và hiển thị 4 góc
├── pyproject.toml          # Quản lý thư viện qua uv (numpy, matplotlib)
└── docs/                   # 3 bộ tài liệu phân công cho 3 thành viên
```

### 2.2 Các quy tắc bất biến của Lõi Dùng Chung (`core/`)

1. **Chuẩn hóa khung tín hiệu (Framing)**:
   - Kích thước khung (Frame Size): $20\text{ ms} \Rightarrow N = 0.02 \times f_s$ mẫu.
   - Bước nhảy khung (Hop Size): $10\text{ ms} \Rightarrow H = 0.01 \times f_s$ mẫu.
   - Năng lượng ngắn hạn chuẩn hóa:
     $$
     STE[m] = \sum_{n=0}^{N-1} x^2[m \cdot H + n], \quad STE_{norm}[m] = \frac{STE[m]}{\max_k STE[k]}
     $$
2. **Quy tắc gộp khoảng lặng ảo $< 200\text{ ms}$ (20 khung)**:
   - Mọi chuỗi số 0 (silence) nằm giữa hai đoạn tiếng nói có độ dài $< 200\text{ ms}$ (tức $< 20$ khung với bước nhảy 10 ms) **bắt buộc phải chuyển thành 1 (tiếng nói)** để tránh ngắt quãng tại các âm tắc (stop consonants).
3. **Độ đo sai số định lượng (MAE & RMSE)**:
   - Sai số tính bằng **miligiây (ms)**:
     $$
     \text{MAE} = \frac{|\hat{T}_{start} - T_{start}| + |\hat{T}_{end} - T_{end}|}{2} \times 1000\text{ (ms)}
     $$
     $$
     \text{RMSE} = \sqrt{\frac{(\hat{T}_{start} - T_{start})^2 + (\hat{T}_{end} - T_{end})^2}{2}} \times 1000\text{ (ms)}
     $$

### 2.3 Cơ chế định vị 4 cửa sổ tại 4 góc màn hình trong `main.py`
Để đáp ứng yêu cầu khắt khe của giảng viên: **Chỉ bấm Run 01 lần duy nhất trên `main.py` và hiển thị đồng thời 4 cửa sổ tại 4 góc màn hình**:

```python
coords_map = {
    "top_left": (0, 0),                 # phone_F2
    "top_right": (win_w, 0),            # phone_M2
    "bottom_left": (0, win_h + 30),     # studio_F2
    "bottom_right": (win_w, win_h + 30) # studio_M2
}
```
* Tệp `main.py` tự động nhận diện backend đồ họa (TkAgg / Qt) và thiết lập vị trí màn hình thông qua hàm `setup_screen_window`.

---

## 3. Phần B: Thuật Toán 3 (TT3) - Phân Bố Xác Suất Gauss Bayes

### 3.1 Cơ sở lý thuyết toán học
Mỗi khung tín hiệu được xem như một biến ngẫu nhiên liên tục biểu diễn mức năng lượng chuẩn hóa $X = STE_{norm}$. Tín hiệu bao gồm 2 phân bố dữ liệu riêng biệt:
- Lớp Khoảng lặng (Silence): $X \sim \mathcal{N}(\mu_{sil}, \sigma_{sil}^2)$.
- Lớp Tiếng nói (Speech): $X \sim \mathcal{N}(\mu_{sp}, \sigma_{sp}^2)$.

Hàm mật độ xác suất chuẩn (Gaussian PDF):

$$
p(x|\text{class}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

### 3.2 Khảo sát thống kê trên 4 tệp huấn luyện (`TinHieuHuanLuyen`)
Sinh viên 3 trích xuất toàn bộ các khung âm thanh, gắn nhãn dựa trên file `.lab` chuẩn:
* **Tập khung Khoảng lặng ($499$ frames)**:
  - $\mu_{sil} \approx 0.000359$
  - $\sigma_{sil} \approx 0.000715$
* **Tập khung Tiếng nói ($795$ frames)**:
  - $\mu_{sp} \approx 0.196747$
  - $\sigma_{sp} \approx 0.233472$

### 3.3 So sánh 2 phương pháp xác định ngưỡng

#### Cách 1: Điểm cân bằng Z-score ($T_{equal}$) — *Thất bại*
$$
T_{equal} = \frac{\mu_{sil}\sigma_{sp} + \mu_{sp}\sigma_{sil}}{\sigma_{sil} + \sigma_{sp}} \approx \mathbf{0.000959}
$$
* **Nguyên nhân thất bại**: Do $\sigma_{sil} \ll \sigma_{sp}$, giá trị $T_{equal}$ bị kéo sát về phía $\mu_{sil}$. Trong môi trường điện thoại (`phone`), sàn nhiễu có năng lượng lên tới $0.0012$, dẫn đến việc nhận nhầm toàn bộ nhiễu thành tiếng nói ($\text{MAE} = 203.75\text{ ms}$).

#### Cách 2: Điểm giao xác suất Bayes (Minimum Error Bayes) — *Thành công xuất sắc*
Giả sử xác suất tiên nghiệm của 2 lớp bằng nhau ($P(\text{sil}) = P(\text{sp})$), điểm quyết định tối ưu là nghiệm phương trình $p(x|\text{sil}) = p(x|\text{sp})$:

$$
-\ln(\sigma_{sil}) - \frac{(x - \mu_{sil})^2}{2\sigma_{sil}^2} = -\ln(\sigma_{sp}) - \frac{(x - \mu_{sp})^2}{2\sigma_{sp}^2}
$$

Chuyển về phương trình bậc hai: $A x^2 + B x + C = 0$, với:
* $A = \frac{1}{\sigma_{sil}^2} - \frac{1}{\sigma_{sp}^2}$
* $B = -2 \left(\frac{\mu_{sil}}{\sigma_{sil}^2} - \frac{\mu_{sp}}{\sigma_{sp}^2}\right)$
* $C = \frac{\mu_{sil}^2}{\sigma_{sil}^2} - \frac{\mu_{sp}^2}{\sigma_{sp}^2} + 2 \ln\left(\frac{\sigma_{sil}}{\sigma_{sp}}\right)$

Giải phương trình trong khoảng $[\mu_{sil}, \mu_{sp}]$ cho nghiệm duy nhất:

$$
T_{Bayes} \approx \mathbf{0.002864}
$$

> [!NOTE]
> **Điểm kỳ diệu**: Ngưỡng lý thuyết thống kê $T_{Bayes} \approx 0.00286$ gần như trùng khớp hoàn hảo với ngưỡng thực nghiệm tối ưu $T_{opt} = 0.0025$ mà TT1 tìm được qua Grid Search!

---

## 4. Bản Đặc Tả Giao Diện & Mã Nguồn Cần Hoàn Thiện

Tệp `algorithms/tt3_gaussian.py` đã có sẵn đầy đủ khung giao diện chuẩn (interface) cùng comment `# //TODO` và lệnh in định danh hàm:

### 4.1 Hàm `extract_speech_silence_ste_frames`
```python
def extract_speech_silence_ste_frames(training_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    # //TODO: Partition training frames into silence vs speech collections based on ground truth
    print("extract_speech_silence_ste_frames")
    ...
```

### 4.2 Hàm `estimate_gaussian_parameters`
```python
def estimate_gaussian_parameters(
    silence_ste: np.ndarray,
    speech_ste: np.ndarray
) -> Tuple[float, float, float, float]:
    # //TODO: Compute mean and std for silence and speech distributions using basic numpy functions
    print("estimate_gaussian_parameters")
    ...
```

### 4.3 Hàm `solve_bayes_decision_threshold`
```python
def solve_bayes_decision_threshold(
    mu_sil: float,
    sigma_sil: float,
    mu_sp: float,
    sigma_sp: float
) -> float:
    # //TODO: Solve quadratic equation for Bayes decision boundary where Gaussian PDFs intersect
    print("solve_bayes_decision_threshold")
    ...
```

### 4.4 Hàm `predict_vad_tt3`
```python
def predict_vad_tt3(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float = 0.002864
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # //TODO: Run full VAD pipeline with Gaussian Bayes threshold, return boundaries and STE curve
    print("predict_vad_tt3")
    ...
```

---

## 5. Bảng Đối Sánh Tổng Hợp Toàn Đồ Án (Benchmark)

Chạy lệnh kiểm thử mặc định:
```bash
uv run python main.py --algo tt3 --no-plot
```

Bảng tổng hợp kết quả của 3 thành viên trên tập kiểm thử (`TinHieuKiemThu`):

| Tệp Kiểm Thử | Ground-Truth | TT1 (Hodgkinson, $T=0.0025$) | TT2 (Histogram, $W=5$) | TT3 (Gaussian Bayes, $T=0.00286$) |
|---|---|---|---|---|
| `phone_F2.wav` | $[1.02, 4.04]$ | $[1.02, 4.08]$ (MAE: **$20.0\text{ ms}$**) | $[1.11, 4.01]$ (MAE: $60.0\text{ ms}$) | $[1.02, 4.08]$ (MAE: **$20.0\text{ ms}$**) |
| `phone_M2.wav` | $[0.53, 2.52]$ | $[0.53, 2.52]$ (MAE: **$0.0\text{ ms}$**)  | $[0.54, 2.51]$ (MAE: $10.0\text{ ms}$) | $[0.53, 2.52]$ (MAE: **$0.0\text{ ms}$**)  |
| `studio_F2.wav`| $[0.77, 2.37]$ | $[0.76, 2.36]$ (MAE: **$10.0\text{ ms}$**) | $[0.77, 2.21]$ (MAE: $80.0\text{ ms}$) | $[0.76, 2.36]$ (MAE: **$10.0\text{ ms}$**) |
| `studio_M2.wav`| $[0.45, 1.93]$ | $[0.46, 1.93]$ (MAE: **$5.0\text{ ms}$**)  | $[0.47, 1.89]$ (MAE: $30.0\text{ ms}$) | $[0.46, 1.93]$ (MAE: **$5.0\text{ ms}$**)  |
| **TRUNG BÌNH** | — | **MAE: $8.75\text{ ms}$** | MAE: $45.00\text{ ms}$ | **MAE: $8.75\text{ ms}$** |
| **RMSE TB**    | — | **RMSE: $11.34\text{ ms}$**| RMSE: $55.46\text{ ms}$ | **RMSE: $11.34\text{ ms}$** |

---

## 6. Cấu Trúc Slide Thuyết Trình (Đúng 3 Phút) Dành Cho SV 3

* **Slide 1: Trang bìa (15 giây)**
  - Tên đề tài: *Phân đoạn Tiếng nói / Khoảng lặng bằng Mô hình Phân bố Gauss Bayes (Gaussian Bayes VAD)*.
  - Họ tên, MSSV: Sinh viên 3 (Trưởng nhóm).
  - Lớp học phần: Xử lý tín hiệu số (2026).

* **Slide 2: Mô hình Thống kê Xác suất & Giải phương trình Bayes (60 giây)**
  - Đồ thị 2 hàm mật độ phân bố Gauss $p(x|\text{sil})$ và $p(x|\text{sp})$ cắt nhau.
  - Bảng tham số thống kê trích xuất từ 4 file huấn luyện: $(\mu_{sil}, \sigma_{sil})$ và $(\mu_{sp}, \sigma_{sp})$.
  - Phương trình bậc hai tìm nghiệm điểm cắt Bayes: $T_{Bayes} \approx 0.002864$.
  - Đối chiếu: So sánh vì sao $T_{equal} = 0.00096$ thất bại còn $T_{Bayes}$ đạt đỉnh cao.

* **Slide 3: Kết quả thực nghiệm & Bảng đối sánh Benchmark 3 thuật toán (60 giây)**
  - Bảng đối sánh đầy đủ cả 3 thuật toán trên 4 file kiểm thử.
  - Đồ thị 4 góc màn hình với các đường vạch xanh/đỏ chuẩn xác.
  - Nhấn mạnh: TT3 và TT1 đạt độ chính xác tương đồng cao nhất ($\text{MAE} = 8.75\text{ ms}$).

* **Slide 4: Tổng kết Đồ án & Đánh giá Kiến trúc Hệ thống (45 giây)**
  - Đánh giá kiến trúc mô-đun dùng chung: Chạy 01 lần duy nhất, chia 4 cửa sổ tự động.
  - Phân tích ảnh hưởng của SNR: Môi trường `studio` ($\text{SNR} > 37\text{ dB}$) cho độ lệch $\le 10\text{ ms}$, môi trường `phone` ($\text{SNR} \approx 23 - 27\text{ dB}$) chịu tác động của âm thở nhẹ cuối câu.

---

## 7. Kịch Bản Vấn Đáp Bảo Vệ Đồ Án (Dành Riêng Cho SV 3 & Trưởng Nhóm)

1. **Câu hỏi**: *Tại sao điểm cân bằng Z-score ($T_{equal}$) lại thất bại nặng nề trên dữ liệu phone, trong khi Bayes lại thành công?*
   - **Trả lời**: *Thưa thầy/cô, độ lệch chuẩn của khoảng lặng ($\sigma_{sil} \approx 0.0007$) nhỏ hơn độ lệch chuẩn của tiếng nói ($\sigma_{sp} \approx 0.233$) tới hơn 300 lần. Công thức Z-score chỉ lấy trung bình có trọng số theo độ lệch chuẩn tuyến tính nên bị kéo sát về 0 ($T_{equal} = 0.00096$). Khi vào môi trường phone có công suất nhiễu xấp xỉ $0.001$, toàn bộ nhiễu bị nhận nhầm thành tiếng nói. Ngược lại, tiêu chuẩn Bayes xét trên hàm phân bố xác suất logarithm bậc hai, tìm đúng điểm mà xác suất lỗi tổng thể đạt cực tiểu, từ đó sinh ra ngưỡng $0.00286$ hoàn toàn vượt qua sàn nhiễu điện thoại.*

2. **Câu hỏi**: *Với vai trò Pipeline Manager, làm thế nào để đảm bảo 3 thành viên làm việc độc lập mà không gây xung đột mã nguồn?*
   - **Trả lời**: *Thưa thầy/cô, em đã chia tách kiến trúc phần mềm thành tầng `core/` dùng chung (đảm bảo tính nhất quán về chuẩn hóa dữ liệu, kích thước khung 20ms/10ms, bộ lọc 200ms và phép đo MAE/RMSE) và thư mục `algorithms/` độc lập cho từng thành viên. Chương trình `main.py` đóng vai trò điều phối trung tâm, nạp các thuật toán qua interface đồng nhất, tự động định vị 4 Figure tại 4 góc màn hình mà không can thiệp vào mã logic bên trong của từng bạn.*

3. **Câu hỏi**: *Quy cách đóng gói nộp bài thi của nhóm như thế nào?*
   - **Trả lời**: *Thưa thầy/cô, toàn bộ mã nguồn chương trình và các slide thuyết trình định dạng PDF được đóng gói vào thư mục `MaTheSV-HoTen`. Theo đúng hướng dẫn thi cử, nhóm tuyệt đối không nộp kèm các tệp âm thanh `*.wav` để tối ưu dung lượng tệp nộp.*
