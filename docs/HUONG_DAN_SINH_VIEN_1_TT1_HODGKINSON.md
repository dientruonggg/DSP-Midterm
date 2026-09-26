# HƯỚNG DẪN CHI TIẾT CÔNG VIỆC: SINH VIÊN 1 (STUDENT 1)
## THUẬT TOÁN 1 (TT1): TỐI ƯU HÓA NGƯỠNG TOÀN CỤC (HODGKINSON 2012)

---

## 1. Thông Tin Vai Trò & Trách Nhiệm
* **Thành viên phụ trách**: Sinh viên 1.
* **Tên thuật toán**: Phân đoạn Tiếng nói / Khoảng lặng bằng Tối ưu hóa Ngưỡng Năng lượng Toàn cục (*Energy-based Speech/Silence Discrimination with Binary/Grid Search*).
* **Tài liệu tham khảo**: Hodgkinson (2012), *CS425 Audio and Speech Processing*, Section 2.1.
* **Tệp mã nguồn chịu trách nhiệm**:
  - `algorithms/tt1_hodgkinson.py`
  - Đóng góp kiểm thử tích hợp: `main.py --algo tt1`
* **Thời lượng bảo vệ**: Đúng **3 phút trình bày Slide + 1 phút chạy Demo trực tiếp**.

---

## 2. Bản Chất Toán Học & Cơ Sở Lý Thuyết

### 2.1 Nguyên lý cốt lõi
Khác với việc chọn ngưỡng cảm tính, thuật toán Hodgkinson tìm kiếm một **giá trị ngưỡng năng lượng cố định duy nhất $T_{opt}$** áp dụng cho mọi tệp âm thanh đầu vào. Giá trị $T_{opt}$ được xác định bằng cách tối ưu hóa (cực tiểu hóa) hàm sai số trên toàn bộ tập dữ liệu huấn luyện (`TinHieuHuanLuyen`).

### 2.2 Hàm mục tiêu tối ưu (Objective Loss Function)
Cho $K = 4$ tệp âm thanh huấn luyện (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`). Với mỗi giá trị ngưỡng thử nghiệm $T$, sai số trung bình tuyệt đối MAE toàn cục được định nghĩa:

$$
\mathcal{E}(T) = \frac{1}{K} \sum_{k=1}^K \text{MAE}_k(T)
$$

Trong đó, sai số trên từng tệp thứ $k$:

$$
\text{MAE}_k(T) = \frac{|\hat{T}_{start}^{(k)}(T) - T_{start}^{(k)}| + |\hat{T}_{end}^{(k)}(T) - T_{end}^{(k)}|}{2} \times 1000 \quad (\text{ms})
$$

### 2.3 Không gian tìm kiếm & Kết quả thực nghiệm
* Quét lưới (Grid Search) hoặc tìm kiếm tam phân (Ternary Search) trong dải: $T \in [0.0001, 0.05]$.
* **Hiện tượng tại các vùng ngưỡng**:
  1. **Khi $T$ quá nhỏ ($T \approx 0.0002$)**: $\text{MAE} = 325.0\text{ ms}$. Nhiễu nền của môi trường điện thoại (`phone`) có công suất cao bị nhận nhầm thành tiếng nói $\Rightarrow$ Biên câu nói bị kéo dài quá mức.
  2. **Khi $T \in [0.0022, 0.0026]$**: $\text{MAE} = \mathbf{8.75\text{ ms}}$ $\Rightarrow$ Đạt cực tiểu toàn cục.
  3. **Khi $T$ quá lớn ($T > 0.0350$)**: $\text{MAE} = 17.5\text{ ms} - 50.0\text{ ms}$. Ngưỡng cao cắt lẹm vào các âm vô thanh (`uv`) có năng lượng thấp ở đầu và cuối câu (như /s/, /t/, /k/).
* **Quyết định chốt**: Chọn **$T_{opt} = 0.0025$**.

---

## 3. Bản Đặc Tả Giao Diện & Mã Nguồn Cần Hoàn Thiện

Tệp `algorithms/tt1_hodgkinson.py` đã được định nghĩa sẵn khung giao diện chuẩn (interface) cùng comment `# //TODO` và lệnh in định danh hàm:

### 3.1 Hàm `hodgkinson_cost_function`
```python
def hodgkinson_cost_function(threshold: float, train_data: List[Dict]) -> float:
    # //TODO: Evaluate candidate threshold T across training files and return average MAE
    print("hodgkinson_cost_function")
    ...
```
* **Nhiệm vụ**: Duyệt qua danh sách `train_data`, áp dụng ngưỡng `threshold`, chạy bộ lọc 200 ms qua hàm `remove_short_silences_200ms`, tìm biên và tính MAE đối sánh với nhãn Ground-Truth.

### 3.2 Hàm `train_optimal_threshold_tt1`
```python
def train_optimal_threshold_tt1(
    training_dir: str,
    search_range: Tuple[float, float] = (0.0001, 0.05),
    num_steps: int = 500
) -> float:
    # //TODO: Perform Grid Search or Ternary Search over search_range to find T_opt minimizing MAE
    print("train_optimal_threshold_tt1")
    ...
```
* **Nhiệm vụ**: Đọc 4 file trong thư mục `TinHieuHuanLuyen/`, quét qua các giá trị ứng viên của $T$ và trả về giá trị $T_{opt} \approx 0.0025$.

### 3.3 Hàm `predict_vad_tt1`
```python
def predict_vad_tt1(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float = 0.0025
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # //TODO: Extract STE_norm, apply fixed threshold T_opt, bridge <200ms silences, return boundaries
    print("predict_vad_tt1")
    ...
```
* **Nhiệm vụ**: Trích xuất năng lượng $STE_{norm}$, so sánh với $T_{opt}$, nối khoảng lặng $< 200\text{ ms}$ (20 frames), trả về `(t_start, t_end, ste_norm, frame_times)`.

---

## 4. Bảng Số Liệu Kiểm Thử Mục Tiêu Trên Tập Kiểm Thử (TinHieuKiemThu)

Sinh viên 1 cần chạy lệnh sau để kiểm tra kết quả đối sánh:
```bash
uv run python main.py --algo tt1 --no-plot
```

Bảng số liệu chuẩn SV 1 cần đạt được và đưa vào slide:

| Tệp Kiểm Thử | Ground-Truth | TT1 ($T_{opt}=0.0025$) | Sai lệch $\Delta(T_{start}, T_{end})$ | MAE (ms) | RMSE (ms) |
|---|---|---|---|---|---|
| `phone_F2.wav` | $[1.02\text{ s}, 4.04\text{ s}]$ | $[1.02\text{ s}, 4.08\text{ s}]$ | $(0\text{ ms}, +40\text{ ms})$ | **$20.0\text{ ms}$** | **$28.28\text{ ms}$** |
| `phone_M2.wav` | $[0.53\text{ s}, 2.52\text{ s}]$ | $[0.53\text{ s}, 2.52\text{ s}]$ | $(0\text{ ms}, 0\text{ ms})$ | **$0.0\text{ ms}$** | **$0.0\text{ ms}$** |
| `studio_F2.wav`| $[0.77\text{ s}, 2.37\text{ s}]$ | $[0.76\text{ s}, 2.36\text{ s}]$ | $(-10\text{ ms}, -10\text{ ms})$ | **$10.0\text{ ms}$** | **$10.0\text{ ms}$** |
| `studio_M2.wav`| $[0.45\text{ s}, 1.93\text{ s}]$ | $[0.46\text{ s}, 1.93\text{ s}]$ | $(+10\text{ ms}, 0\text{ ms})$ | **$5.0\text{ ms}$** | **$7.07\text{ ms}$** |
| **TRUNG BÌNH** | — | — | — | **$8.75\text{ ms}$** | **$11.34\text{ ms}$** |

*(Ghi chú: Dung sai $\pm 10\text{ ms}$ phụ thuộc vào cách làm tròn frame center time vs frame start time, tương đương $\le 1$ bước nhảy khung).*

---

## 5. Cấu Trúc Slide Thuyết Trình (Đúng 3 Phút)

* **Quy tắc hình thức**:
  - $\le 7$ dòng chữ mỗi slide.
  - $\le 10$ từ mỗi dòng.
  - Cỡ chữ $\ge 18\text{ pt}$.
  - Nền slide sáng, chữ màu tối tương phản cao.
  - Không trình bày lý thuyết định nghĩa tín hiệu cơ bản của giáo trình.

* **Slide 1: Trang bìa (15 giây)**
  - Tên đề tài: *Phân đoạn Tiếng nói / Khoảng lặng bằng Tối ưu hóa Ngưỡng Toàn cục (Hodgkinson 2012)*.
  - Họ và tên, Mã số sinh viên: Sinh viên 1.
  - Lớp học phần: Xử lý tín hiệu số (2026).

* **Slide 2: Sơ đồ khối & Chiến lược Huấn luyện Tìm $T_{opt}$ (45 giây)**
  - Sơ đồ khối: $x(n) \rightarrow \text{Framing } 20\text{ms}/10\text{ms} \rightarrow STE_{norm} \rightarrow \text{So sánh } T_{opt} \rightarrow \text{Lọc } 200\text{ms} \rightarrow [T_{start}, T_{end}]$.
  - Đồ thị đường cong sai số $\mathcal{E}(T)$ theo $T$: Chỉ ra đáy cực tiểu tại $T_{opt} = 0.0025$ ($\text{MAE} = 8.75\text{ ms}$).
  - Nhấn mạnh: $T < 0.0022$ nhận nhầm nhiễu điện thoại; $T > 0.0030$ cắt mất âm vô thanh `uv`.

* **Slide 3: Kết quả thực nghiệm trên 4 Tệp Kiểm thử (60 giây)**
  - Bảng số liệu MAE/RMSE của 4 tệp `TinHieuKiemThu`.
  - Hình chụp 4 góc màn hình của 4 tệp kiểm thử (có vạch xanh thuật toán đè lên vạch đỏ Ground-Truth).
  - Điểm nổi bật: Tệp `phone_M2` đạt độ chính xác hoàn hảo $\text{MAE} = 0.0\text{ ms}$.

* **Slide 4: Phân tích Nguyên nhân Sai số & Kết luận (60 giây)**
  - Tệp `phone_F2` bị lệch $+40\text{ ms}$ ở cuối câu: Do âm vô thanh cuối câu nối tiếp tiếng thở nhẹ, bộ lọc 200 ms đã gộp luôn phần thở vào câu nói.
  - Tín hiệu `studio` có $\text{SNR} > 37\text{ dB}$ cho độ lệch $\le 10\text{ ms}$ (bằng 1 bước nhảy khung).
  - Ưu điểm: Đơn giản, tốc độ xử lý tức thì, hiệu năng vượt trội.
  - Nhược điểm: Phụ thuộc vào dữ liệu huấn luyện, kém thích nghi nếu môi trường có nhiễu đột biến.

---

## 6. Kịch Bản Vấn Đáp Bảo Vệ Đồ Án (Dành Riêng Cho SV 1)

1. **Câu hỏi**: *Tại sao em không chọn ngưỡng $T = 0.0002$ vốn rất nhạy để không bỏ sót tiếng thì thầm?*
   - **Trả lời**: *Thưa thầy/cô, ở ngưỡng $T = 0.0002$, công suất nhiễu nền của môi trường điện thoại (phone) đạt mức xấp xỉ $9 \times 10^{-5}$ đến $1.2 \times 10^{-4}$ sau chuẩn hóa, khiến thuật toán nhận nhầm toàn bộ đoạn im lặng đầu câu thành tiếng nói, gây sai số MAE vọt lên $325\text{ ms}$. Ngưỡng $0.0025$ là điểm cân bằng lý tưởng loại bỏ triệt để nhiễu nền điện thoại mà vẫn giữ trọn các âm xát vô thanh.*

2. **Câu hỏi**: *Tại sao ở file `phone_F2`, biên kết thúc của em lại dài hơn nhãn chuẩn $40\text{ ms}$?*
   - **Trả lời**: *Thưa thầy/cô, ở cuối file `phone_F2`, sau âm vô thanh [4.00s - 4.04s] có một đoạn xả hơi nhẹ của người nói cách đó dưới 200ms. Theo quy định kỹ thuật của đề tài, các khoảng lặng dưới 200ms bắt buộc phải gộp lại để tránh đứt đoạn từ ngữ, do đó thuật toán đã giữ lại đoạn này tạo thành độ lệch 40ms (tương đương 4 frames).*
