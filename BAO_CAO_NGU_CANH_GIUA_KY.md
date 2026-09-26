# BÁO CÁO TOÀN DIỆN VỀ NGỮ CẢNH & HƯỚNG DẪN THỰC HIỆN BÀI TẬP LỚN GIỮA KỲ

**Học phần**: Xử lý tín hiệu số (XLTHS - 2026)
**Đề tài**: **Hướng dẫn BT 1 - Phân đoạn tín hiệu thành tiếng nói và khoảng lặng**
*(Voice Activity Detection - VAD / Speech-Silence Discrimination)*

---

## MỤC LỤC

1. [Bản chất bài toán &amp; Mục tiêu cốt lõi](#1-bản-chất-bài-toán--mục-tiêu-cốt-lõi)
2. [Khảo sát dữ liệu thực nghiệm &amp; Phân tích Ground Truth](#2-khảo-sát-dữ-liệu-thực-nghiệm--phân-tích-ground-truth)
3. [Ba thuật toán trọng tâm &amp; Cơ sở lý thuyết toán học](#3-ba-thuật-toán-trọng-tâm--cơ-sở-lý-thuyết-toán-học)
4. [Bảng đối sánh thực nghiệm định lượng (Benchmark)](#4-bảng-đối-sánh-thực-nghiệm-định-lượng-benchmark)
5. [Quy định kỹ thuật nghiêm ngặt về Code, Demo &amp; Slide](#5-quy-định-kỹ-thuật-nghiêm-ngặt-về-code-demo--slide)
6. [Thiết kế kiến trúc phần mềm đề xuất](#6-thiết-kế-kiến-trúc-phần-mềm-đề-xuất)
7. [Kế hoạch hành động &amp; Phân công công việc nhóm](#7-kế-hoạch-hành-động--phân-công-công-việc-nhóm)

---

## 1. Bản Chất Bài Toán & Mục Tiêu Cốt Lõi

### 1.1 Mục tiêu đồ án

* **Bài toán VAD (Voice Activity Detection)**: Tự động phân đoạn tín hiệu thu âm thành 2 trạng thái:
  1. **Khoảng lặng / Nhiễu nền (`silence` - `sil`)**: Phần không có tiếng nói của người, chỉ chứa nhiễu thiết bị hoặc môi trường.
  2. **Tiếng nói (`speech`)**: Bao gồm cả âm hữu thanh (Voiced - `v`) và âm vô thanh (Unvoiced - `uv`).
* **Đầu ra mong muốn**: Tìm 2 mốc thời gian:
  * $\hat{T}_{start}$: Thời điểm bắt đầu câu nói (giây).
  * $\hat{T}_{end}$: Thời điểm kết thúc câu nói (giây).
* **Độ đo đánh giá chất lượng**:
  * **Sai số tuyệt đối trung bình (MAE)** tính bằng miligiây (ms):
    $$
    \text{MAE} = \frac{|\hat{T}_{start} - T_{start}| + |\hat{T}_{end} - T_{end}|}{2}
    $$
  * **Căn bậc hai sai số bình phương trung bình (RMSE)** tính bằng miligiây (ms):
    $$
    \text{RMSE} = \sqrt{\frac{(\hat{T}_{start} - T_{start})^2 + (\hat{T}_{end} - T_{end})^2}{2}}
    $$

### 1.2 Lưu ý cốt tử: Vấn đề đường Pitch $F_0$

> [!IMPORTANT]
> **Khẳng định dứt khoát**: Đề tài VAD (BT 1) **HOÀN TOÀN KHÔNG CẦN TÍNH TOÁN HAY VẼ ĐƯỜNG PITCH $F_0$**.
>
> - File `Hướng dẫn trình bày slide và nộp bài thi.pdf` có nhắc tới cụm từ *"đường F0 của tín hiệu"* vì đây là tài liệu khung áp dụng chung cho nhiều bài tập (bao gồm cả BT 2 về Pitch Detection).
> - File `Hướng dẫn BT 1 - Phân đoạn tín hiệu thành tiếng nói và khoảng lặng_XLTHS_GK 2026.docx` chỉ yêu cầu vẽ dạng sóng, đặc trưng năng lượng STE và các vạch biên phân đoạn.
> - Hai dòng cuối `F0mean` và `F0std` trong các file `.lab` chỉ là dữ liệu sinh tự động từ phần mềm gán nhãn Praat, không liên quan đến bài VAD.

---

## 2. Khảo Sát Dữ Liệu Thực Nghiệm & Phân Tích Ground Truth

### 2.1 Cấu trúc file nhãn chuẩn (`*.lab`)

Định dạng từng dòng theo chuẩn Praat: `<thời_điểm_đầu> <thời_điểm_cuối> <nhãn>`

* `sil`: Khoảng lặng (Silence).
* `v`: Âm hữu thanh (Voiced speech) — nguyên âm, phụ âm hữu thanh có tính tuần hoàn, năng lượng lớn.
* `uv`: Âm vô thanh (Unvoiced speech) — các phụ âm bật hơi, âm xát (/s/, /t/, /k/...) năng lượng rất yếu.

**Quy luật bất biến (Invariant)**: Mọi file đều có cấu trúc:

$$
\text{Khoảng lặng đầu } [0, T_{start}] \longrightarrow \text{Chuỗi tiếng nói liên tục } [T_{start}, T_{end}] \longrightarrow \text{Khoảng lặng đuôi } [T_{end}, T_{duration}]
$$

Không có khoảng lặng thực nào ở giữa câu nói. Đầu câu và cuối câu thường mở đầu/kết thúc bằng âm vô thanh (`uv`).

### 2.2 Thống kê chi tiết 8 file âm thanh & Tỉ số tín hiệu trên nhiễu (SNR)

Âm thanh chuẩn PCM 16-bit Mono:

| Tập dữ liệu                                | Tệp âm thanh    | Tần số$f_s$     | Thời lượng     | Biên Ground-Truth$[T_{start}, T_{end}]$ | Công suất nhiễu$P_{sil}$ | Công suất tiếng nói$P_{sp}$                       | SNR (dB)                                  | Đặc điểm |
| --------------------------------------------- | ----------------- | ------------------- | ----------------- | ------------------------------------------ | ----------------------------- | ------------------------------------------------------- | ----------------------------------------- | ------------ |
| **Huấn luyện** (`TinHieuHuanLuyen`) | `phone_F1.wav`  | $16000\text{ Hz}$ | $3.24\text{ s}$ | $[0.53\text{ s}, 2.75\text{ s}]$         | $9.14 \times 10^{-5}$       | $2.18 \times 10^{-2}$ | **$23.77\text{ dB}$** | Nhiễu nền cao, có tạp âm             |              |
|                                               | `phone_M1.wav`  | $16000\text{ Hz}$ | $4.16\text{ s}$ | $[0.46\text{ s}, 3.52\text{ s}]$         | $1.23 \times 10^{-5}$       | $2.61 \times 10^{-3}$ | **$23.28\text{ dB}$** | Nhiễu nền cao, tiếng vang nhẹ         |              |
|                                               | `studio_F1.wav` | $44100\text{ Hz}$ | $2.86\text{ s}$ | $[0.68\text{ s}, 2.15\text{ s}]$         | $7.60 \times 10^{-7}$       | $5.75 \times 10^{-3}$ | **$38.77\text{ dB}$** | Rất sạch, phòng thu chuẩn             |              |
|                                               | `studio_M1.wav` | $44100\text{ Hz}$ | $2.73\text{ s}$ | $[0.87\text{ s}, 2.06\text{ s}]$         | $5.90 \times 10^{-7}$       | $3.54 \times 10^{-3}$ | **$37.80\text{ dB}$** | Rất sạch, bắt đầu bằng`uv`        |              |
| **Kiểm thử** (`TinHieuKiemThu`)     | `phone_F2.wav`  | $16000\text{ Hz}$ | $4.80\text{ s}$ | $[1.02\text{ s}, 4.04\text{ s}]$         | $3.83 \times 10^{-6}$       | $1.15 \times 10^{-3}$ | **$24.77\text{ dB}$** | Đuôi câu là`uv` năng lượng thấp |              |
|                                               | `phone_M2.wav`  | $16000\text{ Hz}$ | $2.80\text{ s}$ | $[0.53\text{ s}, 2.52\text{ s}]$         | $4.69 \times 10^{-6}$       | $2.35 \times 10^{-3}$ | **$27.00\text{ dB}$** | Tương đối rõ ràng                   |              |
|                                               | `studio_F2.wav` | $44100\text{ Hz}$ | $3.15\text{ s}$ | $[0.77\text{ s}, 2.37\text{ s}]$         | $2.50 \times 10^{-7}$       | $2.10 \times 10^{-2}$ | **$49.30\text{ dB}$** | Cực sạch, âm lượng lớn              |              |
|                                               | `studio_M2.wav` | $44100\text{ Hz}$ | $2.38\text{ s}$ | $[0.45\text{ s}, 1.93\text{ s}]$         | $7.60 \times 10^{-7}$       | $4.43 \times 10^{-3}$ | **$37.67\text{ dB}$** | Bắt đầu bằng`uv` ngắn (30ms)       |              |

> [!NOTE]
> Mức công suất nhiễu của môi trường `phone` lớn gấp **15 đến 150 lần** môi trường `studio`. Đây là nguyên nhân cốt lõi khiến các thuật toán chọn ngưỡng dễ bị nhận diện sai ở tín hiệu điện thoại.

---

## 3. Ba Thuật Toán Trọng Tâm & Cơ Sở Lý Thuyết Toán Học

```
                        ┌──────────────────────────────────────────────┐
                        │              TÍN HIỆU ĐẦU VÀO                │
                        └──────────────────────┬───────────────────────┘
                                               │
                                ┌──────────────▼─────────────┐
                                │   Framing: 20 ms / 10 ms   │
                                │    Short-Time Energy (STE) │
                                └──────────────┬─────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
      ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
      │  Sinh viên 1    │             │  Sinh viên 2    │             │  Sinh viên 3    │
      │  Thuật toán TT1 │             │  Thuật toán TT2 │             │  Thuật toán TT3 │
      │(Hodgkinson 2012)│             │(Giannakopoulos) │             │ (Gaussian Bayes)│
      └────────┬────────┘             └────────┬────────┘             └────────┬────────┘
               │ T_opt cố định                 │ T thích nghi từng file        │ T_Bayes thống kê
               │ (0.0025)                      │ qua Histogram                 │ (0.00286)
               └───────────────────────────────┼───────────────────────────────┘
                                               │
                                ┌──────────────▼─────────────┐
                                │       Hậu xử lý VAD        │
                                │ Nối khoảng lặng ảo < 200ms │
                                └──────────────┬─────────────┘
                                               │
                                ┌──────────────▼─────────────┐
                                │   Biên [T_start, T_end]    │
                                │   Đo MAE / RMSE (đơn vị ms)│
                                └────────────────────────────┘
```

### 3.1 Tiền xử lý & Hậu xử lý dùng chung (Tất cả thành viên đều phải cài đặt)

1. **Phân khung tín hiệu (Framing)**:
   * Độ dài khung (Frame size): $20\text{ ms} \Rightarrow N = 0.02 \times f_s$ mẫu.
   * Bước nhảy khung (Hop size / Shift): $10\text{ ms} \Rightarrow H = 0.01 \times f_s$ mẫu.
   * Khung thứ $m$ bắt đầu tại mẫu $m \times H$ và kéo dài $N$ mẫu.
2. **Năng lượng ngắn hạn chuẩn hóa ($STE_{norm}$)**:
   $$
   STE[m] = \sum_{n=0}^{N-1} x^2[m \cdot H + n], \quad STE_{norm}[m] = \frac{STE[m]}{\max_k STE[k]}
   $$
3. **Quy tắc hậu xử lý loại bỏ khoảng lặng ảo (< 200 ms)**:
   * Nếu giữa hai đoạn tiếng nói xuất hiện một khoảng lặng $< 200\text{ ms}$ ($< 20$ frames), bắt buộc phải gộp lại (chuyển thành tiếng nói) để không bị ngắt quãng tại các âm tắc (stop consonants).

---

### 3.2 Thuật toán 1 (TT1): Tối ưu hóa ngưỡng toàn cục (Hodgkinson 2012)

* **Phân công**: Sinh viên 1.
* **Nguyên lý**: Xác định một giá trị ngưỡng năng lượng cố định $T_{opt}$ duy nhất dùng chung cho tất cả các tín hiệu.
* **Thuật toán tìm kiếm**:
  * Hàm mục tiêu: $\mathcal{E}(T) = \frac{1}{4} \sum_{k=1}^4 \text{MAE}_k(T)$ trên 4 file huấn luyện.
  * Quét lưới (Grid Search) hoặc Tìm kiếm tam phân (Ternary Search) trong dải $T \in [0.0001, 0.05]$.
* **Kết quả khảo sát**:
  * Khi $T = 0.0002$: $\text{MAE} = 325.0\text{ ms}$ (Nhiễu nền phone bị nhận nhầm thành tiếng nói).
  * Khi $T \in [0.0022, 0.0026]$: $\text{MAE} = \mathbf{8.75\text{ ms}}$ (Điểm cực tiểu toàn cục).
  * Khi $T = 0.0350$: $\text{MAE} = 17.5\text{ ms}$ (Ngưỡng cao cắt lẹm vào các âm vô thanh `uv`).
  * $\Rightarrow$ **Chọn $T_{opt} = 0.0025$**.

---

### 3.3 Thuật toán 2 (TT2): Ngưỡng tự thích nghi theo Histogram (Giannakopoulos 2014)

* **Phân công**: Sinh viên 2.
* **Nguyên lý**: Mỗi tín hiệu có mức nhiễu khác nhau nên cần tính toán ngưỡng $T$ riêng biệt cho từng file dựa vào lược đồ tần suất của chuỗi $STE_{norm}$.
* **Quy trình 5 bước**:
  1. Trích xuất dãy $STE_{norm}$ của tín hiệu cần phân đoạn.
  2. Tạo biểu đồ tần suất (Histogram) gồm $100$ bins trong khoảng $[0, 1]$.
  3. Làm mịn histogram bằng bộ lọc trung bình trượt (Moving Average với cửa sổ 5 bins).
  4. Xác định 2 đỉnh cực đại địa phương đầu tiên:
     * $M_1$: Đỉnh của khoảng lặng / nhiễu nền (vùng năng lượng thấp).
     * $M_2$: Đỉnh của tiếng nói (vùng năng lượng cao).
  5. Tính ngưỡng thích nghi:
     $$
     T = \frac{W \cdot M_1 + M_2}{W + 1} \quad (\text{với } W = 5)
     $$
* **Nhận xét**: Ngưỡng thích nghi dao động trong khoảng $T \in [0.058, 0.086]$. Do $M_2$ phản ánh đỉnh năng lượng của các nguyên âm mạnh, ngưỡng tính ra tương đối cao, dẫn đến khuynh hướng cắt trễ ở các phụ âm mở đầu và kết thúc câu.

---

### 3.4 Thuật toán 3 (TT3): Phân bố xác suất Gauss Bayes (Gaussian Bayes)

* **Phân công**: Sinh viên 3.
* **Nguyên lý**: Lấy toàn bộ các khung Silence và Speech từ 4 file huấn luyện (dựa theo ground-truth) để ước lượng tham số phân bố Gauss:
  * Tập khung Silence ($499$ frames): $\mu_{sil} = 0.000359, \quad \sigma_{sil} = 0.000715$
  * Tập khung Speech ($795$ frames): $\mu_{sp} = 0.196747, \quad \sigma_{sp} = 0.233472$
* **Cách xác định ngưỡng tối ưu**:
  * *Cách 1 — Điểm cân bằng Z-score* ($T_{equal} = \frac{\mu_{sil}\sigma_{sp} + \mu_{sp}\sigma_{sil}}{\sigma_{sil} + \sigma_{sp}} \approx 0.000959$): **Thất bại nặng** trên môi trường phone (gây ra $\text{MAE} = 203.75\text{ ms}$ vì ngưỡng sát 0, nhận nhầm nhiễu là tiếng nói).
  * *Cách 2 — Điểm giao xác suất Bayes (Minimum Error Bayes)*: Giải phương trình $p(x|\text{sil}) = p(x|\text{sp})$:
    $$
    T_{Bayes} \approx \mathbf{0.002864}
    $$
  * Giá trị lý thuyết $T_{Bayes} \approx 0.00286$ gần như trùng khít với giá trị thực nghiệm $T_{opt} = 0.0025$ của TT1!

---

## 4. Bảng Đối Sánh Thực Nghiệm Định Lượng (Benchmark)

Kiểm thử độc lập cả 3 thuật toán trên 4 file của tập kiểm thử (`TinHieuKiemThu`):

| Tệp Kiểm Thử       | Ground-Truth     | TT1 (Hodgkinson,$T=0.0025$)                                                   | TT2 (Histogram$W=5$)                                                | TT3 (Gaussian Bayes$T=0.00286$)                                               |
| --------------------- | ---------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `phone_F2.wav`      | $[1.02, 4.04]$ | $[1.02, 4.08]$$\Delta: (0, 40)\text{ ms}$**MAE: $20.0\text{ ms}$**  | $[1.11, 4.01]$$\Delta: (90, 30)\text{ ms}$MAE: $60.0\text{ ms}$ | $[1.02, 4.08]$$\Delta: (0, 40)\text{ ms}$**MAE: $20.0\text{ ms}$**  |
| `phone_M2.wav`      | $[0.53, 2.52]$ | $[0.53, 2.52]$$\Delta: (0, 0)\text{ ms}$**MAE: $0.0\text{ ms}$**    | $[0.54, 2.51]$$\Delta: (10, 10)\text{ ms}$MAE: $10.0\text{ ms}$ | $[0.53, 2.52]$$\Delta: (0, 0)\text{ ms}$**MAE: $0.0\text{ ms}$**    |
| `studio_F2.wav`     | $[0.77, 2.37]$ | $[0.76, 2.36]$$\Delta: (10, 10)\text{ ms}$**MAE: $10.0\text{ ms}$** | $[0.77, 2.21]$$\Delta: (0, 160)\text{ ms}$MAE: $80.0\text{ ms}$ | $[0.76, 2.36]$$\Delta: (10, 10)\text{ ms}$**MAE: $10.0\text{ ms}$** |
| `studio_M2.wav`     | $[0.45, 1.93]$ | $[0.46, 1.93]$$\Delta: (10, 0)\text{ ms}$**MAE: $5.0\text{ ms}$**   | $[0.47, 1.89]$$\Delta: (20, 40)\text{ ms}$MAE: $30.0\text{ ms}$ | $[0.46, 1.93]$$\Delta: (10, 0)\text{ ms}$**MAE: $5.0\text{ ms}$**   |
| **Trung bình** | —               | **MAE: $8.75\text{ ms}$****RMSE: $11.34\text{ ms}$**            | MAE:$45.00\text{ ms}$RMSE: $55.46\text{ ms}$                      | **MAE: $8.75\text{ ms}$****RMSE: $11.34\text{ ms}$**            |

### Bình luận & Giải thích nguyên nhân sai lệch (Phục vụ phần Vấn đáp):

1. **Ảnh hưởng của môi trường thu âm**:
   * Tín hiệu `studio` có $\text{SNR} > 37\text{ dB}$, biên tìm được sai lệch chỉ $0 - 10\text{ ms}$ (tương đương $\le 1$ bước nhảy khung).
   * Tín hiệu `phone` có $\text{SNR} \approx 23 - 27\text{ dB}$, sàn năng lượng nhiễu dao động lớn khiến việc phân định ranh giới phụ thuộc mạnh vào ngưỡng.
2. **Ảnh hưởng của âm vô thanh (`uv`) ở đầu/cuối**:
   * Ở `phone_F2`, âm vô thanh kết thúc ở $[4.00, 4.04]$ có năng lượng rất thấp. Bộ lọc khoảng lặng $200\text{ ms}$ của TT1 và TT3 đã nối luôn đoạn thở đuôi tạo thành sai lệch $+40\text{ ms}$.
   * Ở TT2 (Histogram), ngưỡng bị kéo lên cao ($0.06 - 0.08$) làm mất các âm xát yếu ở đuôi câu, khiến đoạn kết thúc của `studio_F2` bị cắt sớm tới $160\text{ ms}$.

---

## 5. Quy Định Kỹ Thuật Nghiêm Ngặt Về Code, Demo & Slide

### 5.1 Quy định mã nguồn (Coding Standards)

* **CẤM TUYỆT ĐỐI**: Không sử dụng các thư viện hay toolbox xử lý tín hiệu chuyên dụng (`scipy.signal` lọc/tìm peak, `librosa`, MATLAB Audio Toolbox).
* **ĐƯỢC PHÉP**: Chỉ dùng các hàm toán học cơ bản (`sum`, `mean`, `std`, `max`, `min`, `abs`, các phép tính ma trận cơ bản).
* **Quy tắc chú thích**: Bắt buộc phải có **comment giải thích cho từng khối mã gồm 5 - 10 dòng code**.

### 5.2 Quy định chạy Demo

* Bấm **Run đúng 01 lần duy nhất** trên file chạy chính (`main.py` hoặc `main.m`).
* Chương trình tự động duyệt 4 file kiểm thử và hiển thị đồng thời **4 cửa sổ Figure tại 4 góc màn hình**:
  * **Góc trên - trái**: `phone_F2`
  * **Góc trên - phải**: `phone_M2`
  * **Góc dưới - trái**: `studio_F2`
  * **Góc dưới - phải**: `studio_M2`
* **Nội dung trên mỗi Figure**:
  * Trục hoành: Thời gian $t$ (giây). Trục tung: Biên độ.
  * Dạng sóng tín hiệu $x(n)$.
  * Đường đặc trưng năng lượng ngắn hạn $STE$ chuẩn hóa vẽ đè lên dạng sóng.
  * **Đường kẻ dọc màu đỏ**: Biên phân đoạn chuẩn (Ground-Truth).
  * **Đường kẻ dọc màu xanh**: Biên phân đoạn thuật toán tìm được.
  * Đầy đủ `title`, `xlabel`, `ylabel`, `legend`.

### 5.3 Quy định Slide & Thuyết trình

* **Thời lượng**: Đúng **3 phút trình bày slide + 1 phút chạy demo** cho mỗi sinh viên (quá giờ bị ngắt và trừ điểm).
* **Nội dung slide** (Tối đa 4 - 5 slides):
  * Slide 1: Trang bìa (Tên đề tài/nhiệm vụ, Họ tên, MSSV).
  * Slide 2: Sơ đồ khối (Flowchart) hoặc các bước của thuật toán phụ trách; cách tìm/tính ngưỡng trên tập huấn luyện.
  * Slide 3-4: Kết quả thực nghiệm trên 4 file kiểm thử (hình ảnh 4 đồ thị có vạch xanh/đỏ, bảng số liệu MAE/RMSE), phân tích nguyên nhân vì sao đúng/sai.
  * **TUYỆT ĐỐI KHÔNG**: Trình bày lý thuyết giáo trình, công thức định nghĩa cơ bản.
* **Quy cách trình bày**:
  * $\le 10$ từ/dòng, $\le 7$ dòng/slide.
  * Cỡ chữ $\ge 18\text{ pt}$, màu chữ tương phản rõ rệt với nền slide.

### 5.4 Quy cách nộp bài

* Thư mục nộp: Đặt tên theo cú pháp `MaTheSV-HoTen`.
* Nội dung nộp: File Slide dạng `.pdf` và toàn bộ mã nguồn chương trình.
* **TUYỆT ĐỐI KHÔNG**: Nộp kèm các file âm thanh `*.wav` (tránh dung lượng lớn).

---

## 6. Thiết Kế Kiến Trúc Phần Mềm Đề Xuất

Để các thành viên làm việc nhóm song song mà không bị conflict Git, dự án nên được cấu trúc như sau:

```
DSP_MidTerm/
├── core/
│   ├── __init__.py
│   ├── io_utils.py       # Đọc wav (bằng wave/struct chuẩn), đọc file lab
│   ├── features.py       # Tự code framing (20ms/10ms), tính STE (chỉ dùng toán cơ bản)
│   ├── postprocess.py    # Bộ lọc loại bỏ khoảng lặng ảo < 200ms
│   └── metrics.py        # Tính MAE, RMSE biên thời gian (đơn vị ms)
│
├── algorithms/
│   ├── __init__.py
│   ├── tt1_hodgkinson.py # [SV 1] Tìm kiếm T_opt trên 4 file huấn luyện
│   ├── tt2_histogram.py  # [SV 2] Histogram 100 bins, Moving Average, M1/M2 thích nghi
│   └── tt3_gaussian.py   # [SV 3] Thống kê Gauss và Ngưỡng Bayes
│
├── main.py               # Script chạy chính: hiển thị 4 cửa sổ tại 4 góc màn hình
├── BAO_CAO_NGU_CANH_GIUA_KY.md # Bản tài liệu này
└── slides/               # Thư mục chứa slide báo cáo của từng thành viên
```

---

## 7. Kế Hoạch Hành Động & Phân Công Công Việc Nhóm

### Bước 1: Thống nhất nhóm & Phân chia thuật toán

* Nếu nhóm **3 người**:
  * SV 1 phụ trách **TT1 (Hodgkinson 2012)**.
  * SV 2 phụ trách **TT2 (Giannakopoulos Histogram)**.
  * SV 3 phụ trách **TT3 (Gaussian Bayes)**.
* Nếu nhóm **2 người**: Chọn làm TT1 và TT2 (hoặc TT1 và TT3).

### Bước 2: Hoàn thiện module dùng chung (`core/`)

* Viết và kiểm thử các hàm cơ bản: `read_wav`, `read_lab`, `compute_ste_norm`, `filter_silence_200ms`, `calc_mae_rmse`.
* Đảm bảo viết comment đầy đủ cho từng đoạn 5 - 10 dòng code.

### Bước 3: Triển khai thuật toán riêng & Tối ưu ngưỡng

* Mỗi thành viên tự hoàn thiện file thuật toán của mình trong `algorithms/`.
* Huấn luyện và ghi nhận giá trị ngưỡng từ 4 file trong `TinHieuHuanLuyen/`.

### Bước 4: Tích hợp `main.py` & Test hiển thị 4 góc màn hình

* Chạy thử nghiệm trên tập `TinHieuKiemThu/`.
* Căn chỉnh tọa độ hiển thị 4 Figure sao cho vừa vặn màn hình mà không bị đè lên nhau.

### Bước 5: Làm slide & Tập dượt thuyết trình (3 phút + 1 phút)

* Thiết kế slide đúng chuẩn ($\le 7$ dòng/slide, font $\ge 18\text{ pt}$, tập trung vào kết quả và phân tích).
* Bấm giờ tập nói: Đúng 3 phút slide + 1 phút mở terminal gõ chạy demo.
