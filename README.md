# BÀI TẬP LỚN GIỮA KỲ: PHÂN ĐOẠN TÍN HIỆU TIẾNG NÓI & KHOẢNG LẶNG (VAD)
**Học phần**: Xử lý tín hiệu số (XLTHS - 2026)  
**Nhóm nghiên cứu**: 3 Sinh viên  

---

## 📌 Cấu Trúc Dự Án (Architecture Overview)

Dự án được thiết kế phân tầng khoa học theo Section 6 của [BAO_CAO_NGU_CANH_GIUA_KY.md](BAO_CAO_NGU_CANH_GIUA_KY.md), phân định rạch ròi giữa lõi dùng chung (`core/`), các mô-đun thuật toán độc lập (`algorithms/`) và chương trình điều phối trung tâm (`main.py`):

```
DSP_MidTerm/
├── core/                                   # [Lõi dùng chung]
│   ├── __init__.py
│   ├── io_utils.py                         # Đọc WAV chuẩn bằng wave, phân tích Praat LAB
│   ├── features.py                         # Framing 20ms/10ms, tính STE và chuẩn hóa STE_norm
│   ├── postprocess.py                      # Áp ngưỡng, nối khoảng lặng ảo < 200ms (20 khung)
│   └── metrics.py                          # Tính sai số định lượng MAE và RMSE (đơn vị ms)
│
├── algorithms/                             # [Thuật toán của 3 thành viên]
│   ├── __init__.py
│   ├── tt1_hodgkinson.py                   # [SV 1] Tối ưu hóa ngưỡng toàn cục cố định (T_opt = 0.0025)
│   ├── tt2_histogram.py                    # [SV 2] Ngưỡng tự thích nghi theo Histogram (Giannakopoulos)
│   └── tt3_gaussian.py                     # [SV 3 & PM] Thống kê Gauss & Ngưỡng xác suất Bayes (T = 0.00286)
│
├── docs/                                   # [Tài liệu hướng dẫn chi tiết từng thành viên]
│   ├── HUONG_DAN_SINH_VIEN_1_TT1_HODGKINSON.md     # Hướng dẫn chi tiết cho Sinh viên 1
│   ├── HUONG_DAN_SINH_VIEN_2_TT2_HISTOGRAM.md      # Hướng dẫn chi tiết cho Sinh viên 2
│   └── HUONG_DAN_SINH_VIEN_3_VA_PIPELINE_MANAGER.md# Hướng dẫn cho SV 3 & Trưởng nhóm Pipeline
│
├── main.py                                 # Điều phối chính: chạy 01 lần duy nhất, chia 4 Figure 4 góc
├── pyproject.toml                          # Quản lý môi trường và gói thư viện bằng uv
└── BAO_CAO_NGU_CANH_GIUA_KY.md             # Bản báo cáo nghiên cứu ngữ cảnh toàn diện
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy Bằng `uv`

Tuân thủ nghiêm ngặt quy tắc sử dụng `uv` cho môi trường Python:

### 1. Đồng bộ môi trường ảo
```bash
uv sync
```

### 2. Chạy Demo Đồ Án (01 Lần Chạy Duy Nhất - Hiển Thị 4 Góc Màn Hình)
Mặc định chạy với thuật toán tối ưu TT3 (Gaussian Bayes):
```bash
uv run python main.py
```
*(Chương trình sẽ tự động mở đồng thời 4 cửa sổ đồ thị tại 4 góc màn hình: Top-Left `phone_F2`, Top-Right `phone_M2`, Bottom-Left `studio_F2`, Bottom-Right `studio_M2`).*

### 3. Tùy chọn chạy từng thuật toán riêng biệt
* **Sinh viên 1 (Thuật toán TT1 - Hodgkinson)**:
  ```bash
  uv run python main.py --algo tt1
  ```
* **Sinh viên 2 (Thuật toán TT2 - Histogram)**:
  ```bash
  uv run python main.py --algo tt2
  ```
* **Sinh viên 3 (Thuật toán TT3 - Gaussian Bayes)**:
  ```bash
  uv run python main.py --algo tt3
  ```

### 4. Chế độ kiểm thử tự động (Headless / Không mở giao diện đồ họa)
```bash
uv run python main.py --no-plot
```

### 5. Chế độ huấn luyện tối ưu ngưỡng trên tập huấn luyện (`TinHieuHuanLuyen`)
* **Huấn luyện TT1 (Quét lưới tìm $T_{opt}$)**:
  ```bash
  uv run python main.py --train --algo tt1
  ```
* **Huấn luyện TT3 (Ước lượng Gauss & Giải xác suất Bayes)**:
  ```bash
  uv run python main.py --train --algo tt3
  ```

---

## 📊 Bảng Đối Sánh Thực Nghiệm Định Lượng (Benchmark)

Kết quả kiểm thử trên 4 tệp của tập kiểm thử (`TinHieuKiemThu`):

| Tệp Âm Thanh | Ground-Truth $[T_{start}, T_{end}]$ | TT1 (Hodgkinson $T=0.0025$) | TT2 (Histogram $W=5$) | TT3 (Gaussian Bayes $T=0.00286$) |
|---|---|---|---|---|
| `phone_F2.wav` | $[1.02\text{ s}, 4.04\text{ s}]$ | $[1.02\text{ s}, 4.08\text{ s}]$ (MAE: **$20.0\text{ ms}$**) | $[1.11\text{ s}, 4.01\text{ s}]$ (MAE: $60.0\text{ ms}$) | $[1.02\text{ s}, 4.08\text{ s}]$ (MAE: **$20.0\text{ ms}$**) |
| `phone_M2.wav` | $[0.53\text{ s}, 2.52\text{ s}]$ | $[0.53\text{ s}, 2.52\text{ s}]$ (MAE: **$0.0\text{ ms}$**)  | $[0.54\text{ s}, 2.51\text{ s}]$ (MAE: $10.0\text{ ms}$) | $[0.53\text{ s}, 2.52\text{ s}]$ (MAE: **$0.0\text{ ms}$**)  |
| `studio_F2.wav`| $[0.77\text{ s}, 2.37\text{ s}]$ | $[0.76\text{ s}, 2.36\text{ s}]$ (MAE: **$10.0\text{ ms}$**) | $[0.77\text{ s}, 2.21\text{ s}]$ (MAE: $80.0\text{ ms}$) | $[0.76\text{ s}, 2.36\text{ s}]$ (MAE: **$10.0\text{ ms}$**) |
| `studio_M2.wav`| $[0.45\text{ s}, 1.93\text{ s}]$ | $[0.46\text{ s}, 1.93\text{ s}]$ (MAE: **$5.0\text{ ms}$**)  | $[0.47\text{ s}, 1.88\text{ s}]$ (MAE: $35.0\text{ ms}$) | $[0.46\text{ s}, 1.93\text{ s}]$ (MAE: **$5.0\text{ ms}$**)  |
| **TRUNG BÌNH** | — | **MAE: $8.75\text{ ms}$** | MAE: $46.25\text{ ms}$ | **MAE: $8.75\text{ ms}$** |
| **RMSE TB**    | — | **RMSE: $11.34\text{ ms}$**| RMSE: $57.07\text{ ms}$ | **RMSE: $11.34\text{ ms}$** |

---

## 👥 Phân Công Công Việc & Tài Liệu Thành Viên

| Thành Viên | Phụ Trách Thuật Toán | Tệp Mã Nguồn | Tài Liệu Triển Khai | Bản Đặc Tả & Slide Thuyết Trình |
|---|---|---|---|---|
| **Sinh viên 1** | TT1: Hodgkinson 2012 (Ngưỡng toàn cục $T_{opt}$) | `algorithms/tt1_hodgkinson.py` | [HUONG_DAN_SINH_VIEN_1_TT1_HODGKINSON.md](docs/HUONG_DAN_SINH_VIEN_1_TT1_HODGKINSON.md) | [STUDENT_1_HODGKINSON.md](docs/STUDENT_1_HODGKINSON.md) |
| **Sinh viên 2** | TT2: Giannakopoulos 2014 (Ngưỡng thích nghi Histogram) | `algorithms/tt2_histogram.py` | [HUONG_DAN_SINH_VIEN_2_TT2_HISTOGRAM.md](docs/HUONG_DAN_SINH_VIEN_2_TT2_HISTOGRAM.md) | [STUDENT_2_HISTOGRAM.md](docs/STUDENT_2_HISTOGRAM.md) |
| **Sinh viên 3** | TT3: Gaussian Bayes ($T_{Bayes}$) & **Pipeline Manager** | `algorithms/tt3_gaussian.py`, `core/`, `main.py` | [HUONG_DAN_SINH_VIEN_3_VA_PIPELINE_MANAGER.md](docs/HUONG_DAN_SINH_VIEN_3_VA_PIPELINE_MANAGER.md) | [STUDENT_3_GAUSSIAN_PIPELINE.md](docs/STUDENT_3_GAUSSIAN_PIPELINE.md) |
