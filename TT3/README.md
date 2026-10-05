# THUẬT TOÁN 3 (TT3) - GAUSSIAN BAYES VOICE ACTIVITY DETECTION (VAD)

Thư mục triển khai toàn diện giải pháp Thuật toán 3 (TT3) phân đoạn tín hiệu tiếng nói và khoảng lặng dựa trên phân bố Gauss của Năng lượng ngắn hạn (Short-Time Energy - STE) chuẩn hóa.

---

## 1. Cấu trúc thư mục

```text
TT3/
├── src/                          # Toàn bộ mã nguồn Python tự code (không dùng toolbox ngoài)
│   ├── __init__.py
│   ├── io_utils.py               # Đọc file âm thanh WAV PCM 16-bit và nhãn Praat .lab
│   ├── features.py               # Phân khung (25 ms, hop 10 ms), tính STE chuẩn hóa [0, 1]
│   ├── gaussian_vad.py           # Khảo sát dữ liệu, ước lượng Gauss, giải ngưỡng Bayes T_opt, lọc khoảng lặng < 200 ms
│   ├── metrics.py                # Tính sai số định lượng MAE và RMSE (miligiây)
│   ├── visualization.py          # Vẽ đồ thị phân bố Gauss và 04 figure kiểm thử
│   └── main.py                   # Điểm khởi chạy trung tâm của module src
├── notebooks/                    # Thư mục chứa các file Jupyter Notebook (kèm alias tại notebook/)
│   ├── TT3_Gaussian_VAD.ipynb    # Báo cáo thực nghiệm Slide bằng Tiếng Việt (pre-computed)
│   ├── TT3_Gaussian_VAD_EN.ipynb # Báo cáo thực nghiệm Slide bằng Tiếng Anh (pre-computed)
│   └── TT3.ipynb                 # Bản sao tham chiếu chuẩn
├── output/                       # Các biểu đồ và figure kết quả phân đoạn
│   ├── gaussian_distributions_tt3.png # Đồ thị phân bố xác suất Gauss và ngưỡng Bayes
│   ├── phone_F2_vad.png          # Đồ thị Waveform + STE + mốc biên file phone_F2
│   ├── phone_M2_vad.png          # Đồ thị Waveform + STE + mốc biên file phone_M2
│   ├── studio_F2_vad.png         # Đồ thị Waveform + STE + mốc biên file studio_F2
│   └── studio_M2_vad.png         # Đồ thị Waveform + STE + mốc biên file studio_M2
├── main.py                       # Script chuyển tiếp cho phép chạy trực tiếp từ thư mục gốc TT3
└── build_notebook.py             # Script tự động tạo và thực thi 2 file notebook (VI & EN)
```

---

## 2. Thông số kỹ thuật & Bộ tham số tối ưu

- **Độ dài khung (Frame size)**: $25.0\text{ ms}$
- **Bước nhảy khung (Hop size)**: $10.0\text{ ms}$
- **Khoảng lặng tối thiểu (Minimum silence)**: $200.0\text{ ms}$ (loại bỏ khoảng lặng ảo ngắn hơn 20 khung)
- **Tham số Gauss khảo sát từ 04 file huấn luyện (`TinHieuHuanLuyen`)**:
  - Khoảng lặng (Silence): $\mu_{sil} = 0.000387$, $\sigma_{sil} = 0.000709$
  - Tiếng nói (Speech): $\mu_{sp} = 0.202649$, $\sigma_{sp} = 0.235626$
- **Ngưỡng tối ưu Bayes tìm được**:
  $$\mathbf{T_{opt} = 0.002878}$$

---

## 3. Kết quả thực nghiệm trên 04 tín hiệu kiểm thử (`TinHieuKiemThu`)

| File kiểm thử | Ground Truth [s] | Thuật toán TT3 [s] | $\Delta T_{start}$ | $\Delta T_{end}$ | MAE (ms) | RMSE (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `phone_F2.wav` | $[1.02, 4.04]$ | $[1.01, 4.08]$ | $10.0\text{ ms}$ | $45.0\text{ ms}$ | **$27.5\text{ ms}$** | **$32.6\text{ ms}$** |
| `phone_M2.wav` | $[0.53, 2.52]$ | $[0.52, 2.52]$ | $10.0\text{ ms}$ | $5.0\text{ ms}$ | **$7.5\text{ ms}$** | **$7.9\text{ ms}$** |
| `studio_F2.wav` | $[0.77, 2.37]$ | $[0.76, 2.37]$ | $10.0\text{ ms}$ | $5.0\text{ ms}$ | **$7.5\text{ ms}$** | **$7.9\text{ ms}$** |
| `studio_M2.wav` | $[0.45, 1.93]$ | $[0.46, 1.94]$ | $10.0\text{ ms}$ | $5.0\text{ ms}$ | **$7.5\text{ ms}$** | **$7.9\text{ ms}$** |
| **TRUNG BÌNH TOÀN TẬP** | — | — | **$10.0\text{ ms}$** | **$15.0\text{ ms}$** | **$12.50\text{ ms}$** | **$14.08\text{ ms}$** |

---

## 4. Hướng dẫn chạy chương trình

### Chạy chương trình chính (Main Script):
```bash
# Từ thư mục gốc dự án:
uv run python TT3/main.py

# Hoặc từ bên trong module src:
uv run python TT3/src/main.py
```

### Mở và xem Notebook:
- Phiên bản tiếng Việt: `TT3/notebooks/TT3_Gaussian_VAD.ipynb`
- Phiên bản tiếng Anh: `TT3/notebooks/TT3_Gaussian_VAD_EN.ipynb`
```bash
uv run jupyter lab TT3/notebooks/TT3_Gaussian_VAD.ipynb
```
