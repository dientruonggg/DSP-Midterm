# BÁO CÁO GIỮA KỲ XỬ LÝ TÍN HIỆU SỐ (DSP 2026) — NHÓM 08

### Đề tài: Voice Activity Detection (VAD) — Phân đoạn tín hiệu tiếng nói và khoảng lặng

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Package Manager: uv](https://img.shields.io/badge/Package_Manager-uv-blueviolet?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Course](https://img.shields.io/badge/Course-Digital_Signal_Processing_2026-brightgreen)](https://github.com)
[![Topic](https://img.shields.io/badge/Task-Voice_Activity_Detection-orange)](https://github.com)
[![Evaluation](https://img.shields.io/badge/Benchmark-6_Variants_Evaluated-success)](figures/benchmark_plot.png)
[![Submission Package](https://img.shields.io/badge/Package-08--Ph%C3%A2n%20%C4%91o%E1%BA%A1n%20t%C3%ADn%20hi%E1%BB%87u-red)](08-Ph%C3%A2n%20%C4%91o%E1%BA%A1n%20t%C3%ADn%20hi%E1%BB%87u%20ti%E1%BA%BFng%20n%C3%B3i%20-%20kho%E1%BA%A3ng%20l%E1%BA%B7ng)

---

## 👥 1. Danh sách Thành viên & Phân công Trách nhiệm

Dự án được thực hiện bởi **Nhóm 08**. Mỗi thành viên chịu trách nhiệm nghiên cứu, xây dựng mã nguồn độc lập cho một thuật toán chỉ định cùng các thí nghiệm chuyên sâu mở rộng (DSP Advanced Extensions):

| STT | Họ và tên | Mã số SV | Thuật toán phụ trách | Đóng góp nghiên cứu & Thí nghiệm mở rộng | File Notebook nộp bài |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **1** | **Lương Duy Toàn** | `102240280` | **Thuật toán 1 (TT1)**<br>Năng lượng ngắn hạn (STE) & Tìm kiếm nhị phân | • Cài đặt thuật toán chia đôi tìm kiếm ngưỡng tối ưu toàn cục $T_{\text{opt}}$.<br>• Mở rộng: Bộ phân lớp tuyến tính đa đặc trưng (**DSP Classifier**) kết hợp STE, ZCR, Spectral Flatness. | [`102240280-Luong_Duy_Toan-Binary.ipynb`](08-Phân%20đoạn%20tín%20hiệu%20tiếng%20nói%20-%20khoảng%20lặng/102240280-Luong_Duy_Toan-Binary.ipynb) |
| **2** | **Nguyễn Anh Hoàng** | `102240246` | **Thuật toán 2 (TT2)**<br>Lược đồ thích nghi Histogram (Adaptive Histogram) | • Trích xuất phân bố 2 mode (Bimodal Histogram), làm trơn Moving Average 5-bin, tính ngưỡng thích nghi theo từng câu nói.<br>• Mở rộng: Khảo sát đệm biên (**Boundary Padding Sweep**) và phân đoạn phối hợp Đa đặc trưng (Dual STE + Spectral Centroid). | [`102240246-Nguyen_Anh_Hoang-Histogram.ipynb`](08-Phân%20đoạn%20tín%20hiệu%20tiếng%20nói%20-%20khoảng%20lặng/102240246-Nguyen_Anh_Hoang-Histogram.ipynb) |
| **3** | **Trương Bùi Điền** | `102240237` | **Thuật toán 3 (TT3)**<br>Mô hình phân lớp Bayes tham số Gaussian (Gaussian Bayes) | • Ước lượng tham số hợp lý cực đại ($\mu, \sigma$) cho Silence và Speech từ dữ liệu huấn luyện; giải phương trình bậc hai tìm nghiệm giải tích Bayes $T_{\text{Bayes}}$.<br>• Mở rộng: Nghiên cứu đối sánh mô hình **1D Gaussian vs 4D Multivariate Gaussian** (minh chứng nguyên lý Occam's Razor). | [`102240237-Truong_Bui_Dien-SimpleStatics.ipynb`](08-Phân%20đoạn%20tín%20hiệu%20tiếng%20nói%20-%20khoảng%20lặng/102240237-Truong_Bui_Dien-SimpleStatics.ipynb) |

---

## 🎯 2. Tổng quan Bài toán & Kiến trúc Pipeline Thống nhất

### Mục tiêu bài toán (Problem Statement)
Hệ thống **Voice Activity Detection (VAD)** có nhiệm vụ phân loại chính xác từng khoảng thời gian trong tín hiệu âm thanh thành 2 trạng thái:
- **Speech**: Bao gồm cả âm hữu thanh (**Voiced - `v`**) và âm vô thanh (**Unvoiced - `uv`**).
- **Silence (`sil`)**: Khoảng lặng hoặc nhiễu nền môi trường.

Đầu vào là file âm thanh WAV chuẩn (16 kHz, đơn kênh). Đầu ra là các mốc thời gian bắt đầu ($\text{Start}$) và kết thúc ($\text{End}$) của vùng tiếng nói, được đối sánh định lượng với nhãn chuẩn chuyên gia (Praat `.lab`) bằng hai độ đo **MAE** và **RMSE** (đơn vị: milliseconds).

### Sơ đồ Khung xử lý Xử lý tín hiệu số (Unified DSP Pipeline)

```mermaid
flowchart LR
    A["Audio WAV<br/>16 kHz Mono"] --> B["Framing<br/>25ms / Hop 10ms"]
    B --> C["Feature Extraction<br/>Norm STE ∈ [0, 1]"]
    C --> D["Classification<br/>TT1 / TT2 / TT3"]
    D --> E["Post-Processing<br/>200ms Bridge Rule<br/>+ 50ms Speech Floor"]
    E --> F["VAD Boundaries<br/>Start & End (ms)"]
    F --> G["Benchmark Eval<br/>MAE, RMSE, F1, UV"]

    style A fill:#0D2536,stroke:#30AFFF,stroke-width:1.5px,color:#fff
    style B fill:#0D2536,stroke:#30AFFF,stroke-width:1.5px,color:#fff
    style C fill:#0D2536,stroke:#30AFFF,stroke-width:1.5px,color:#fff
    style D fill:#0D2536,stroke:#10B981,stroke-width:2px,color:#fff
    style E fill:#0D2536,stroke:#30AFFF,stroke-width:1.5px,color:#fff
    style F fill:#0D2536,stroke:#92EEFF,stroke-width:1.5px,color:#fff
    style G fill:#0D2536,stroke:#F59E0B,stroke-width:1.5px,color:#fff
```

### Bộ thông số kỹ thuật chuẩn hóa (Standard DSP Configurations)
| Tham số | Giá trị | Ý nghĩa vật lý & Cơ sở lý thuyết |
| :--- | :---: | :--- |
| **Tần số lấy mẫu ($F_s$)** | $16\text{ kHz}$ | Băng thông âm thanh tiếng nói chuẩn ($0 - 8\text{ kHz}$). |
| **Độ dài khung (`frame_ms`)** | $25.0\text{ ms}$ ($400\text{ samples}$) | Thỏa mãn giả thiết tựa dừng (quasi-stationary) của âm vị tiếng nói. |
| **Bước dịch khung (`hop_ms`)** | $10.0\text{ ms}$ ($160\text{ samples}$) | Chồng lấp $60\%$, tạo lưới phân giải thời gian mượt mà, theo dõi sát sao chuyển tiếp ngữ âm. |
| **Quy tắc cầu khoảng lặng (`min_silence_ms`)** | $200.0\text{ ms}$ ($20\text{ hops}$) | Yêu cầu bắt buộc của đề tài: bắc cầu qua các đoạn dừng hơi sinh lý tự nhiên ngắn $< 200\text{ ms}$. |
| **Độ dài tiếng nói tối thiểu (`min_speech_ms`)** | $50.0\text{ ms}$ ($5\text{ hops}$) | Loại trừ các gai nhiễu xung cơ học (click, pop, tiếng gõ micro). |

---

## 📊 3. Bảng Đối sánh Định lượng Đa Thuật toán (Benchmark Table)

Bảng tổng hợp dưới đây được trích xuất trực tiếp từ kết quả kiểm thử trên toàn bộ **4 file dữ liệu kiểm thử** (`TinHieuKiemThu/`), đối sánh chính xác với **Slide 19** của buổi báo cáo:

| Nhóm Thuật toán | Biến thể Đánh giá | MAE trung bình (ms) | RMSE trung bình (ms) | Frame F1-Score | Unvoiced Sensitivity (UV) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TT1 (Tìm kiếm nhị phân STE)** | **TT1-1 (Hodgkinson Baseline)** | 10.00 | 10.33 | 0.993 | 0.971 |
| | **TT1-2 (DSP Classifier Mở rộng)** | **6.25** | **7.80** | **0.996** | 0.968 |
| **TT2 (Lược đồ thích nghi Histogram)** | **TT2-1 (Dual STE-SC + Pad 10ms)** | 12.50 | 13.37 | 0.985 | **0.947** |
| | **TT2-2 (Baseline Không đệm No-Pad)** | 18.75 | 19.34 | 0.977 | 0.891 |
| **TT3 (Phân lớp Bayes Gaussian)** | **TT3-1 (1D Gaussian STE)** | 12.50 | 14.08 | 0.994 | 0.971 |
| | **TT3-2 (4D Multivariate Gaussian)** | 15.00 | 16.34 | 0.978 | 0.903 |

> 📌 **Biểu đồ trực quan hóa Benchmark:**  
> Đồ thị cột so sánh trực quan sai số MAE/RMSE và F1-Score của 6 biến thể được lưu trữ tại [`figures/benchmark_plot.png`](figures/benchmark_plot.png) và bản in vector [`figures/benchmark_plot.pdf`](figures/benchmark_plot.pdf).

---

### 💡 Ba Bài học & Đúc kết Chuyên sâu (Scientific Insights)

#### 1. Nguyên lý dao cạo Occam (Occam's Razor) — Tinh giản tham số vượt trội mô hình phức tạp
* **Hiện tượng:** Trong Thuật toán 3, mô hình phân bố đơn biến **1D Gaussian** chỉ với 4 tham số ($\mu_{\text{sil}}, \sigma_{\text{sil}}, \mu_{\text{sp}}, \sigma_{\text{sp}}$) đạt sai số MAE trung bình **$12.50\text{ ms}$** (trong đó có 3/4 file test đạt sai số xuất sắc **$7.50\text{ ms}$**). Ngược lại, mô hình mở rộng **4D Multivariate Gaussian** sử dụng ma trận hiệp phương sai đầy đủ lại bị thoái hóa hiệu năng, MAE tăng lên **$15.00\text{ ms}$** ($+20\%$).
* **Bản chất khoa học:** Trong bối cảnh tập huấn luyện nhỏ (small-sample regime gồm 4 file), mô hình phức tạp với ma trận hiệp phương sai nhiều chiều rất dễ bị hiện tượng quá khớp (overfitting) với đặc thù micro của tập train và nhạy cảm với sai số tính toán nghịch đảo ma trận. Mô hình tinh giản tham số giải tích (parsimonious model) có tính khái quát hóa bền vững hơn hẳn.

#### 2. Tầm quan trọng của Đệm biên (Boundary Padding) — Cứu vãn âm vô thanh năng lượng thấp
* **Hiện tượng:** Trong Thuật toán 2 (Histogram), việc áp dụng kỹ thuật đệm biên $+10\text{ ms}$ (+1 frame) vào hai đầu đoạn phát hiện giúp MAE giảm mạnh từ **$18.75\text{ ms} \to 12.50\text{ ms}$** (giảm tới **$33.3\%$** sai số), đồng thời cải thiện độ nhạy phát hiện âm vô thanh (Unvoiced Sensitivity) từ **$0.891 \to 0.947$**.
* **Bản chất khoa học:** Các phụ âm vô thanh ở đầu câu (như `/s/`, `/t/`, `/f/`, `/k/`) có cơ chế tạo âm do ma sát dòng khí, năng lượng ngắn hạn (STE) tiệm cận mức sàn của nhiễu nền nên dễ bị ngưỡng năng lượng cắt sớm. Kỹ thuật đệm biên 1 frame bù đắp hoàn hảo độ trễ quán tính này.

#### 3. Phối hợp Đặc trưng DSP so với Năng lượng đơn lẻ — Bước nhảy vọt về độ chính xác
* **Hiện tượng:** Biến thể mở rộng **TT1-2 (DSP Classifier)** kết hợp năng lượng thời gian ngắn (STE) với độ lệch tần số và tỷ lệ bằng phẳng phổ đạt sai số MAE kỷ lục toàn benchmark: **$6.25\text{ ms}$** (sai số dưới 1 frame hop) cùng hệ số F1-Score lên tới **$0.996$**.
* **Bản chất khoa học:** Năng lượng chỉ phản ánh biên độ tức thời, rất dễ nhầm lẫn giữa âm vô thanh yếu và tiếng thở/nhiễu nền. Khi bổ sung thông tin phổ (Spectral Features), bộ phân loại phân biệt dứt khoát phổ phẳng của nhiễu trắng và phổ năng lượng cao ở dải trên của phụ âm vô thanh.

---

## 🔍 4. Bảng Kết quả Chi tiết Từng File Kiểm thử (Test Set Breakdown)

Bảng đối soát biên thời gian thực tế giữa Ground Truth (`.lab`) và dự đoán của các thuật toán trên 4 file kiểm thử:

| File Kiểm thử | Ground Truth (s) | TT1 (Hodgkinson) | TT2 (+10ms Pad) | TT3 (1D Bayes) | Đặc điểm âm học & Kênh thu âm |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `phone_F2.wav` | $[1.020, 4.040]$ | MAE: $15.0\text{ ms}$ | MAE: $17.5\text{ ms}$ | MAE: $27.5\text{ ms}$ | Giọng nữ qua kênh điện thoại (SNR thấp, âm đuôi kéo dài). |
| `phone_M2.wav` | $[0.530, 2.520]$ | MAE: $7.5\text{ ms}$ | MAE: $12.5\text{ ms}$ | MAE: **$7.5\text{ ms}$** | Giọng nam điện thoại, ranh giới rơi đúng vào mắt lưới hop 10ms. |
| `studio_F2.wav` | $[0.770, 2.370]$ | MAE: $10.0\text{ ms}$ | MAE: $17.5\text{ ms}$ | MAE: **$7.5\text{ ms}$** | Giọng nữ phòng thu sạch, dải động rộng. |
| `studio_M2.wav` | $[0.450, 1.930]$ | MAE: $7.5\text{ ms}$ | MAE: **$2.5\text{ ms}$** | MAE: **$7.5\text{ ms}$** | Giọng nam phòng thu lý tưởng, đạt độ chính xác gần như tuyệt đối ($2.5\text{ ms}$). |

---

## 📂 5. Cấu trúc Thư mục Dự án (Project Directory Guide)

Dự án được sắp xếp chặt chẽ, phục vụ cả mục đích chấm bài độc lập của Giảng viên và mở rộng nghiên cứu khoa học:

```text
DSP_MidTerm/
├── 08-Phân đoạn tín hiệu tiếng nói - khoảng lặng/    # 📦 GÓI NỘP BÀI CHÍNH THỨC CỦA NHÓM 08
│   ├── 102240280-Luong_Duy_Toan-Binary.ipynb         # Notebook TT1 (Lương Duy Toàn) - Run All độc lập
│   ├── 102240246-Nguyen_Anh_Hoang-Histogram.ipynb    # Notebook TT2 (Nguyễn Anh Hoàng) - Run All độc lập
│   ├── 102240237-Truong_Bui_Dien-SimpleStatics.ipynb # Notebook TT3 (Trương Bùi Điền) - Run All độc lập
│   ├── Slide-Team8.pdf                               # Slide trình bày chính thức (Beamer / Canva 16:9)
│   ├── Script_Thuyet_Trinh_Slide_11_26.docx          # Kịch bản thuyết trình chi tiết (Word)
│   └── Script_Thuyet_Trinh_Slide_11_26.md            # Kịch bản thuyết trình chi tiết (Markdown)
│
├── src/                                              # 🛠️ MÃ NGUỒN PIPELINE MODULE THEO THUẬT TOÁN
│   ├── TT1/                                          # Module Thuật toán 1: Tìm kiếm nhị phân & STE
│   │   ├── 1.ipynb                                   # Bản chuẩn theo yêu cầu của Thầy (Baseline)
│   │   ├── 2.ipynb                                   # Bản mở rộng: Phân lớp đa đặc trưng DSP Classifier
│   │   └── main.ipynb                                # Tích hợp pipeline kiểm thử toàn diện TT1
│   ├── TT2/                                          # Module Thuật toán 2: Histogram thích nghi
│   │   ├── 1.ipynb                                   # Bản chuẩn Histogram STE
│   │   ├── 2.ipynb                                   # Bản mở rộng: Dual STE + Spectral Centroid
│   │   └── main.ipynb                                # Tích hợp pipeline kiểm thử toàn diện TT2
│   └── TT3/                                          # Module Thuật toán 3: Mô hình phân lớp Bayes Gaussian
│       ├── 1.ipynb                                   # Bản chuẩn 1D Gaussian Bayes
│       ├── 2.ipynb                                   # Bản mở rộng: 4D Multivariate Gaussian
│       ├── main.py                                   # Mã nguồn CLI độc lập huấn luyện và đánh giá TT3
│       └── build_notebook.py                         # Trình biên dịch tạo notebook tự động TT3
│
├── figures/                                          # 📈 HỆ THỐNG BIỂU ĐỒ & HÌNH ẢNH XUẤT BẢN
│   ├── benchmark_plot.png (và .pdf)                  # Biểu đồ so sánh đối chuẩn 6 biến thể (Slide 19)
│   ├── tt1_composite.png (và .pdf)                   # Kết quả tổng hợp 4 file test của TT1
│   ├── tt2_composite.png (và .pdf)                   # Kết quả tổng hợp 4 file test của TT2
│   ├── tt3_composite.png (và .pdf)                   # Kết quả tổng hợp 4 file test của TT3
│   ├── tt1_train_convergence.png                     # Đồ thị hội tụ 40 bước chia đôi TT1
│   ├── tt2_padding_effect.png                        # Đồ thị thực nghiệm khảo sát đệm biên TT2
│   └── tt3_train_gaussian.png                        # Đồ thị hai hàm mật độ xác suất Gaussian TT3
│
├── report/                                           # 📝 BÁO CÁO KỸ THUẬT & PHỤ LỤC VẤN ĐÁP
│   ├── VAD_MIDTERM_REPORT_VI.md                      # Báo cáo chuyên sâu bằng tiếng Việt
│   ├── VAD_MIDTERM_REPORT.md                         # Báo cáo kỹ thuật bằng tiếng Anh
│   ├── DEFENSE_NOTES_VI.md                           # Sổ tay ghi chú phản biện
│   └── PHU_LUC_VAN_DAP_VI.md                         # 8 Câu hỏi "bẫy" kinh điển và kịch bản trả lời Thầy
│
├── TinHieuHuanLuyen/                                 # 🎓 DỮ LIỆU HUẤN LUYỆN (4 CẶP WAV + LAB)
│   ├── phone_F1.wav & phone_F1.lab
│   ├── phone_M1.wav & phone_M1.lab
│   ├── studio_F1.wav & studio_F1.lab
│   └── studio_M1.wav & studio_M1.lab
│
├── TinHieuKiemThu/                                   # 🎯 DỮ LIỆU KIỂM THỬ (4 CẶP WAV + LAB ĐỘC LẬP)
│   ├── phone_F2.wav & phone_F2.lab
│   ├── phone_M2.wav & phone_M2.lab
│   ├── studio_F2.wav & studio_F2.lab
│   └── studio_M2.wav & studio_M2.lab
│
├── pyproject.toml & uv.lock                          # Cấu hình môi trường Python (chuẩn Astral uv)
└── README.md                                         # Tài liệu tổng quan dự án (File này)
```

---

## ⚡ 6. Hướng dẫn Cài đặt & Chạy Thực nghiệm (Quick Start)

Dự án tuân thủ nghiêm ngặt chuẩn quản lý môi trường hiện đại bằng công cụ siêu tốc **`uv`**. Không phụ thuộc vào cài đặt hệ thống toàn cục, đảm bảo tái lập 100% trên mọi máy tính.

### Bước 1: Cài đặt công cụ `uv` (Nếu chưa có)
```bash
# Cài đặt uv trên Linux / macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Hoặc cài đặt trên Windows qua PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Bước 2: Khởi tạo môi trường ảo và Đồng bộ phụ thuộc
Tại thư mục gốc của dự án (`/home/bim/Projects/DSP_MidTerm`), thực hiện:
```bash
# 1. Tạo môi trường ảo cục bộ (.venv)
uv venv

# 2. Đồng bộ toàn bộ thư viện chính xác theo uv.lock
uv sync
```

### Bước 3: Khởi chạy và Chấm điểm Notebooks
Để mở giao diện Jupyter và kiểm tra các notebook nộp bài:
```bash
uv run jupyter lab
```
Hoặc:
```bash
uv run jupyter notebook
```

#### Quy trình chấm điểm nhanh cho Thầy:
1. Mở bất kỳ notebook nào trong thư mục gói nộp bài:
   - `08-Phân đoạn tín hiệu tiếng nói - khoảng lặng/102240280-Luong_Duy_Toan-Binary.ipynb`
   - `08-Phân đoạn tín hiệu tiếng nói - khoảng lặng/102240246-Nguyen_Anh_Hoang-Histogram.ipynb`
   - `08-Phân đoạn tín hiệu tiếng nói - khoảng lặng/102240237-Truong_Bui_Dien-SimpleStatics.ipynb`
2. Chọn menu **Kernel** $\to$ **Restart Kernel and Run All Cells**.
3. Toàn bộ 4 file kiểm thử sẽ được nạp, tính toán đặc trưng, xuất bảng đối chiếu và vẽ biểu đồ waveform kết quả inline ngay lập tức mà không gặp bất kỳ lỗi phụ thuộc nào!

### Chạy kiểm thử dòng lệnh (CLI):
```bash
# Chạy trực tiếp module giải tích Gaussian Bayes (TT3):
uv run python src/TT3/main.py
```

---

## 🔬 7. Tóm tắt Chi tiết Ba Thuật toán Lõi

### Thuật toán 01: Năng lượng ngắn hạn (STE) & Tìm kiếm nhị phân (Fixed Global STE)
* **Nguyên lý:** Năng lượng ngắn hạn chuẩn hóa:
  $$E[m] = \frac{1}{N \cdot E_{\max}} \sum_{n=0}^{N-1} x_m[n]^2$$
* **Thuật toán chia đôi:** Huấn luyện trên 4 file `TinHieuHuanLuyen/`, mục tiêu tối ưu hàm chênh lệch tổng thời lượng tiếng nói $\Delta D(T) = D_{\text{pred}}(T) - D_{\text{ref}}$.
* **Kết quả:** Sau 40 vòng lặp chia đôi, thuật toán hội tụ về ngưỡng toàn cục tối ưu $T_{\text{opt}} \approx 0.00348$.
* **Đặc điểm:** Tốc độ kiểm tra từng khung cực nhanh $O(1)$, tiêu tốn bộ nhớ tối thiểu; phù hợp chip nhúng thời gian thực trong môi trường tĩnh.

### Thuật toán 02: Lược đồ thích nghi Histogram (Adaptive Dual STE-SC Histogram)
* **Nguyên lý:** Không cần nhãn trước (Unsupervised). Xây dựng lược đồ phân bố năng lượng 100 bins trên từng câu nói, áp dụng bộ lọc trung bình trượt 5 bins để triệt tiêu các cực đại giả do nhiễu.
* **Xác định ngưỡng (Giannakopoulos):** Tìm đỉnh khoảng lặng $M_1$ và đỉnh tiếng nói $M_2$, ngưỡng được tính theo tỷ lệ trọng số:
  $$T = \frac{W \cdot M_1 + M_2}{W + 1} \quad (W = 5)$$
* **Đột phá:** Áp dụng đặc trưng kép đồng thời (**Dual Feature STE + Spectral Centroid**) và đệm biên **$+10\text{ ms}$**, giúp khôi phục các âm vô thanh có năng lượng thấp ở biên câu.

### Thuật toán 03: Mô hình phân lớp Bayes tham số Gaussian (Parametric Gaussian Bayes)
* **Nguyên lý:** Giả thiết hàm mật độ xác suất năng lượng của khoảng lặng và tiếng nói tuân theo phân bố chuẩn:
  $$p(x \mid \text{Sil}) \sim \mathcal{N}(\mu_{\text{sil}}, \sigma_{\text{sil}}^2), \quad p(x \mid \text{Sp}) \sim \mathcal{N}(\mu_{\text{sp}}, \sigma_{\text{sp}}^2)$$
* **Ước lượng tham số (MLE từ tập huấn luyện):**
  - Khoảng lặng: $\mu_{\text{sil}} \approx 0.00039, \; \sigma_{\text{sil}} \approx 0.00071$
  - Tiếng nói: $\mu_{\text{sp}} \approx 0.20265, \; \sigma_{\text{sp}} \approx 0.23563$ (chênh lệch năng lượng $> 500$ lần).
* **Nghiệm giải tích Bayes:** Thiết lập điều kiện đẳng xác suất hậu nghiệm $p(x \mid \text{Sil}) = p(x \mid \text{Sp})$, dẫn về phương trình bậc hai $Ax^2 + Bx + C = 0$ và giải nghiệm dương duy nhất nằm giữa hai kỳ vọng: $T_{\text{Bayes}} \approx 0.00288$.

---

## 📑 8. Tài nguyên Trình bày & Tham khảo Dự án

* 📽️ **Slide Thuyết trình Báo cáo:** [Canva Slide VAD Midterm](https://canva.link/jhh1gex7ybd87yk) | Bản in PDF chất lượng cao: [`08-Phân đoạn tín hiệu tiếng nói - khoảng lặng/Slide-Team8.pdf`](08-Phân%20đoạn%20tín%20hiệu%20tiếng%20nói%20-%20khoảng%20lặng/Slide-Team8.pdf)
* 🎤 **Kịch bản Thuyết trình Slide 11 - 26:** [`Script_Thuyet_Trinh_Slide_11_26.md`](08-Phân%20đoạn%20tín%20hiệu%20tiếng%20nói%20-%20khoảng%20lặng/Script_Thuyet_Trinh_Slide_11_26.md)
* 🛡️ **Tài liệu Phụ lục Vấn đáp (Đối phó 8 câu hỏi bẫy của Thầy):** [`report/PHU_LUC_VAN_DAP_VI.md`](report/PHU_LUC_VAN_DAP_VI.md)
* 📖 **Báo cáo Kỹ thuật Đầy đủ (Vietnamese Full Spec):** [`report/VAD_MIDTERM_REPORT_VI.md`](report/VAD_MIDTERM_REPORT_VI.md)

---

> **Cam kết từ Nhóm 08:** Toàn bộ dữ liệu, đồ thị và kết quả trong báo cáo là sản phẩm nghiên cứu thực nghiệm trung thực 100%, có thể tái lập hoàn toàn thông qua mã nguồn đính kèm. Nhóm sẵn sàng thực hiện demo trực tiếp trên bất kỳ file âm thanh kiểm thử bất kỳ theo yêu cầu của Thầy!
