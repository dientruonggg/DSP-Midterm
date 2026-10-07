# PHỤ LỤC VẤN ĐÁP & BACKUP SLIDES (DỰ PHÒNG KHI THẦY HỎI)
**Học phần:** Xử lý tín hiệu số (DSP Midterm)  
**Tài liệu đi kèm:** [Canva Slide VAD Midterm](https://canva.link/jhh1gex7ybd87yk) | [Báo cáo chi tiết tiếng Việt](file:///home/bim/Projects/DSP_MidTerm/report/VAD_MIDTERM_REPORT_VI.md)

---

## 📌 HƯỚNG DẪN SỬ DỤNG PHỤ LỤC NÀY
> [!IMPORTANT]
> * **Trong 3 phút chính thức:** Chỉ trình bày đúng 4 slide cốt lõi (Cover $\to$ Giải pháp lõi $\to$ Kết quả thực nghiệm $\to$ Kết luận). Tuyệt đối **không** chiếu trước các slide phụ lục này để tránh bị quá giờ.
> * **Trong 1 phút Demo & phần Vấn đáp (Q&A):** Nếu Thầy hỏi sâu vào bất kỳ khía cạnh nào (toán học, tham số, hiện tượng lệch biên, âm vô thanh, mô hình đa biến...), hãy bình tĩnh bấm mở các Slide phụ lục từ **Slide 5 trở đi** (hoặc mở file tài liệu này) để trả lời Thầy bằng số liệu và hình vẽ trực quan!

---

## 🗺️ BẢN ĐỒ 5 SLIDE PHỤ LỤC TRÊN CANVA (SLIDES 18 - 22)
1. **[Slide 18] Phụ lục 01: Tiền xử lý & Bộ tham số chuẩn (Framing 25ms/10ms, 200ms silence bridge, 50ms min speech)**
2. **[Slide 19] Phụ lục 02: Kết quả trung gian TT1 (Tìm kiếm nhị phân, 40 bước lặp, T_opt = 0.00348)**
3. **[Slide 20] Phụ lục 03: Kết quả trung gian TT2 (Dual-Feature STE + SC, 100 bins, 5-bin MA, W=5, Tinh chỉnh đệm biên Train: 13.75ms -> 11.25ms)**
4. **[Slide 21] Phụ lục 04: Kết quả trung gian TT3 (Mean/Std Sil & Sp, Giải pt Bayes, T_Bayes = 0.00288)**
5. **[Slide 22] Phụ lục 05: Đối thoại phản biện & 8 câu hỏi bẫy vấn đáp kinh điển**

---

## PHỤ LỤC A: BẢNG THÔNG SỐ CẤU HÌNH HỆ THỐNG
*(Mở slide này nếu Thầy hỏi: "Frame size bao nhiêu? Hop size bao nhiêu? Tại sao lại chọn số này?")*

| Tham số | Giá trị | Ý nghĩa vật lý trong DSP |
|---|:---:|---|
| **Tần số lấy mẫu ($F_s$)** | $16\text{ kHz}$ | Chuẩn âm thanh tiếng nói (băng rộng $0 - 8\text{ kHz}$). |
| **Độ dài khung (`frame_ms`)** | $25.0\text{ ms}$ ($400$ samples) | Tiếng nói có tính tựa dừng (quasi-stationary) trong khoảng $20 - 30\text{ ms}$, cho phép tính các đặc trưng ngắn hạn chuẩn quốc tế. |
| **Bước dịch khung (`hop_ms`)** | $10.0\text{ ms}$ ($160$ samples) | Độ phân giải thời gian của hệ thống. Chồng lấp $60\%$ giúp theo dõi trơn tru chuyển tiếp âm vị. |
| **Khoảng lặng tối thiểu (`min_silence_ms`)** | $200.0\text{ ms}$ ($20$ hops) | Yêu cầu bắt buộc của đề bài. Dùng để nối các đoạn ngắt hơi giả/ngắn giữa các từ thành một khối tiếng nói liên tục. |
| **Độ dài tiếng nói tối thiểu (`min_speech_ms`)** | $50.0\text{ ms}$ ($5$ hops) | Ngăn nhiễu xung (click, pop, gõ mic) bị nhận nhầm thành tiếng nói (âm vị người ngắn nhất $\ge 30-50\text{ ms}$). |

---

## PHỤ LỤC B: BẢNG ĐỐI CHIẾU ĐỊNH LƯỢNG 4 CHỈ SỐ
*(Mở bảng này nếu Thầy hỏi: "Ngoài MAE ra các em có đánh giá thêm độ lệch chuẩn, độ chính xác frame hay âm vô thanh không?")*

| Thuật toán | Biến thể | MAE trung bình (ms) | RMSE trung bình (ms) | Frame F1-score | Unvoiced Recall (UV) |
|---|---|:---:|:---:|:---:|:---:|
| **TT1 (Nhị phân STE)** | Chuẩn 25ms/10ms (`1-mine.ipynb`) | $10.00$ | $10.33$ | $0.993$ | $0.971$ |
| **TT1 (DSP Classifier)** | Logistic đa đặc trưng (`2.ipynb`) | **$6.25$** | **$7.80$** | **$0.996$** | $0.968$ |
| **TT2 (Dual STE + SC)** | Chuẩn tối ưu (+10ms Pad) (`1-mine.ipynb`) | $12.50$ | $13.37$ | $0.985$ | **$0.947$** |
| **TT2 (Cơ sở STE thuần)** | Bản không đệm biên (No-Pad) | $18.75$ | $19.34$ | $0.977$ | $0.891$ |
| **TT3 (Gaussian 1D STE)** | Chuẩn Bayes 25ms/10ms (`src/TT3/main.py`) | $12.50$ | $14.08$ | $0.994$ | $0.971$ |
| **TT3 (Gaussian đa biến)** | Mở rộng 4 chiều (`2.ipynb`) | $15.00$ | $16.34$ | $0.978$ | $0.903$ |

> **Bảng số liệu chi tiết TT2 (Dual-Feature STE + Spectral Centroid) trên 4 file Test (trích `output/mine/intermediate_results.csv`):**
> 
> | File Test | $M_{1\_E}$ | $M_{2\_E}$ | **Ngưỡng $T_E$** | $M_{1\_C}$ | $M_{2\_C}$ | **Ngưỡng $T_C$** | Sau AND | Sau 200ms | Final (+10ms Pad) | Sai số MAE |
> | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
> | `phone_F2.wav` | 0.0051 | 0.0751 | **0.0167** | 0.1655 | 0.2248 | **0.1754** | 212 frames | 269 frames | **271 frames** | 17.50 ms |
> | `phone_M2.wav` | 0.0050 | 0.1750 | **0.0333** | 0.1860 | 0.2208 | **0.1918** | 117 frames | 172 frames | **174 frames** | 12.50 ms |
> | `studio_F2.wav` | 0.0050 | 0.0650 | **0.0150** | 0.0050 | 0.0950 | **0.0200** | 134 frames | 157 frames | **159 frames** | 17.50 ms |
> | `studio_M2.wav` | 0.0050 | 0.0750 | **0.0167** | 0.1518 | 0.2115 | **0.1618** | 120 frames | 144 frames | **146 frames** | **2.50 ms** |

> **Bảng số liệu chi tiết TT3 (Gaussian 1D STE - Ngưỡng Bayes) trên 4 file Test (trích từ `src/TT3/main.py`):**
> 
> | File Test | Ground Truth (s) | Dự đoán TT3 (s) | $\Delta\text{Start}$ | $\Delta\text{End}$ | Sai số MAE | Sai số RMSE |
> | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
> | `phone_F2.wav` | $[1.02, 4.04]$ | $[1.01, 4.08]$ | $-10\text{ ms}$ | $+45\text{ ms}$ | **27.50 ms** | 32.60 ms |
> | `phone_M2.wav` | $[0.53, 2.52]$ | $[0.52, 2.52]$ | $-10\text{ ms}$ | $+5\text{ ms}$ | **7.50 ms** | 7.91 ms |
> | `studio_F2.wav` | $[0.77, 2.37]$ | $[0.76, 2.37]$ | $-10\text{ ms}$ | $-5\text{ ms}$ | **7.50 ms** | 7.91 ms |
> | `studio_M2.wav` | $[0.45, 1.93]$ | $[0.46, 1.94]$ | $+10\text{ ms}$ | $+5\text{ ms}$ | **7.50 ms** | 7.91 ms |
> | **Trung bình** | — | — | — | — | **12.50 ms** | **14.08 ms** |

---

## PHỤ LỤC C: BỘ HÌNH ẢNH DỰ PHÒNG & GIẢI THÍCH HIỆN TƯỢNG TỪNG FILE
*(Mở các file ảnh này từ thư mục dự phòng khi Thầy muốn xem chi tiết)*

### 1. Hình ảnh Giải pháp lõi & Huấn luyện (Train)
* **TT1 - Hội tụ chia đôi 40 epochs:** [`output_plots/intermediate_tt1_binary_search.png`](file:///home/bim/Projects/DSP_MidTerm/output_plots/intermediate_tt1_binary_search.png)
  * Hội tụ về ngưỡng toàn cục tối ưu $T_{\text{opt}} = 0.003478 \approx 0.00348$.
  * Sai số thời lượng có dấu $\Delta D(T) \to 0.00\text{ ms}$. MAE trên tập train đạt $11.25\text{ ms}$.
* **TT2 - Khảo sát đệm biên (Boundary-Padding Sweep):** [`figures/tt2_padding_effect.png`](file:///home/bim/Projects/DSP_MidTerm/figures/tt2_padding_effect.png)
  * Khảo sát đệm biên $0 - 40\text{ ms}$ ($0 - 4$ khung). Đệm $10\text{ ms}$ ($1$ khung) giúp MAE train giảm từ $13.75\text{ ms} \to 11.25\text{ ms}$ (giảm $-18.2\%$).
  * Phục hồi các chu kỳ đầu của âm vô thanh bị ngưỡng năng lượng cắt sớm.
* **TT3 - Đồ thị phân bố Gaussian:** [`figures/tt3_train_gaussian.png`](file:///home/bim/Projects/DSP_MidTerm/figures/tt3_train_gaussian.png)
  * $\mu_{\text{sil}} = 0.000387 \approx 0.00039$, $\sigma_{\text{sil}} = 0.000709 \approx 0.00071$ (Khoảng lặng).
  * $\mu_{\text{sp}} = 0.202649 \approx 0.20265$, $\sigma_{\text{sp}} = 0.235626 \approx 0.23563$ (Tiếng nói).
  * Ngưỡng Bayes tối ưu (giao thoa đẳng xác suất): $T_{\text{opt}} = 0.002878 \approx 0.00288$.

### 2. Hình ảnh Composite tổng hợp 4 file kiểm thử (Test)
* **TT1 Composite (Waveform + STE):** [`figures/tt1_composite.png`](file:///home/bim/Projects/DSP_MidTerm/figures/tt1_composite.png)
* **TT2 Composite (Waveform + STE + Spectral Centroid + Histograms):** [`figures/tt2_composite.png`](file:///home/bim/Projects/DSP_MidTerm/figures/tt2_composite.png)
* **TT3 Composite (Waveform + STE + Gaussian Posterior):** [`figures/tt3_composite.png`](file:///home/bim/Projects/DSP_MidTerm/figures/tt3_composite.png)
* **Đồ thị so sánh Benchmark:** [`figures/benchmark_plot.png`](file:///home/bim/Projects/DSP_MidTerm/figures/benchmark_plot.png)

### 3. Phân tích hiện tượng đặc biệt ở từng file test
* **`studio_M2` ($\text{MAE} = 2.5\text{ ms}$ ở TT2, $7.5\text{ ms}$ ở TT1):**
  * Nền Studio cực sạch, chuyển tiếp tiếng nói dứt khoát. Sau khi bù $+10\text{ ms}$ đệm biên, sai số bắt đầu là $0.0\text{ ms}$ và sai số kết thúc chỉ lệch $-5.0\text{ ms}$ (nửa hop $10\text{ ms}$), cho MAE xuất sắc $2.5\text{ ms}$.
* **`phone_M2` ($\text{MAE} = 7.5\text{ ms}$ ở TT1, $12.5\text{ ms}$ ở TT2):**
  * Ground truth: $[0.53\text{ s}, 2.52\text{ s}]$. Cả hai mốc đều là bội số nguyên của hop $10\text{ ms}$ (frame 53 và frame 252).
* **`phone_F2` & `studio_F2` ($\text{MAE} = 12.5 - 17.5\text{ ms}$):**
  * Âm sắc nữ có tần số cơ bản cao, năng lượng rải rộng. Âm đuôi có hơi thở nhẹ tiệm cận mức nhiễu của micro khiến biên kết thúc có độ trễ nhẹ $1-2$ khung. Đây là giới hạn tự nhiên của phương pháp năng lượng.

---

## PHỤ LỤC D: KỊCH BẢN ỨNG PHÓ 8 CÂU HỎI "BẪY" KINH ĐIỂN CỦA THẦY

### Câu 1: "Tại sao không dùng F0 (Pitch) để xác định biên tiếng nói luôn cho tiện?"
* **Trả lời sắc bén:**  
  * *"Dạ thưa Thầy, F0 chỉ tồn tại ở các âm hữu thanh (Voiced) do dây thanh quản rung tuần hoàn. Các âm vô thanh (Unvoiced) như phụ âm /s/, /f/, /t/, /k/ không có F0 ổn định (đường F0 bị đứt quãng).*  
  * *Nếu dùng F0 để phân đoạn, toàn bộ âm vô thanh ở đầu và cuối câu sẽ bị cắt bỏ nhầm thành khoảng lặng. Nhóm em chỉ vẽ F0 để minh họa trực quan vùng hữu thanh chứ không đưa F0 vào điều kiện quyết định biên."*

### Câu 2: "Tại sao có file đạt MAE = 0 ms? Có phải các em hack số hay overfit không?"
* **Trả lời sắc bén:**  
  * *"Dạ không thưa Thầy! File `phone_M2` có biên chuẩn trong file `.lab` là $0.53\text{ s}$ và $2.52\text{ s}$.*  
  * *Do hệ thống dùng bước dịch khung `hop_ms = 10 ms` ($0.01\text{ s}$), nên mốc $0.53\text{ s}$ chính là khung thứ $53$ và $2.52\text{ s}$ là khung thứ $252$. Khi phân lớp đúng frame này thì sai số đo được bằng $0\text{ ms}$. Đây là sự trùng khớp tự nhiên theo độ phân giải lưới thời gian chứ nhóm không can thiệp thủ công."*

### Câu 3: "Vì sao ở TT2 (Histogram), các em chọn trọng số $W = 5$ mà không phải $W = 1$?"
* **Trả lời sắc bén:**  
  * *"Dạ theo nghiên cứu của Giannakopoulos (2014), $M_1$ là đỉnh của khoảng lặng và $M_2$ là đỉnh của tiếng nói.*  
  * *Nếu chọn $W = 1$, ngưỡng $T$ sẽ là trung bình cộng nằm chính giữa, dẫn tới ngưỡng quá cao và cắt mất các âm vô thanh có năng lượng thấp. Chọn $W = 5$ nhằm gán trọng số lớn hơn cho $M_1$, kéo ngưỡng dịch sát về phía khoảng lặng để bảo toàn các phụ âm yếu."*

### Câu 4: "Tại sao phải làm trơn (Smoothing) Histogram bằng Moving Average?"
* **Trả lời sắc bén:**  
  * *"Dạ thưa Thầy, biểu đồ Histogram thô của STE có rất nhiều dao động nhỏ ngẫu nhiên (nhiễu cục bộ). Nếu không lọc trơn, hàm tìm cực đại cục bộ sẽ bị lừa và bắt nhầm các gai nhiễu làm $M_1, M_2$, dẫn đến tính sai ngưỡng hoàn toàn.*  
  * *Làm trơn $5$ điểm đóng vai trò như một bộ lọc thông thấp (low-pass filter) trên miền tần suất, làm nổi bật 2 mode thực sự của phân bố."*

### Câu 5: "Quy tắc khoảng lặng 200 ms: Nếu người nói ngắt 150 ms giữa 2 từ thì có bị mất không?"
* **Trả lời sắc bén:**  
  * *"Dạ theo đúng yêu cầu đề bài của Thầy, khoảng lặng thật sự phân tách các câu nói phải dài tối thiểu $200\text{ ms}$.*  
  * *Các khoảng ngắt hơi ngắn $< 200\text{ ms}$ giữa các từ được xem là khoảng lặng sinh lý trong câu. Thuật toán sẽ gộp (bridge) chúng lại để giữ câu nói thành một đoạn thống nhất từ điểm bắt đầu đến điểm kết thúc, khớp với định dạng chuẩn của file `.lab`."*

### Câu 6: "Ở TT3, vì sao mô hình Gaussian đa biến phức tạp hơn nhưng MAE lại kém hơn Gaussian 1 chiều?"
* **Trả lời sắc bén:**  
  * *"Dạ thưa Thầy, đây là minh chứng rõ ràng cho hiện tượng Overfitting khi thiếu dữ liệu:*  
  * *Tập huấn luyện chỉ có 4 file âm thanh (kích thước mẫu nhỏ), trong khi mô hình Gaussian đa biến $4$ chiều phải ước lượng ma trận hiệp phương sai đầy đủ gồm nhiều tham số. Năng lượng STE một chiều vốn đã phân tách 2 lớp rất tốt, nên mô hình đơn giản có phương sai ước lượng thấp hơn và khái quát hóa trên tập test tốt hơn mô hình phức tạp."*

### Câu 7: "Tại sao nhóm không tối ưu tiếp ngưỡng trực tiếp trên 4 file kiểm thử (Test) để MAE thấp hơn nữa?"
* **Trả lời sắc bén:**  
  * *"Dạ làm như vậy là vi phạm nguyên tắc Data Leakage trong xử lý tín hiệu và học máy.*  
  * *Tập test phải được giữ nguyên vẹn để phản ánh khách quan năng lực tổng quát hóa của thuật toán. Nhóm em chỉ huấn luyện trên 4 file Train (dùng `studio_M1` làm validation cục bộ) và chỉ chạy kiểm thử đúng 1 lần duy nhất để báo cáo."*

### Câu 8: "Sự khác biệt giữa môi trường Phone và Studio ảnh hưởng thế nào đến kết quả?"
* **Trả lời sắc bén:**  
  * *"Dạ môi trường Studio có tỷ số SNR cao, nền âm học sạch nên năng lượng khoảng lặng sát 0, biên tiếng nói rất sắc nét $\rightarrow \text{MAE}$ chỉ $5 - 10\text{ ms}$.*  
  * *Môi trường Phone có SNR thấp hơn, micro điện thoại có đáp ứng tần số phi tuyến và đuôi hơi thở kéo dài, khiến ngưỡng năng lượng dễ bị kéo trễ ở biên kết thúc $\rightarrow \text{MAE}$ dao động $20 - 30\text{ ms}$."*

---

## PHỤ LỤC E: KIẾN THỨC NÂNG CAO CHO PHẦN MỞ RỘNG (VARIANT 2)
*(Dùng khi Thầy muốn hỏi thêm để lấy điểm 10 tuyệt đối)*

1. **Bộ lọc tiền nhấn Pre-emphasis FIR:**
   $$y[n] = x[n] - 0.97 x[n-1]$$
   * **Bản chất:** Bộ lọc thông cao bậc 1 ($H(z) = 1 - 0.97z^{-1}$).
   * **Tác dụng:** Bù lại sự suy giảm phổ $-6\text{ dB/octave}$ do bức xạ môi, nâng biên độ các tần số cao ($> 3\text{ kHz}$) của âm vô thanh (/s/, /f/) lên để năng lượng sau lọc phân biệt rõ với khoảng lặng.
2. **Tỷ lệ đổi dấu Zero Crossing Rate (ZCR):**
   * Âm hữu thanh (Voiced): Năng lượng tập trung ở tần số thấp $\rightarrow$ ZCR thấp.
   * Âm vô thanh (Unvoiced): Giàu thành phần cao tần $\rightarrow$ ZCR cao.
   * Khoảng lặng (Silence): Năng lượng rất thấp nhưng ZCR có thể cao do nhiễu $\rightarrow$ Bắt buộc phải kết hợp ngưỡng sàn năng lượng (Energy Floor) chứ không dùng ZCR độc lập.
3. **Histogram 2 chiều $H[\text{STE}, \text{ZCR}]$:**
   * Chia không gian đặc trưng thành lưới 2D $50 \times 50$ ô.
   * Phân lớp dựa trên Log-likelihood ratio có làm trơn:
     $$\text{Score} = \ln \frac{P(\text{STE}, \text{ZCR} \mid \text{Speech}) + \epsilon}{P(\text{STE}, \text{ZCR} \mid \text{Silence}) + \epsilon}$$
