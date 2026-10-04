# PHỤ LỤC VẤN ĐÁP & BACKUP SLIDES (DỰ PHÒNG KHI THẦY HỎI)
**Học phần:** Xử lý tín hiệu số (DSP Midterm)  
**Tài liệu đi kèm:** [Canva Slide VAD Midterm](https://canva.link/jhh1gex7ybd87yk) | [Báo cáo chi tiết tiếng Việt](file:///home/bim/Projects/DSP_MidTerm/report/VAD_MIDTERM_REPORT_VI.md)

---

## 📌 HƯỚNG DẪN SỬ DỤNG PHỤ LỤC NÀY
> [!IMPORTANT]
> * **Trong 3 phút chính thức:** Chỉ trình bày đúng 4 slide cốt lõi (Cover $\to$ Giải pháp lõi $\to$ Kết quả thực nghiệm $\to$ Kết luận). Tuyệt đối **không** chiếu trước các slide phụ lục này để tránh bị quá giờ.
> * **Trong 1 phút Demo & phần Vấn đáp (Q&A):** Nếu Thầy hỏi sâu vào bất kỳ khía cạnh nào (toán học, tham số, hiện tượng lệch biên, âm vô thanh, mô hình đa biến...), hãy bình tĩnh bấm mở các Slide phụ lục từ **Slide 5 trở đi** (hoặc mở file tài liệu này) để trả lời Thầy bằng số liệu và hình vẽ trực quan!

---

## MỤC LỤC PHỤ LỤC
1. [Phụ lục A: Bảng thông số cấu hình hệ thống (System Hyperparameters)](#phụ-lục-a-bảng-thông-số-cấu-hình-hệ-thống)
2. [Phụ lục B: Bảng đối chiếu định lượng 4 chỉ số (MAE, RMSE, Frame F1, UV Recall)](#phụ-lục-b-bảng-đối-chiếu-định-lượng-4-chỉ-số)
3. [Phụ lục C: Bộ hình ảnh dự phòng & Giải thích hiện tượng từng file](#phụ-lục-c-bộ-hình-ảnh-dự-phòng--giải-thích-hiện-tượng-từng-file)
4. [Phụ lục D: Kịch bản ứng phó 8 câu hỏi "bẫy" kinh điển của Thầy](#phụ-lục-d-kịch-bản-ứng-phó-8-câu-hỏi-bẫy-kinh-điển-của-thầy)
5. [Phụ lục E: Kiến thức nâng cao cho phần mở rộng (Advanced DSP - Variant 2)](#phụ-lục-e-kiến-thức-nâng-cao-cho-phần-mở-rộng)

---

## PHỤ LỤC A: BẢNG THÔNG SỐ CẤU HÌNH HỆ THỐNG
*(Mở slide này nếu Thầy hỏi: "Frame size bao nhiêu? Hop size bao nhiêu? Tại sao lại chọn số này?")*

| Tham số | Giá trị | Ý nghĩa vật lý trong DSP |
|---|:---:|---|
| **Tần số lấy mẫu ($F_s$)** | $16\text{ kHz}$ | Chuẩn âm thanh tiếng nói (băng rộng $0 - 8\text{ kHz}$). |
| **Độ dài khung (`frame_ms`)** | $20.0\text{ ms}$ ($320$ samples) | Tiếng nói có tính tựa dừng (quasi-stationary) trong khoảng $10 - 30\text{ ms}$, cho phép tính các đặc trưng ngắn hạn. |
| **Bước dịch khung (`hop_ms`)** | $10.0\text{ ms}$ ($160$ samples) | Độ phân giải thời gian của hệ thống. Chồng lấp $50\%$ giúp không bỏ sót sự thay đổi đột ngột giữa các khung. |
| **Khoảng lặng tối thiểu (`min_silence_ms`)** | $200.0\text{ ms}$ ($20$ hops) | Yêu cầu bắt buộc của đề bài. Dùng để nối các đoạn ngắt hơi giả/ngắn giữa các từ thành một khối tiếng nói liên tục. |
| **Độ dài tiếng nói tối thiểu (`min_speech_ms`)** | $50.0\text{ ms}$ ($5$ hops) | Ngăn nhiễu xung (click, pop, gõ mic) bị nhận nhầm thành tiếng nói (âm vị người ngắn nhất $\ge 30-50\text{ ms}$). |

---

## PHỤ LỤC B: BẢNG ĐỐI CHIẾU ĐỊNH LƯỢNG 4 CHỈ SỐ
*(Mở bảng này nếu Thầy hỏi: "Ngoài MAE ra các em có đánh giá thêm độ lệch chuẩn, độ chính xác frame hay âm vô thanh không?")*

| Thuật toán | Biến thể | MAE trung bình (ms) | RMSE trung bình (ms) | Frame F1-score | Unvoiced Recall (UV) |
|---|---|:---:|:---:|:---:|:---:|
| **TT1 (Nhị phân STE)** | Baseline (`1.ipynb`) | $8.75$ | $11.34$ | $0.993$ | $0.971$ |
| **TT1 (Logistic đa đặc trưng)** | Mở rộng (`2.ipynb`) | **$6.25$** | **$7.80$** | **$0.996$** | $0.968$ |
| **TT2 (Histogram 1D STE)** | Baseline (`1.ipynb`) | $17.50$ | $18.81$ | $0.977$ | $0.891$ |
| **TT2 (Histogram 2D STE-ZCR)** | Mở rộng (`2.ipynb`) | $20.00$ | $21.56$ | $0.988$ | **$0.947$** |
| **TT3 (Gaussian 1D STE)** | Baseline (`1.ipynb`) | $11.25$ | $14.87$ | $0.994$ | **$0.971$** |
| **TT3 (Gaussian đa biến)** | Mở rộng (`2.ipynb`) | $15.00$ | $16.34$ | $0.978$ | $0.903$ |

> **Nhận xét chuyên sâu:**
> * **TT1-2 (Mở rộng)** là thuật toán định vị biên tốt nhất toàn diện: $\text{MAE} = 6.25\text{ ms}$.
> * **TT2-2 (Histogram 2D)** cứu âm vô thanh xuất sắc nhất: $\text{UV Recall}$ tăng từ $89.1\%$ lên **$94.7\%$**, chấp nhận đánh đổi biên ngoài lệch nhẹ $1-2$ hop do bảo toàn vùng vô thanh biên giới.
> * **TT3-1 (Gaussian 1D)** vượt trội hơn TT3-2 (Đa biến) vì tập train chỉ có 4 file, mô hình ít tham số hoạt động ổn định và khái quát hóa tốt hơn ma trận hiệp phương sai đầy đủ.

---

## PHỤ LỤC C: BỘ HÌNH ẢNH DỰ PHÒNG & GIẢI THÍCH HIỆN TƯỢNG TỪNG FILE
*(Mở các file ảnh này từ thư mục dự phòng khi Thầy muốn xem chi tiết)*

### 1. Hình ảnh Giải pháp lõi & Huấn luyện (Train)
* **TT3 - Đồ thị phân bố Gaussian:** [`gaussian_distribution_slide.png`](file:///home/bim/Projects/DSP_MidTerm/src/TT3/output/1/gaussian_distribution_slide.png)
  * $\mu_{sil} = 0.00033$, $\sigma_{sil} = 0.00047$ (Khoảng lặng).
  * $\mu_{sp} = 0.19652$, $\sigma_{sp} = 0.23327$ (Tiếng nói).
  * Ngưỡng giao thoa tối ưu: $T = 0.00202$.
* **TT2 - Đồ thị Histogram 1D (Giannakopoulos):** Hai mode $M_1$ (silence) và $M_2$ (speech), làm trơn bằng Moving Average $5$ điểm, tính ngưỡng với $W = 5$: $T = \frac{5 M_1 + M_2}{6}$.

### 2. Hình ảnh Composite tổng hợp 4 file kiểm thử (Test)
* **TT1 Composite:** [`output_plots/composite_4files_tt1.png`](file:///home/bim/Projects/DSP_MidTerm/output_plots/composite_4files_tt1.png)
* **TT2 Composite:** [`output_plots/composite_4files_tt2.png`](file:///home/bim/Projects/DSP_MidTerm/output_plots/composite_4files_tt2.png)
* **TT3 Composite:** [`output_plots/composite_4files_tt3.png`](file:///home/bim/Projects/DSP_MidTerm/output_plots/composite_4files_tt3.png)
* **Đồ thị so sánh Benchmark:** [`output_plots/benchmark_comparison_plot.png`](file:///home/bim/Projects/DSP_MidTerm/output_plots/benchmark_comparison_plot.png)

### 3. Phân tích hiện tượng đặc biệt ở từng file test
* **`phone_M2` ($\text{MAE} = 0.0\text{ ms}$ ở TT1):**
  * Ground truth: $[0.53\text{ s}, 2.52\text{ s}]$. Cả hai mốc đều là bội số nguyên của hop $10\text{ ms}$ (frame 53 và frame 252). Thuật toán bắt trúng mốc nên sai số tuyệt đối bằng 0.
* **`phone_F2` ($\text{MAE} = 20 - 30\text{ ms}$):**
  * Năng lượng cuối câu giảm từ từ kèm tiếng thở nhẹ tiệm cận mức nhiễu của micro điện thoại (`phone`). Biên kết thúc bị trễ $3-4$ frame ($30-40\text{ ms}$) là giới hạn vật lý của phương pháp dựa trên năng lượng thuần túy.
* **`studio_F2` & `studio_M2` ($\text{MAE} = 5 - 10\text{ ms}$):**
  * Tỷ số tín hiệu trên nhiễu (SNR) cao, nền tĩnh lặng, biên độ bật/tắt tiếng nói dứt khoát nên độ chính xác đạt mức gần như tuyệt đối (chỉ lệch tối đa $1$ hop).

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
