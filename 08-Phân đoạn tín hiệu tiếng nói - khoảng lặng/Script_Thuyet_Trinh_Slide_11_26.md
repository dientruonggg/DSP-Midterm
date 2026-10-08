# BÁO CÁO GIỮA KỲ XỬ LÝ TÍN HIỆU SỐ (DSP 2026) — NHÓM 08

## KỊCH BẢN THUYẾT TRÌNH CHI TIẾT: VOICE ACTIVITY DETECTION (VAD)

**Phạm vi:** Từ Slide 11 đến Slide 26 (Thuật toán 3, Thực nghiệm đối sánh, Critique & Phản biện Q&A)
**Người trình bày phân đoạn:** Trương Bùi Điền (và nhóm 08)

---

### HƯỚNG DẪN KỸ NĂNG NÓI TO, RÕ RÀNG VÀ TỰ TIN TRƯỚC HỘI ĐỒNG:

* `//` : Điểm ngắt nghỉ ngắn (0.5 – 1 giây) để **lấy hơi bằng bụng (hít sâu phình bụng)**, giúp giọng to, dày và không bao giờ bị hụt hơi.
* **IN ĐẬM VIẾT HOA** hoặc **in đậm**: Từ khóa kỹ thuật trọng tâm cần **nhấn mạnh dằn giọng dứt khoát**.
* `[Hành động & Chỉ slide]`: Chỉ dẫn cử chỉ tay, mắt quét và vị trí chỉ vào màn hình slide.
* `Bản chất kỹ thuật (Deep-dive)`: Diễn giải trực quan cốt lõi để tự tin trả lời vấn đáp của Thầy/Cô.

---

### SLIDE 11: STUDENT 3 · ALGORITHM 03

*(Parametric Gaussian Bayes Model — Trương Bùi Điền)*

* **Thời lượng dự kiến:** 35 – 45 giây
* **Mục tiêu:** Chuyển giao quyền trình bày, định vị phương pháp tiếp cận xác suất thống kê cổ điển.
* **Hành động & Cử chỉ:** Bước lên nửa bước, mắt nhìn thẳng Hội đồng, mỉm cười tự tin, tay mở hướng về slide.

> **LỜI THOẠI TRỰC TIẾP:**
> "Kính thưa Thầy và các bạn! // Tiếp nối hai thuật toán tìm kiếm nhị phân và lược đồ thích nghi Histogram của Toàn và Hoàng, // em là **Trương Bùi Diễn**, // sau đây xin đại diện nhóm trình bày **Thuật toán 03: Mô hình phân lớp Bayes tham số Gaussian (Parametric Gaussian Bayes Model)**. //
>
> Nếu như hai thuật toán trước mang tính **kinh nghiệm thực nghiệm**, // thì Thuật toán 3 tiếp cận bài toán VAD dưới lăng kính **Nhận dạng mẫu thống kê cổ điển (Statistical Pattern Recognition)**. // Ý tưởng cốt lõi là coi năng lượng của **khoảng lặng** và **tiếng nói** là hai phân phối Gaussian độc lập. // Từ tập huấn luyện, ta ước lượng tham số $\mu$ và $\sigma$, // rồi **giải phương trình bậc hai giải tích** để tìm ngưỡng quyết định tối ưu Bayes. //
>
> Ưu điểm lớn nhất là: // Ngưỡng phân tách hoàn toàn **có cơ sở toán học xác suất vững chắc**, // không phụ thuộc vào việc mò mẫm tham số! //"

*Bản chất kỹ thuật:* Mô hình giả định biến ngẫu nhiên $x$ (Short-Time Energy chuẩn hóa) tuân theo hai hàm mật độ xác suất chuẩn $p(x|\text{Sil}) \sim \mathcal{N}(\mu_{sil}, \sigma_{sil}^2)$ và $p(x|\text{Sp}) \sim \mathcal{N}(\mu_{sp}, \sigma_{sp}^2)$.

---

### SLIDE 12: ALG 3 · CORE METHOD

*(From Empirical Statistics to Analytical Bayes Root)*

* **Thời lượng dự kiến:** 60 – 75 giây
* **Mục tiêu:** Giải thích chi tiết 6 bước từ trích xuất STE đến giải nghiệm phương trình bậc 2.
* **Hành động & Cử chỉ:** Tay quét ngang theo 6 khối từ trái sang phải, sau đó chỉ vào từng tham số bên dưới.

> **LỜI THOẠI TRỰC TIẾP:**"Trên màn hình là **quy trình xử lý 6 bước khép kín** của mô hình Bayes: //
>
> 1. **Bước 1**: Trích xuất năng lượng ngắn hạn chuẩn hóa STE trên **4 file huấn luyện**. //
> 2. **Bước 2**: Dựa vào nhãn Praat Ground-Truth, tách các khung thành hai tập: **Khoảng lặng (Silence)** và **Tiếng nói (Speech)**. //
> 3. **Bước 3**: Khớp hai hàm mật độ Gaussian: //
>    - Khoảng lặng có trung bình cực nhỏ: **$\mu_{sil} = 0.00039$**, độ lệch rất hẹp: **$\sigma_{sil} = 0.00071$**. //
>    - Tiếng nói có trung bình lên tới: **$\mu_{sp} = 0.20265$**, và $\sigma_{sp} = 0.23563$. //
>      Sự cách biệt năng lượng này lên tới **hơn 500 lần**! //
> 4. **Bước 4 & 5**: Thiết lập phương trình đẳng xác suất: $p(x | \text{Sil}) = p(x | \text{Sp})$. Khai triển logarit, ta được phương trình bậc hai dạng chuẩn: **$Ax^2 + Bx + C = 0$**. //
> 5. **Bước 6**: Giải phương trình, ta tìm được nghiệm dương duy nhất giữa hai kỳ vọng: **$T_{\text{Bayes}} \approx 0.00288$**! // Sau đó áp dụng bộ lọc bắc cầu **200 ms** để loại bỏ các khoảng dừng sinh lý. //"

*Bản chất kỹ thuật:* Nghiệm $T_{\text{Bayes}}$ là điểm mà tại đó xác suất phân loại sai thành Speech bằng đúng xác suất phân loại sai thành Silence (Equiprobable Bayesian root), giúp cực tiểu hóa hàm tổn thất 0-1 (Minimum Error Rate Classification).

---

### SLIDE 13: ALG 3 · INTERMEDIATE RESULTS

*(Parametric Gaussian Fit & Bayes Intersection)*

* **Thời lượng dự kiến:** 40 – 50 giây
* **Mục tiêu:** Minh họa hình học giao điểm xác suất và chứng minh tốc độ thực thi tức thời O(1).
* **Hành động & Cử chỉ:** Chỉ vào đỉnh Gaussian nhọn bên trái, sau đó trỏ vào tọa độ giao điểm T.

> **LỜI THOẠI TRỰC TIẾP:**
> "Nhìn vào phân phối hình học, Thầy và các bạn có thể thấy: // Phân phối của khoảng lặng có dáng hình **rất nhọn và hẹp**, ôm sát trục 0. // Trong khi phân phối tiếng nói lại trải rộng về phía bên phải. // Giao điểm xác suất rơi chính xác vào phần đuôi dưới (lower tail) của tiếng nói tại tọa độ **$T \approx 0.00288$**. //
>
> Về mặt hình học quyết định (Decision Geometry): // Vùng diện tích chồng lấn (overlap error area) giữa hai phân phối là **cực kỳ nhỏ**, // nghĩa là xác suất nhận dạng sai theo lý thuyết Bayes đã được **tối thiểu hóa**. //
>
> Đặc biệt, sau khi đã tìm được nghiệm T ở pha huấn luyện, // ở pha kiểm thử thực tế, thuật toán chỉ cần **so sánh một phép toán $O(1)$ trên từng frame**. // Tốc độ xử lý tức thời và tiêu tốn bộ nhớ gần như bằng 0! //"

---

### SLIDE 14: ALG 3 · TEST EXECUTION

*(Test Boundary Detection & Error - 4 Test Files)*

* **Thời lượng dự kiến:** 45 – 50 giây
* **Mục tiêu:** Báo cáo sai số định lượng trên 4 file kiểm thử độc lập và phân tích hiện tượng âm học.
* **Hành động & Cử chỉ:** Chỉ vào từng hàng kết quả định lượng, nhấn mạnh con số 7.5 ms.

> **LỜI THOẠI TRỰC TIẾP:**"Kiểm tra mô hình trên **4 file test hoàn toàn mới** (gồm cả môi trường phòng thu Studio và điện thoại thoại Phone): //
>
> - File **phone_F2**: sai số MAE là **27.5 ms**. //
> - File **phone_M2**: MAE đạt mức ấn tượng: **7.5 ms**. //
> - File **studio_F2**: đạt **7.5 ms**. //
> - File **studio_M2**: cũng đạt **7.5 ms**. //
> - Trung bình MAE toàn tập test đạt **12.50 ms**! //
>
> **Nhận xét âm học quan trọng**: // Có tới **3 trên 4 file test** đạt sai số chỉ **7.5 ms**, tức là **dưới 1 bước dịch khung (hop size 10 ms)**! // Ranh giới ở môi trường phòng thu được cắt cực kỳ sắc nét. //
>
> Riêng file phone_F2, sai số điểm kết thúc trễ **+45 ms**. // Lý do là người nữ phát âm có **tiếng thở hắt ra ở cuối câu (trailing breath)** mang năng lượng lớn hơn $T_{\text{Bayes}}$, khiến thuật toán kéo dài thêm 4 frame trước khi hạ xuống khoảng lặng. //"

---

### SLIDE 15: TEST · PHONE F2

*(Test Case 1: phone_F2.wav · Comparative VAD)*

* **Thời lượng dự kiến:** 40 – 45 giây
* **Hành động:** Chỉ vào cột $\Delta\text{End}$ và MAE của 3 thuật toán trong bảng Scorecard.

| Thuật toán                   | $\Delta\text{Start}$ (ms) | $\Delta\text{End}$ (ms) |     MAE (ms)     |
| ------------------------------ | :-------------------------: | :-----------------------: | :---------------: |
| **TT1 (Fixed STE)**      |           -10 ms           |          +15 ms          | **12.5 ms** |
| **TT2 (Adaptive Hist)**  |           +20 ms           |          -15 ms          | **17.5 ms** |
| **TT3 (Gaussian Bayes)** |           -10 ms           |          +45 ms          | **27.5 ms** |

> **LỜI THOẠI TRỰC TIẾP:**"Ở trường hợp kiểm thử số 1: `phone_F2.wav` — kênh thoại băng hẹp 8 kHz, thời lượng 3.02 giây. //
>
> - **TT1** đạt MAE: **12.5 ms** (đầu lệch -10 ms, cuối lệch +15 ms). //
> - **TT2** đạt MAE: **17.5 ms**. //
> - **TT3** đạt MAE: **27.5 ms** (điểm cuối trễ +45 ms như vừa phân tích). //
>
> Tại sao TT1 lại ít trễ hơn TT3? // Bởi vì ngưỡng tĩnh của TT1 (0.00348) cao hơn ngưỡng Bayes (0.00288). Do đó TT1 đã 'vô tình' cắt đứt tiếng thở sớm hơn, // trong khi TT3 nhạy hơn nên bắt trọn cả phần đuôi hơi thở. //"

---

### SLIDE 16: TEST · PHONE M2

*(Test Case 2: phone_M2.wav · Comparative VAD)*

* **Thời lượng dự kiến:** 35 – 40 giây
* **Hành động:** Chỉ vào con số 7.5 ms của TT1 và TT3.

| Thuật toán                   | $\Delta\text{Start}$ (ms) | $\Delta\text{End}$ (ms) |     MAE (ms)     |
| ------------------------------ | :-------------------------: | :-----------------------: | :---------------: |
| **TT1 (Fixed STE)**      |           -10 ms           |           -5 ms           | **7.5 ms** |
| **TT2 (Adaptive Hist)**  |           +20 ms           |           +5 ms           | **12.5 ms** |
| **TT3 (Gaussian Bayes)** |           -10 ms           |           +5 ms           | **7.5 ms** |

> **LỜI THOẠI TRỰC TIẾP:**"Sang trường hợp thứ hai: `phone_M2.wav` — giọng nam qua kênh thoại. //Ở bản ghi này, năng lượng giọng nói phân tách cực kỳ dứt khoát: //
>
> - Cả **TT1** và **TT3** đều đạt kết quả xuất sắc: MAE chỉ **7.5 ms**! // Điểm đầu lệch đúng 1 hop (-10 ms), điểm cuối lệch vỏn vẹn nửa hop (+5 ms). //
> - **TT2** với cơ chế thích nghi hai đặc trưng đạt MAE là **12.5 ms**. //
>
> Cả ba thuật toán đều bắt trúng ranh giới, chứng minh rằng khi âm thanh dứt khoát, hiện tượng trôi ranh giới hoàn toàn biến mất! //"

---

### SLIDE 17: TEST · STUDIO F2

*(Test Case 3: studio_F2.wav · Comparative VAD)*

* **Thời lượng dự kiến:** 35 – 40 giây
* **Hành động:** Chỉ vào kết quả 7.5 ms của TT3 so với 12.5 ms của TT1 và 17.5 ms của TT2.

| Thuật toán                   | $\Delta\text{Start}$ (ms) | $\Delta\text{End}$ (ms) |     MAE (ms)     |
| ------------------------------ | :-------------------------: | :-----------------------: | :---------------: |
| **TT1 (Fixed STE)**      |           -10 ms           |          -15 ms          | **12.5 ms** |
| **TT2 (Adaptive Hist)**  |           -20 ms           |          -15 ms          | **17.5 ms** |
| **TT3 (Gaussian Bayes)** |           -10 ms           |           -5 ms           | **7.5 ms** |

> **LỜI THOẠI TRỰC TIẾP:**"Trường hợp thứ ba: `studio_F2.wav` — giọng nữ trong môi trường **phòng thu Studio có SNR rất cao**. //Kết quả thể hiện rõ ưu thế lý thuyết: //
>
> - **TT3** vượt trội với MAE chỉ **7.5 ms** (điểm kết thúc chỉ lệch -5 ms). //
> - Trong khi đó, **TT1** đạt **12.5 ms**, // và **TT2** đạt **17.5 ms** do cả hai đều cắt sớm ở cả hai đầu (-15 ms đến -20 ms). //
>
> Trong phòng thu tĩnh, phân phối của khoảng lặng hầu như không bị nhiễu nền làm méo, giúp ngưỡng Bayes phát huy tối đa độ nhạy lý thuyết! //"

---

### SLIDE 18: TEST · STUDIO M2

*(Test Case 4: studio_M2.wav · Comparative VAD)*

* **Thời lượng dự kiến:** 35 – 40 giây
* **Hành động:** Chỉ vào con số 2.5 ms và Delta Start = 0 ms của TT2.

| Thuật toán                   | $\Delta\text{Start}$ (ms) | $\Delta\text{End}$ (ms) |     MAE (ms)     |
| ------------------------------ | :-------------------------: | :-----------------------: | :--------------: |
| **TT1 (Fixed STE)**      |           +10 ms           |           +5 ms           | **7.5 ms** |
| **TT2 (Adaptive Hist)**  |           ±0 ms           |           -5 ms           | **2.5 ms** |
| **TT3 (Gaussian Bayes)** |           +10 ms           |           +5 ms           | **7.5 ms** |

> **LỜI THOẠI TRỰC TIẾP:**"Trường hợp kiểm thử cuối: `studio_M2.wav` — giọng nam môi trường phòng thu. //
>
> - Cả **TT1** và **TT3** đều duy trì sự ổn định tuyệt đối: MAE **7.5 ms**. //
> - Đặc biệt, **TT2 của bạn Hoàng** đạt mức sai số kỷ lục: **2.5 ms**! // Điểm bắt đầu trùng khớp tuyệt đối **±0 ms** với Ground-Truth, điểm kết thúc chỉ lệch đúng -5 ms! //
>
> Nhờ cơ chế đệm biên **+10 ms padding**, TT2 đã khôi phục hoàn hảo phụ âm đầu mà không làm biến dạng ranh giới tổng thể. //"

---

### SLIDE 19: BENCHMARK · 4 TEST FILES

*(Quantitative Performance Comparison Across Algorithms)*

* **Thời lượng dự kiến:** 75 – 90 giây
* **Mục tiêu:** Trình bày bảng benchmark tổng hợp định lượng toàn diện và rút ra kết luận cốt lõi.
* **Hành động:** Đứng thẳng, tay hướng trọn vẹn vào Bảng Benchmark, đọc từng chỉ số chính xác.

| Phương pháp                      |    MAE (ms)    |    RMSE (ms)    |    F1-Score    | Độ nhạy vô thanh (UV) |
| ----------------------------------- | :-------------: | :-------------: | :-------------: | :-----------------------: |
| **TT1-1 (Hodgkinson gốc)**   |      10.00      |      10.33      |      0.993      |           0.971           |
| **TT1-2 (DSP Classifier)**    | **6.25** | **7.80** | **0.996** |           0.968           |
| **TT2-1 (Dual STE-SC + Pad)** |      12.50      |      13.37      |      0.985      |           0.947           |
| **TT2-2 (Baseline No-Pad)**   |      18.75      |      19.34      |      0.977      |           0.891           |
| **TT3-1 (1D Gaussian Bayes)** | **12.50** | **14.08** | **0.994** |      **0.971**      |
| **TT3-2 (4D Multivariate)**   |      15.00      |      16.34      |      0.978      |           0.903           |

> **LỜI THOẠI TRỰC TIẾP:**"Kính thưa Thầy, đây là **bức tranh toàn cảnh định lượng** của cả ba thuật toán trên toàn bộ 4 file kiểm thử: //
>
> 1. **TT1-2 (Hodgkinson cải tiến)**: Đạt sai số biên thấp nhất: **MAE = 6.25 ms**, RMSE = 7.80 ms, điểm F1 = **0.996**. //
> 2. **TT2-1 (Histogram có đệm biên +10ms)**: Đạt MAE **12.50 ms**, cải thiện vượt bậc **33.3%** so với bản không đệm (18.75 ms). //
> 3. **TT3-1 (Gaussian 1D của em)**: Đạt MAE **12.50 ms**, F1 = **0.994**, độ nhạy âm vô thanh UV đạt **0.971**! //
>
> Một phát hiện thực nghiệm cực kỳ đắt giá: // Khi nhóm thử nghiệm phiên bản **4D Multivariate Gaussian** kết hợp phổ, // sai số lại **tăng lên 15.00 ms**, kém hơn bản 1D! //
>
> **Kết luận cốt lõi**: TT1-2 đạt sai số biên thấp nhất nhờ tối ưu trực tiếp thời lượng. Nhưng TT3 chứng minh sức mạnh toán học khi **3 trên 4 file test đạt MAE 7.5 ms** hoàn toàn bằng suy diễn giải tích đóng! //"

---

### SLIDE 20: ALG 1 · CRITIQUE

*(Algorithm 01: Strengths, Weaknesses, and Best Use Case)*

* **Thời lượng dự kiến:** 45 giây
* **Hành động:** Chia tay làm 3 hướng tương ứng 3 cột trên slide.

> **LỜI THOẠI TRỰC TIẾP:**"Phê bình chuyên sâu **Thuật toán 1 (Ngưỡng tĩnh toàn cục)**: //
>
> - **Ưu điểm**: Kiểm tra khung với độ phức tạp $O(1)$, không tốn RAM đệm, tốc độ nhanh nhất và độ chính xác ranh giới cao nhất khi nhiễu dừng. //
> - **Nhược điểm chí mạng**: Ngưỡng bị đóng cứng (rigid static). Khi môi trường có nhiễu nền biến động hoặc SNR tụt dốc, thuật toán mất hoàn toàn khả năng thích nghi. //
> - **Ứng dụng lý tưởng**: Phù hợp cho **vi điều khiển nhúng công suất thấp (MCU)**, chip IoT biên, hoặc hệ thống streaming thời gian thực trong phòng thu tĩnh. //
>
> **Bài học rút ra**: *Đơn giản chính là đỉnh cao khi môi trường nhiễu có tính dừng!* //"

---

### SLIDE 21: ALG 2 · CRITIQUE

*(Algorithm 02: Dual-Feature Strengths, Weaknesses, and Best Use Case)*

* **Thời lượng dự kiến:** 45 giây
* **Hành động:** Hướng tay sang phân tích lược đồ Histogram 2 đặc trưng.

> **LỜI THOẠI TRỰC TIẾP:**"Đối với **Thuật toán 2 (Lược đồ Histogram thích nghi)**: //
>
> - **Ưu điểm vượt trội**: Hoàn toàn **không giám sát (Unsupervised)**! Tự thích nghi trên từng câu nói nhờ kết hợp cả năng lượng thời gian (STE) và trọng tâm phổ tần số (Spectral Centroid). Không cần dữ liệu gán nhãn trước. //
> - **Nhược điểm**: Bắt buộc phân phối phải có dạng **hai đỉnh (Bimodal)**. Với câu nói quá ngắn dưới 1 giây, hoặc SNR quá thấp làm bẹp đáy thung lũng giữa 2 đỉnh, thuật toán sẽ chọn sai ngưỡng. Phải giữ toàn bộ câu vào RAM ($O(N)$). //
> - **Ứng dụng thực tế**: Cực kỳ phù hợp cho **hệ thống xử lý âm thanh tự động ngoại tuyến (Offline)**, cắt gọt khoảng lặng cho Podcast, lập chỉ mục kho âm thanh quy mô lớn. //
>
> **Bài học**: *Lược đồ thời gian - tần số mang lại ngưỡng tự thích nghi mà không cần nhãn!* //"

---

### SLIDE 22: ALG 3 · CRITIQUE

*(Model Complexity: 1D Gaussian vs 4D Covariance Matrix)*

* **Thời lượng dự kiến:** 50 giây
* **Hành động:** Nhìn thẳng Hội đồng, nhấn mạnh giọng triết lý khoa học.

> **LỜI THOẠI TRỰC TIẾP:**
> "Đối với **Thuật toán 3**, nhóm đặt ra câu hỏi học thuật: // **Liệu tăng độ phức tạp mô hình lên đa biến 4D Gaussian có giúp VAD tốt hơn không?** //
>
> Kết quả thực nghiệm chứng minh một bài học kinh điển: //
>
> - **Mô hình 1D Gaussian** chỉ có 4 tham số, nghiệm giải tích đóng ổn định tuyệt đối, đạt MAE **12.50 ms**. //
> - **Mô hình 4D Multivariate** phải ước lượng ma trận hiệp phương sai đầy đủ. Khi tập huấn luyện chỉ có 4 file (cỡ mẫu nhỏ), mô hình bị **quá khớp (Overfitting)** và gặp rủi ro số học khi nghịch đảo ma trận, khiến MAE tăng lên **15.00 ms**! //
>
> **Triết lý Dao cạo Occam (Occam’s Razor)**: // Trong điều kiện dữ liệu mẫu nhỏ, **sự tối giản tham số (Parsimony) luôn đánh bại sự phức tạp hóa mô hình**! //"

---

### SLIDE 23: PREPROCESSING · STANDARDS

*(Standard Parameters & 200 ms Silence Bridging)*

* **Thời lượng dự kiến:** 45 giây
* **Hành động:** Chỉ vào các thông số chuẩn hóa DSP trên màn hình.

> **LỜI THOẠI TRỰC TIẾP:**"Để kết quả so sánh giữa 3 thành viên đạt độ khách quan khoa học, // cả nhóm đã thống nhất một **chuẩn tiền xử lý DSP đồng nhất**: //
>
> 1. **Tần số lấy mẫu $F_s = 16\text{ kHz}$**: Chuẩn hóa toàn bộ âm thanh về mono 16 kHz. //
> 2. **Khung 25 ms** (400 mẫu): Đảm bảo tính chất chuẩn dừng (quasi-stationary) của bộ máy phát âm người. //
> 3. **Bước nhảy 10 ms** (160 mẫu): Tạo độ chồng lấn 60%, thiết lập lưới thời gian chính xác 10 ms. //
> 4. **Quy tắc bắc cầu bất biến (Bridging Invariant)**: //
>    - Khoảng lặng ngắn hơn **200 ms** sẽ bị **bắc cầu nối liền (Bridge)**, // vì về mặt ngữ âm, đó chỉ là khoảng dừng lấy hơi hoặc đóng thanh quản tạm thời giữa các âm tiết. //
>    - Đoạn tiếng nói ngắn hơn **50 ms** bị **loại bỏ hoàn toàn**, // để triệt tiêu tiếng lách cách (click noise) do micro va chạm! //"

---

### SLIDE 24: LIVE DEMO · SUBMISSION

*(Four-Quadrant Display on Teacher Laptop)*

* **Thời lượng dự kiến:** 40 giây
* **Hành động:** Chỉ vào 4 khối 01 - 02 - 03 - 04.

> **LỜI THOẠI TRỰC TIẾP:**"Về quy cách nộp bài và tổ chức thực nghiệm: //
>
> - **Khối 01**: Dữ liệu đầu vào gồm 4 file test `.wav` và file nhãn chuẩn `.lab`. Không upload âm thanh lên mạng hay lưu trữ dư thừa. //
> - **Khối 02**: Môi trường ảo được quản lý chuẩn hóa qua **công cụ `uv`**, Python 3.12, dùng thư viện toán học thuần numpy và scipy. //
> - **Khối 03**: Ba file Jupyter Notebook độc lập đều đã được chạy sẵn kết quả (Pre-run outputs), đồ thị sẵn sàng, **độ trễ demo bằng 0**! //
> - **Khối 04**: Giao diện được thiết kế theo **bố cục 4 góc phần tư (Four-Quadrant Display)** trên màn hình laptop của Thầy, giúp đối chiếu đồng thời cả dạng sóng, đường STE và nhãn phân đoạn cùng lúc! //"

---

### SLIDE 25: DEFENSE · Q&A

*(Rebuttal for Critical Examination Questions)*

* **Thời lượng dự kiến:** 80 – 90 giây
* **Hành động:** Đứng vững chãi, hướng thẳng về Thầy, nói dõng dạc, rành mạch từng câu hỏi.

> **LỜI THOẠI TRỰC TIẾP:**
> "Trước khi Thầy đặt câu hỏi, nhóm xin chủ động giải trình **4 vấn đề then chốt** mà Hội đồng thường quan tâm nhất: //
>
> **Câu hỏi 1: Tại sao nhóm không dùng tần số cơ bản $F_0$ (Pitch) để phân đoạn tiếng nói?** //
> **Trả lời**: Bởi vì các **phụ âm vô thanh (Unvoiced consonants)** như /s/, /t/, /f/ hoàn toàn **không có tần số cơ bản $F_0$**! // Nếu dùng $F_0$, chúng ta sẽ cắt cụt toàn bộ các âm đầu và âm cuối của từ, phá hủy ngữ nghĩa câu nói! //
>
> **Câu hỏi 2: Tại sao ở file `phone_M2`, có thuật toán đạt sai số gần như 0.0 ms? Có phải do Overfitting?** //
> **Trả lời**: Hoàn toàn **không phải Overfitting**! // Nhãn Ground-truth $[0.53\text{s}, 2.52\text{s}]$ trùng khớp chính xác vào khung 53 và khung 252 trên lưới bước nhảy 10 ms. Đây là hiện tượng **khớp lưới bước nhảy (Grid snapping)**, chứng minh độ phân giải lưới 10 ms hoạt động cực chuẩn xác! //
>
> **Câu hỏi 3: Mức nhiễu nền SNR ảnh hưởng như thế nào đến độ dịch ranh giới?** //
> **Trả lời**: Ở Studio (SNR cao), sai số chỉ 5–10 ms. Nhưng ở Phone (băng hẹp 8kHz, SNR thấp), tiếng thở hắt đuôi câu tạo ra độ trễ từ 20–45 ms. //
>
> **Câu hỏi 4: Tại sao 1D Gaussian lại đánh bại đa biến 4D trong TT3?** //
> **Trả lời**: Tập huấn luyện chỉ có 4 file. Ước lượng ma trận hiệp phương sai 4D gây Overfitting nặng. Mô hình 1D ít tham số hơn nên có tính **tổng quát hóa vượt trội** trên tập kiểm thử! //"

---

### SLIDE 26: DIGITAL SIGNAL PROCESSING · MIDTERM PROJECT 2026

*(Thank You & Q&A Session)*

* **Thời lượng dự kiến:** 25 – 30 giây
* **Hành động:** Cúi đầu chào nhẹ, hai tay mở hướng về Thầy và lớp học.

> **LỜI THOẠI TRỰC TIẾP:**
> "Kính thưa Thầy và toàn thể các bạn! //
> Trên đây là toàn bộ báo cáo kết quả nghiên cứu đề tài **Phân đoạn tín hiệu tiếng nói – khoảng lặng (Voice Activity Detection)** của Nhóm 08. //
> Mã nguồn, sổ tay tính toán Notebook và các biểu đồ định lượng đều đã sẵn sàng trong thư mục nộp bài. //
> Nhóm chúng em đã sẵn sàng cho phần **Live Demo trực tiếp trên bất kỳ file âm thanh nào** mà Thầy yêu cầu, // và rất mong nhận được những nhận xét, góp ý quý báu từ Thầy! //
>
> **Em xin trân trọng cảm ơn Thầy và các bạn đã chú ý lắng nghe!** //"
