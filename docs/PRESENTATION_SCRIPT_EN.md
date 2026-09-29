# Voice Activity Detection (VAD) — Academic Defense Script

**Project Title:** Comparative Evaluation of Time-Domain Voice Activity Detection Algorithms on Vietnamese Speech  
**Total Presentation Time:** 12 Minutes (4 Minutes per Student: 3 Minutes Slides + 1 Minute Live Demo)  
**Canva Presentation Deck:** [View Slides (Canva)](https://www.canva.com/d/1ORkaZ2xSMETJZo) | [Edit Slides (Canva)](https://www.canva.com/d/40P-J_xvX3G4rBQ)

---

## Overall Agenda & Time Budget

| Section | Speaker | Topic | Slides | Slide Time | Demo Time | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Part 1** | **Student 1** | Introduction, Pre-processing & Global Fixed Threshold (TT1) | Slides 1–5 | 3 min 00 s | 1 min 00 s | 4 min 00 s |
| **Part 2** | **Student 2** | Unsupervised Adaptive Histogram-Based VAD (TT2) | Slides 6–10 | 3 min 00 s | 1 min 00 s | 4 min 00 s |
| **Part 3** | **Student 3** | Parametric Bayes Classifier (TT3) & Multi-Algorithm Benchmark | Slides 11–16 | 3 min 00 s | 1 min 00 s | 4 min 00 s |

---

# Part 1: Student 1 — Foundations & Fixed Global Energy Threshold (TT1)

### Speaker Profile & Objective
* **Speaker:** Student 1 (Signal Processing Lead)
* **Coverage:** Slides 1 to 5 (Intro, Pre-processing, Ground-Truth, TT1 Formulation, TT1 Evaluation)
* **Allocated Time:** 3:00 (Presentation) + 1:00 (Live Demo) = 4:00

---

### [0:00 - 0:35] Slide 1 & 2: Introduction, Motivation & Pre-Processing Pipeline
* **Slide 1:** Title Cover  
* **Slide 2:** Signal Pre-Processing & Feature Extraction  
* **Action:** Stand upright, make eye contact with the evaluation panel, and advance to Slide 2.

> *"Distinguished professors and members of the examination committee, good morning.*  
>  
> *Today, our group presents a rigorous comparative evaluation of three Voice Activity Detection algorithms implemented entirely on time-domain representations for Vietnamese speech.*  
>  
> *Looking at Slide 2, reliable speech endpointing forms the foundational front-end of modern speech processing pipelines. To establish a standardized benchmark, all raw acoustic signals are normalized to a uniform 16 kHz sampling rate and segmented using a rectangular window of 20 milliseconds—equivalent to 320 samples—with a 10-millisecond frame shift, yielding a 50% overlap.*  
>  
> *For each analysis frame $m$, we extract the short-time root-mean-square energy $E(m)$. This framing scheme guarantees sufficient quasi-stationarity while preserving temporal boundary precision at a 10-millisecond step."*

---

### [0:35 - 1:10] Slide 3: Ground-Truth Protocol & Evaluation Metrics
* **Slide 3:** Ground-Truth Protocol & Quantitative Metrics  
* **Action:** Point to the formal mathematical definitions of MAE and RMSE.

> *"Moving to Slide 3: To eliminate subjectivity, our ground-truth boundaries $(t_{start}, t_{end})$ were established by manual acoustic-phonetic alignment in Praat, cross-referenced with wideband spectrograms and fundamental frequency tracks.*  
>  
> *To evaluate algorithmic accuracy against these reference timestamps, we employ two quantitative metrics:*  
> 1. *First, **Mean Absolute Error (MAE)**, measuring the average absolute deviation across both leading and trailing boundaries in milliseconds.*  
> 2. *Second, **Root Mean Squared Error (RMSE)**, which imposes an $L_2$ penalty to highlight severe boundary dropouts or false trigger outliers.*  
>  
> *An algorithm is judged superior if and only if it minimizes both MAE and RMSE without manual file-by-file recalibration."*

---

### [1:10 - 2:05] Slide 4: Algorithm 1 (TT1) — Hodgkinson Binary Search & Global Optimization
* **Slide 4:** Algorithm 1: Fixed Global Energy Threshold (Hodgkinson 2012)  
* **Action:** Highlight the cost function and the empirical training sweep.

> *"On Slide 4, we examine our baseline: Algorithm 1, adapted from Hodgkinson (2012).*  
>  
> *The operational hypothesis here is straightforward: speech frames possess significantly higher short-time energy than background noise. If an optimal global scalar threshold $T_{opt}$ exists, binary classification can be achieved via a step function:*  
> $$\hat{y}(m) = \mathbb{I}(E(m) \ge T)$$  
>  
> *To find this optimal scalar without ad-hoc guesswork, we formulated a bounded optimization problem over four diverse training recordings—`phone_F1`, `phone_M1`, `studio_F1`, and `studio_M1`. By running a high-resolution grid search with 500 iterations over the interval $T \in [0.0001, 0.05]$ to minimize cumulative boundary error, the objective function reached its global minimum at:*  
> $$T_{opt} = 0.0021 \quad (\text{aligned with textbook baseline } T \approx 0.0025)$$  
>  
> *Any frame exceeding $0.0025$ is preliminarily flagged as speech, followed by a 200 ms morphological median filter to eliminate transient spurious clicks."*

---

### [2:05 - 3:00] Slide 5: TT1 Empirical Results, Critical Analysis & Failure Modes
* **Slide 5:** TT1 Empirical Evaluation on Test Set  
* **Action:** Gesture to the 4-panel multi-file plot on the right side of the slide.

> *"Slide 5 summarizes TT1's performance across our unseen test set:*  
> *On `phone_M2`, TT1 achieved an exact match with zero-millisecond error. Across all four test files, TT1 delivers a Mean Absolute Error of **8.75 milliseconds** and an RMSE of **11.34 milliseconds**.*  
>  
> *However, from an academic standpoint, we must highlight TT1's critical architectural vulnerability:*  
> *TT1 relies on a static global threshold. When the acoustic channel changes—for example, switching from clean condenser microphones to lossy cellular codecs with varying ambient noise floors—a fixed threshold cannot adjust. If the background noise floor rises above $0.0025$, TT1 collapses into catastrophic false alarms.*  
>  
> *This inherent limitation directly motivates our investigation into adaptive, unsupervised techniques, which Student 2 will now explain."*

---

### [3:00 - 4:00] Student 1: Live Interactive Demo (1 Minute)
* **Terminal Command:**  
  ```bash
  uv run python main.py --algo tt1
  ```
* **Verbal Commentary during Terminal Execution:**

> *"To substantiate these findings, I will now execute our standardized `uv` pipeline for Algorithm 1 in real-time.*  
>  
> *(Runs command: `uv run python main.py --algo tt1`)*  
>  
> *As observed on the terminal output:*  
> *The 4 evaluation files are processed synchronously. You can observe the console reporting MAE values: 20 ms for `phone_F2`, exactly 0.0 ms for `phone_M2`, 10 ms for `studio_F2`, and 5 ms for `studio_M2`.*  
>  
> *On screen, four independent diagnostic windows have popped up at the four display quadrants:*  
> *Notice the red dashed vertical lines denoting the Ground-Truth Praat boundaries, and the solid blue lines denoting TT1's predictions. The alignment is nearly coincident on high-SNR files, confirming an 8.75 ms average error.*  
>  
> *I will now hand over to Student 2 to discuss the unsupervised histogram methodology."*

---

# Part 2: Student 2 — Unsupervised Adaptive Histogram Thresholding (TT2)

### Speaker Profile & Objective
* **Speaker:** Student 2 (Algorithm & Optimization Specialist)
* **Coverage:** Slides 6 to 10 (Histogram Concept, Smoothing Filter, Threshold Formula, TT2 Experiments, Hyperparameter Sweep)
* **Allocated Time:** 3:00 (Presentation) + 1:00 (Live Demo) = 4:00

---

### [0:00 - 0:40] Slide 6 & 7: The Unsupervised Adaptive Paradigm & Histogram Smoothing
* **Slide 6:** Algorithm 2 Cover — Adaptive Histogram Thresholding (Giannakopoulos 2014)  
* **Slide 7:** Energy Distribution & 5-Bin Moving Average Smoothing  
* **Action:** Transition smoothly from Student 1, advance to Slide 7, and highlight the bimodal assumption.

> *"Thank you, Student 1. Good morning professors.*  
>  
> *While TT1 delivers low error on calibrated signals, real-world deployment requires unsupervised adaptability without prior manual training.*  
>  
> *On Slide 7, we implement Algorithm 2, based on the statistical histogram method proposed by Giannakopoulos (2014).*  
> *The underlying premise is that any speech utterance contains two prominent acoustic modes: a low-energy mode corresponding to silence and background noise, and a higher-energy mode corresponding to active phonation.*  
>  
> *However, raw short-time energy histograms are heavily corrupted by local quantization noise. To extract stable distribution peaks, we compute a 100-bin histogram and convolve it with a symmetric moving average smoothing filter of window length $W=5$:*  
> $$H_{smooth}[k] = \frac{1}{W} \sum_{j=-(W-1)/2}^{(W-1)/2} H_{raw}[k+j]$$  
>  
> *This step dampens spurious fluctuations, allowing reliable peak detection."*

---

### [0:40 - 1:25] Slide 8: Mathematical Formulation of the Adaptive Threshold
* **Slide 8:** Bimodal Peak Detection & Weighting Factor Equation  
* **Action:** Point out $M_1, M_2$, and explain the weighting parameter $W$.

> *"Turning to Slide 8, let us inspect the mathematical formulation:*  
> *From the smoothed distribution $H_{smooth}$, we locate two distinct local maxima:*  
> 1. *$M_1$: The silence and ambient noise mode, located near zero energy.*  
> 2. *$M_2$: The primary voiced speech mode, situated at higher energy.*  
>  
> *The adaptive threshold $T_{adapt}$ is computed per-utterance via a weighted convex combination:*  
> $$T = \frac{W \cdot M_1 + M_2}{W + 1}$$  
>  
> *In Giannakopoulos's canonical literature, the weight parameter is set to $W = 5.0$.*  
> *This heavily biases the threshold towards $M_1$ by assigning a 5-to-1 importance weight to the noise peak. Because $M_1$ and $M_2$ are recomputed for every individual audio file, the threshold automatically floats with the recording's dynamic range—requiring zero pre-training."*

---

### [1:25 - 2:15] Slide 9: TT2 Test Evaluation & The Unvoiced Phoneme Truncation Issue
* **Slide 9:** TT2 Empirical Evaluation on Test Set  
* **Action:** Direct attention to the right-hand plot showing `studio_F2` and `studio_M2`.

> *"Slide 9 reveals our empirical findings on the unseen test set:*  
> *On telephone speech—`phone_M2`—TT2 performed remarkably well, achieving a 10.0 ms MAE.*  
> *However, on studio recordings, TT2 suffered noticeable boundary degradation: 80.0 ms on `studio_F2` and 35.0 ms on `studio_M2`, resulting in an overall average MAE of **46.25 milliseconds** and an RMSE of **57.07 milliseconds**.*  
>  
> *Why did this occur?*  
> *Our rigorous acoustic analysis identified a key structural explanation:*  
> *In clean studio environments, the dynamic contrast between vowels and unvoiced consonants (such as fricatives /s/, /f/ and stops /t/, /k/) is immense. Setting $W=5.0$ yielded adaptive thresholds between $0.068$ and $0.093$—substantially higher than TT1's $0.0025$. Consequently, low-energy trailing unvoiced phonemes fell below the decision boundary and were truncated prematurely."*

---

### [2:15 - 3:00] Slide 10: Hyperparameter Investigation & Algorithmic Trade-offs
* **Slide 10:** Hyperparameter Investigation & Optimization Analysis  
* **Action:** Point to the hyperparameter sweep curves ($W \in [1, 20]$).

> *"To verify whether this degradation was inherent to the method or simply a parameter mismatch, we conducted a systematic sweep on Slide 10 across weight parameters $W \in [1, 20]$ and bin counts $B \in [50, 200]$.*  
>  
> *The empirical evidence is conclusive:*  
> *Increasing $W$ beyond 12 pulls the threshold closer to $M_1$, recovering unvoiced phoneme boundaries on clean audio. However, in noisy conditions, a large $W$ causes the threshold to dip below ambient noise spikes, triggering false alarms.*  
>  
> *Thus, TT2's true trade-off is clear: it offers genuine, training-free adaptability across variable recording conditions, but requires careful tuning of the weighting parameter $W$ depending on the acoustic signal-to-noise ratio.*  
>  
> *Next, Student 3 will present our statistical approach using Gaussian Bayes classification."*

---

### [3:00 - 4:00] Student 2: Live Interactive Demo (1 Minute)
* **Terminal Command:**  
  ```bash
  uv run python main.py --algo tt2
  ```
* **Verbal Commentary during Terminal Execution:**

> *"I will now demonstrate Algorithm 2 in live execution using our `uv` CLI framework.*  
>  
> *(Runs command: `uv run python main.py --algo tt2`)*  
>  
> *Looking at the live console:*  
> *Notice that for each individual file, the algorithm computes its own unique adaptive threshold: $0.0683$ for `phone_F2`, $0.0717$ for `phone_M2`, $0.0933$ for `studio_F2`, and $0.0783$ for `studio_M2`.*  
>  
> *Looking at the generated multi-window visualizations:*  
> *Observe the four subplots. On `phone_M2` (top-right), the blue predicted line tracks the red ground truth with only 10 ms discrepancy.*  
> *Conversely, on `studio_F2` (bottom-left), you can clearly see the trailing edge terminating approximately 80 ms before the red reference line, illustrating the unvoiced energy truncation we analyzed on Slide 9.*  
>  
> *I now invite Student 3 to present Algorithm 3 and the comprehensive multi-algorithm benchmark."*

---

# Part 3: Student 3 — Parametric Gaussian Bayes (TT3) & Comprehensive Benchmark

### Speaker Profile & Objective
* **Speaker:** Student 3 (Statistical Modeling Lead & Project Coordinator)
* **Coverage:** Slides 11 to 16 (Bayes Framework, Gaussian Estimation, Test Results, Comparative Benchmark, Demo Summary, Conclusion)
* **Allocated Time:** 3:00 (Presentation) + 1:00 (Live Demo) = 4:00

---

### [0:00 - 0:40] Slide 11 & 12: Parametric Bayes Decision Theory & Parameter Estimation
* **Slide 11:** Algorithm 3 Cover — Statistical Gaussian Bayes Classifier  
* **Slide 12:** Parametric Maximum Likelihood & Bayes Decision Boundary  
* **Action:** Transition professionally from Student 2, advance to Slide 12, and highlight the statistical equations.

> *"Thank you, Student 2. Good morning members of the committee.*  
>  
> *To complete our comparative spectrum, Algorithm 3 reformulates Voice Activity Detection from a formal statistical pattern classification perspective.*  
>  
> *On Slide 12, we model the frame energy $x = E(m)$ as originating from two underlying Gaussian distributions: silence ($\mathcal{C}_0$) and speech ($\mathcal{C}_1$):*  
> $$p(x \mid \mathcal{C}_i) = \frac{1}{\sqrt{2\pi}\sigma_i} \exp\left(-\frac{(x - \mu_i)^2}{2\sigma_i^2}\right), \quad i \in \{0, 1\}$$  
>  
> *Using our annotated training corpus of 1,295 frames (497 silence frames and 798 speech frames), we computed the Maximum Likelihood parameter estimates:*  
> *For silence: $\mu_0 = 0.000328$, $\sigma_0 = 0.000431$*  
> *For speech: $\mu_1 = 0.196029$, $\sigma_1 = 0.233180$*  
>  
> *Under equal prior probabilities, the minimum-error Bayes decision rule solves $p(x \mid \mathcal{C}_0) = p(x \mid \mathcal{C}_1)$, yielding the theoretical closed-form threshold:*  
> $$T_{Bayes} \approx 0.002864 \quad (\text{empirically tuned to } 0.001900)"$*

---

### [0:40 - 1:25] Slide 13 & 14: Benchmark Analysis & Balanced Methodological Critique
* **Slide 13:** TT3 Experimental Results  
* **Slide 14:** Comprehensive 3-Algorithm Benchmark & Quantitative Trade-offs  
* **Action:** Point out the side-by-side comparison table and the bar chart on Slide 14.

> *"Slide 13 and Slide 14 present our definitive comparative benchmark across all three algorithms on identical test data:*  
>  
> *Looking at the summary table on Slide 14:*  
> *Both TT1 (Binary Search) and TT3 (Gauss Bayes) achieve an identical average MAE of **8.75 milliseconds** and an RMSE of **11.34 milliseconds**, closely approaching the physical 10 ms frame discretization limit.*  
> *TT2 (Histogram) delivers a higher average MAE of **46.25 milliseconds** due to standard parameter bias on studio data.*  
>  
> *However, we emphasize a crucial, objective critique:*  
> *While TT3 exhibits high quantitative accuracy here, its core theoretical assumption is fundamentally flawed from a physical standpoint:*  
> *Energy is strictly non-negative and heavily right-skewed, violating the symmetry and infinite support of a Gaussian distribution. TT3 functions effectively only because the silence and voiced speech modes are separated by several orders of magnitude in clean training recordings. In heavy non-stationary noise, a Gaussian model degrades significantly unless generalized to Chi-Square, Gamma, or Gaussian Mixture Models (GMM)."*

---

### [1:25 - 2:15] Slide 15: Software Engineering, Reproducibility & Pipeline Architecture
* **Slide 15:** Software Architecture, Live Demonstration & Engineering Highlights  
* **Action:** Emphasize software modularity and strict `uv` package isolation.

> *"On Slide 15, we highlight the engineering standards governing this project:*  
> *Our codebase is organized into fully decoupled modules:*  
> *`audio_io.py` handles lossless audio loading and Praat ground-truth parsing;*  
> *`vad_core.py` implements the three mathematical detectors; and*  
> *`evaluator.py` computes standardized MAE and RMSE metrics.*  
>  
> *To guarantee 100% reproducibility across any host operating system, all dependencies are managed strictly via the `uv` package orchestrator, completely eliminating system environment pollution.*  
> *A unified command-line interface allows instant parameter sweeping, training, and headless batch evaluations."*

---

### [2:15 - 3:00] Slide 16: Key Takeaways & Concluding Remarks
* **Slide 16:** Conclusions, Final Recommendations & Future Work  
* **Action:** Deliver concise, confident concluding statements and open the floor for Q&A.

> *"To conclude on Slide 16, our findings provide clear engineering guidelines for time-domain VAD selection:*  
> 1. *For stationary, pre-calibrated acoustic environments, a fixed global energy threshold (TT1) or Bayes threshold (TT3) offers maximum temporal precision with near-zero computational overhead.*  
> 2. *For uncalibrated acoustic channels where training data is unavailable, adaptive histogram thresholding (TT2) is the only viable unsupervised option, provided that the weighting factor $W$ is tuned dynamically.*  
> 3. *For future extensions, incorporating spectral centroid and Zero-Crossing Rate (ZCR) will resolve TT2's unvoiced phoneme truncation.*  
>  
> *Thank you very much for your kind attention. I will now run the live demonstration for Algorithm 3 and the consolidated benchmark."*

---

### [3:00 - 4:00] Student 3: Live Interactive Demo & Comprehensive Output (1 Minute)
* **Terminal Command:**  
  ```bash
  # Step 1: Run live TT3 visual evaluation
  uv run python main.py --algo tt3
  
  # Step 2: In headless mode, generate the complete consolidated report
  uv run python show_all_results.py
  ```
* **Verbal Commentary during Terminal Execution:**

> *"To conclude our presentation, I will execute Algorithm 3 and display the comprehensive quantitative results.*  
>  
> *(Runs command: `uv run python main.py --algo tt3`)*  
>  
> *As the diagnostic windows appear:*  
> *Notice that TT3's parametric Bayes threshold of $0.002864$ produces clean, jitter-free segmentation boundaries across all four test files.*  
>  
> *(Runs command: `uv run python show_all_results.py`)*  
>  
> *Here on the terminal is our comprehensive audit report:*  
> *It displays the exact Gaussian statistics: silence variance $\sigma_0 = 0.000431$, speech variance $\sigma_1 = 0.233180$, the global optimal threshold $T_{opt} = 0.002100$, and the side-by-side performance matrix.*  
>  
> *This concludes our demonstration. We are now ready to address any questions from the committee."*

---

## Quick Reference Sheet for Defense Q&A

| Question | Recommended Answer |
| :--- | :--- |
| **Why did TT2 have higher MAE on studio data?** | Clean studio recordings exhibit wide dynamic range between vowels and weak unvoiced phonemes (/s/, /f/, /t/). The canonical weight $W=5.0$ sets the threshold too high ($\approx 0.08$), truncating low-energy trailing unvoiced consonants. |
| **Is Gaussian distribution appropriate for energy?** | Theoretically no, because energy $E(m) \ge 0$ is strictly positive and non-symmetric. However, in high SNR, the silence and speech clusters are so widely separated that a Gaussian Bayes decision boundary still yields excellent practical classification ($T \approx 0.0028$). For lower SNR, Gamma or Rayleigh distributions would be theoretically sounder. |
| **Why are TT1 and TT3 errors identical on test data?** | Because the optimal Hodgkinson threshold ($T = 0.0025$) and the empirical Bayes threshold ($T = 0.0019 - 0.0028$) both fall cleanly within the wide energy gap separating silence background from speech on these 4 test recordings. Since decisions are filtered through a 10 ms frame shift, minor threshold differences produce identical frame assignments. |
