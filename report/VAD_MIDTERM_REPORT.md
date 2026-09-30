# Speech–Silence Segmentation Using Short-Time DSP Features

**Digital Signal Processing – Midterm Project**  
**Group members:** `[Student name – ID]`, `[Student name – ID]`, `[Student name – ID]`

## 1. Problem and required output

The input is a recorded speech signal. The output is the estimated start and end of the speech region. Each test recording must be processed in one run and shown in one figure with:

- the waveform and short-time feature;
- the ground-truth boundaries from the LAB file;
- the automatically detected boundaries;
- the F0 track as a voiced-speech interpretation aid;
- boundary error in milliseconds.

All methods use 20 ms frames, a 10 ms hop, and the required rule that a real silence interval must last at least 200 ms. A shorter internal zero run is treated as a false silence and is bridged.

## 2. Data and experimental protocol

The supplied data contain four training recordings and four independent test recordings. The labels are `sil` (silence), `v` (voiced speech), and `uv` (unvoiced speech). Both `v` and `uv` are speech for the final binary segmentation.

| Split | Recordings | Role |
|---|---|---|
| Training | `phone_F1`, `phone_M1`, `studio_F1`, `studio_M1` | Estimate thresholds and model parameters |
| Validation | `studio_M1` held out during model selection | Check parameters without looking at test data |
| Test | `phone_F2`, `phone_M2`, `studio_F2`, `studio_M2` | Final comparison only |

The split is performed by complete recording, not by randomly mixing frames. This avoids putting highly similar neighboring frames from the same utterance in both training and validation sets. After the validation check, the selected model is fitted again with all four training recordings. The test labels are used only for the final metrics and red reference lines.

## 3. Common DSP pipeline

For frame $m$ with samples $x_m[n]$, normalized short-time energy is

$$
E_m = \frac{1}{N}\sum_{n=0}^{N-1}x_m^2[n],
\qquad
E_m^{(norm)} = \frac{E_m}{\max_k E_k}.
$$

The baseline methods use normalized STE because voiced speech normally has much more energy than silence. However, weak unvoiced consonants can have low energy and may be removed by an energy-only threshold.

The DSP extension therefore also uses:

- a first-order pre-emphasis FIR filter $h[n]=[1,-0.97]$;
- zero-crossing rate (ZCR), which is often higher for unvoiced sounds;
- high-frequency energy ratio above 3 kHz;
- log-energy, which compresses the large dynamic range.

The filter implements $y[n]=x[n]-0.97x[n-1]$. It reduces the dominance of slowly varying low-frequency voiced components and makes weak high-frequency unvoiced components easier to observe. The F0 curve is estimated by frame autocorrelation and is displayed only for explanation; it is not used to obtain the test boundaries.

## 4. TT1 – global threshold search

### 4.1 Teacher-aligned baseline

TT1 searches for one global normalized-STE threshold. Binary search minimizes the training boundary error. At inference time, a frame is speech when its normalized STE is above the learned threshold, followed by the 200 ms silence rule.

### 4.2 DSP extension

The extension combines log-STE, pre-emphasized energy, ZCR, and high-frequency energy ratio in a small logistic classifier. Its loss history is stored in the notebook. This is still a transparent frame-level input–output mapping; no external machine-learning package is used.

![TT1 baseline - phone F2](../src/TT1/output/1/phone_F2.png)
![TT1 baseline - phone M2](../src/TT1/output/1/phone_M2.png)
![TT1 baseline - studio F2](../src/TT1/output/1/studio_F2.png)
![TT1 baseline - studio M2](../src/TT1/output/1/studio_M2.png)

![TT1 extension - phone F2](../src/TT1/output/2/phone_F2.png)
![TT1 extension - phone M2](../src/TT1/output/2/phone_M2.png)
![TT1 extension - studio F2](../src/TT1/output/2/studio_F2.png)
![TT1 extension - studio M2](../src/TT1/output/2/studio_M2.png)

TT1 gives the best boundary result. The extension decreases mean MAE from **8.75 ms** to **6.25 ms** and mean RMSE from **11.34 ms** to **7.80 ms**. Its frame F1 also increases slightly. UV recall changes from 0.971 to 0.968, which is practically similar on this small set.

## 5. TT2 – histogram method

### 5.1 Teacher-aligned baseline

TT2 creates the histogram $H[n]$ of normalized STE. The low-energy mode represents silence and the higher-energy mode represents speech. The threshold is computed between the two modes and is adapted for each recording.

### 5.2 DSP extension

The extension changes the one-dimensional histogram into a joint histogram

$$
H[n,m] = H\left[E_m^{(norm)},\, ZCR_m\right].
$$

Each histogram bin stores smoothed speech and silence counts learned from the training labels. The decision score is the log ratio of the two counts. A training-derived energy floor prevents very quiet background frames from being accepted only because of a noisy ZCR bin.

![TT2 baseline - phone F2](../src/TT2/output/1/phone_F2.png)
![TT2 baseline - phone M2](../src/TT2/output/1/phone_M2.png)
![TT2 baseline - studio F2](../src/TT2/output/1/studio_F2.png)
![TT2 baseline - studio M2](../src/TT2/output/1/studio_M2.png)

![TT2 extension - phone F2](../src/TT2/output/2/phone_F2.png)
![TT2 extension - phone M2](../src/TT2/output/2/phone_M2.png)
![TT2 extension - studio F2](../src/TT2/output/2/studio_F2.png)
![TT2 extension - studio M2](../src/TT2/output/2/studio_M2.png)

The two-dimensional histogram increases frame F1 from **0.977** to **0.988** and UV recall from **0.891** to **0.947**. However, mean boundary MAE increases from **17.50 ms** to **20.00 ms**. Therefore, it preserves more weak unvoiced frames but moves some outer boundaries by one to three 10 ms hops. This is a useful trade-off, not a universal improvement.

## 6. TT3 – Gaussian statistical model

### 6.1 Teacher-aligned baseline

TT3 estimates the mean and standard deviation of normalized STE separately for speech and silence frames. The threshold is the intersection of the two fitted Gaussian densities. This implements the assignment requirement to use all labeled training frames when estimating the two distributions.

### 6.2 DSP extension

The extension uses a four-dimensional Gaussian for each class. The vector contains log-STE, pre-emphasized energy, ZCR, and high-frequency ratio. Full covariance matrices model correlations between these features, with diagonal regularization for numerical stability.

![TT3 baseline - phone F2](../src/TT3/output/1/phone_F2.png)
![TT3 baseline - phone M2](../src/TT3/output/1/phone_M2.png)
![TT3 baseline - studio F2](../src/TT3/output/1/studio_F2.png)
![TT3 baseline - studio M2](../src/TT3/output/1/studio_M2.png)

![TT3 extension - phone F2](../src/TT3/output/2/phone_F2.png)
![TT3 extension - phone M2](../src/TT3/output/2/phone_M2.png)
![TT3 extension - studio F2](../src/TT3/output/2/studio_F2.png)
![TT3 extension - studio M2](../src/TT3/output/2/studio_M2.png)

The one-dimensional Gaussian obtains **11.25 ms** MAE, while the multivariate model obtains **15.00 ms**. The baseline is better here. Four training recordings are too few to estimate a reliable full covariance model, and normalized energy already separates speech and silence well. This result demonstrates why model complexity must be justified by data, not only by the number of features.

## 7. Quantitative comparison

| Method | Mean MAE (ms) ↓ | Mean RMSE (ms) ↓ | Frame F1 ↑ | UV recall ↑ |
|---|---:|---:|---:|---:|
| TT1-1: global STE threshold | 8.75 | 11.34 | 0.993 | 0.971 |
| **TT1-2: DSP feature classifier** | **6.25** | **7.80** | **0.996** | 0.968 |
| TT2-1: 1-D STE histogram | 17.50 | 18.81 | 0.977 | 0.891 |
| TT2-2: joint STE–ZCR histogram | 20.00 | 21.56 | 0.988 | **0.947** |
| TT3-1: 1-D Gaussian STE | 11.25 | 14.87 | 0.994 | **0.971** |
| TT3-2: multivariate Gaussian | 15.00 | 16.34 | 0.978 | 0.903 |

Boundary MAE is the primary metric required by the assignment. RMSE penalizes a large boundary error more strongly. Frame F1 checks the complete speech region rather than only its two outer boundaries. UV recall specifically measures whether low-energy unvoiced frames are retained. Reporting all four prevents a method from appearing good only because one metric hides its failure mode.

## 8. Main observations

1. **Energy is a strong baseline.** The recordings have clear leading and trailing silence, so a learned normalized-STE threshold is already accurate.
2. **Unvoiced speech is the difficult case.** It has weaker energy and more high-frequency/noise-like content than voiced speech. ZCR and pre-emphasis help describe this difference.
3. **Post-processing matters.** The 200 ms rule removes short false silence gaps inside an utterance and stabilizes the final outer boundaries.
4. **More features do not guarantee lower MAE.** TT2 improves UV retention but slightly worsens outer boundaries. TT3 overfits covariance structure on the small training set.
5. **No test tuning was used.** Parameters were selected with training data and one held-out training recording. The four test files were used only once for the reported evaluation.

## 9. Limitations and possible next step

The dataset is small and contains only two environments, two genders, and one test utterance per condition. Per-recording energy normalization reduces level variation but can be sensitive when a recording contains an isolated impulse. A larger training set would allow speaker-independent cross-validation and a more reliable multivariate density estimate. If more data become available, a diagonal-covariance Gaussian or a small Gaussian mixture should be compared before using a larger learned model.

## 10. Conclusion

The notebooks reproduce the three requested DSP approaches and provide four figures in one run for every method. The most accurate tested method is TT1-2 with **6.25 ms mean boundary MAE**. The main engineering lesson is that voiced/unvoiced features are useful when they target a known energy-only failure, but the simplest correctly estimated model remains preferable when the dataset is small.
