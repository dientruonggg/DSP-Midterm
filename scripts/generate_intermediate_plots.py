#!/usr/bin/env python3
"""
Generate comprehensive intermediate stage plots for VAD training matching the
exact white-background academic/presentation style of the existing project plots:
1. TT1: Binary Search Convergence (Epochs vs Threshold, Signed Duration Error, Loss MAE).
2. TT2: Dual-Peak STE Histograms & 5-point MA smoothing across all 4 training files.
3. TT3: Gaussian PDFs (Silence vs Speech) & Bayes Intersection Root T = 0.00202.
4. Preprocessing: 200 ms minimum silence bridging rule (before vs after false gap removal).
"""

from pathlib import Path
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib to match existing project style: clean white background, Tableau colors
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#111827",
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "font.size": 11,
    "font.family": "sans-serif",
})

FRAME_MS = 20.0
HOP_MS = 10.0
MIN_SILENCE_MS = 200.0

def read_wav(path):
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    return signal / 32768.0, sample_rate

def read_lab(path):
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        label = parts[2].lower()
        if label in {"sil", "v", "uv"}:
            segments.append((start, end, label))
    return segments

def frame_signal(signal, sample_rate, frame_ms=20.0, hop_ms=10.0):
    frame_length = int(round(frame_ms * sample_rate / 1000.0))
    hop_length = int(round(hop_ms * sample_rate / 1000.0))
    frame_count = max(1, int(np.ceil(max(0, len(signal) - frame_length) / hop_length)) + 1)
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.zeros(padded_length, dtype=np.float64)
    padded[:len(signal)] = signal
    frames = np.zeros((frame_count, frame_length), dtype=np.float64)
    for index in range(frame_count):
        start = index * hop_length
        frames[index] = padded[start:start + frame_length]
    starts = np.arange(frame_count) * hop_length / sample_rate
    centers = starts + frame_length / (2.0 * sample_rate)
    return frames, starts, centers

def compute_features(signal, sample_rate, frame_ms=20.0, hop_ms=10.0):
    frames, starts, centers = frame_signal(signal, sample_rate, frame_ms, hop_ms)
    ste_raw = np.mean(frames ** 2, axis=1)
    ste_norm = ste_raw / max(float(ste_raw.max()), 1e-12)
    return {
        "frames": frames,
        "starts": starts,
        "centers": centers,
        "ste_raw": ste_raw,
        "ste_norm": ste_norm,
    }

def frame_labels(centers, segments):
    labels = np.full(len(centers), "sil", dtype="<U3")
    for index, time_value in enumerate(centers):
        for start, end, label in segments:
            if start <= time_value < end or (index == len(centers) - 1 and start <= time_value <= end):
                labels[index] = label
                break
    return labels

def speech_bounds(segments):
    speech = [(start, end) for start, end, label in segments if label in {"v", "uv"}]
    return min(start for start, _ in speech), max(end for _, end in speech)

def bridge_short_silences(decisions, hop_ms=10.0, minimum_ms=200.0):
    output = np.asarray(decisions, dtype=np.int8).copy()
    minimum_frames = int(round(minimum_ms / hop_ms))
    speech_indices = np.flatnonzero(output)
    if len(speech_indices) == 0:
        return output
    index = int(speech_indices[0])
    last = int(speech_indices[-1])
    while index <= last:
        if output[index] == 1:
            index += 1
            continue
        start = index
        while index <= last and output[index] == 0:
            index += 1
        if index - start < minimum_frames:
            output[start:index] = 1
    return output

def decisions_to_bounds(decisions, starts, duration, frame_ms=20.0):
    indices = np.flatnonzero(decisions)
    if len(indices) == 0:
        return 0.0, 0.0
    start = float(starts[indices[0]])
    end = min(duration, float(starts[indices[-1]] + frame_ms / 1000.0))
    return round(start, 4), round(end, 4)

def boundary_errors(predicted, truth):
    start_error = abs(predicted[0] - truth[0]) * 1000.0
    end_error = abs(predicted[1] - truth[1]) * 1000.0
    mae = (start_error + end_error) / 2.0
    rmse = np.sqrt((start_error ** 2 + end_error ** 2) / 2.0)
    return {"start_error_ms": start_error, "end_error_ms": end_error, "mae_ms": mae, "rmse_ms": rmse}

# -------------------------------------------------------------
# Load Train & Test Data
# -------------------------------------------------------------
project_root = Path("/home/bim/Projects/DSP_MidTerm")
train_dir = project_root / "TinHieuHuanLuyen"
out_dir = project_root / "output_plots"
out_dir.mkdir(exist_ok=True)

train_records = []
for wav_path in sorted(train_dir.glob("*.wav")):
    sig, sr = read_wav(wav_path)
    segs = read_lab(wav_path.with_suffix(".lab"))
    feat = compute_features(sig, sr, FRAME_MS, HOP_MS)
    lbls = frame_labels(feat["centers"], segs)
    targets = np.isin(lbls, ["v", "uv"]).astype(np.int8)
    gt = speech_bounds(segs)
    train_records.append({
        "name": wav_path.stem,
        "signal": sig,
        "sample_rate": sr,
        "segments": segs,
        "duration": len(sig) / sr,
        "features": feat,
        "labels": lbls,
        "targets": targets,
        "ground_truth": gt,
    })

print(f"[+] Loaded {len(train_records)} training records from {train_dir}")

# -------------------------------------------------------------
# 1. TT1: Binary Search Convergence Plot (White Academic Style)
# -------------------------------------------------------------
print("[+] Generating Plot 1: TT1 Binary Search Convergence (Academic White Style)...")

class EnergyBinarySearchVAD:
    def __init__(self, iterations=40, search_range=(1e-4, 0.05)):
        self.iterations = iterations
        self.search_range = search_range
        self.threshold_ = None
        self.history_ = []

    def _predict(self, record, threshold):
        raw = (record["features"]["ste_norm"] >= threshold).astype(np.int8)
        return bridge_short_silences(raw, HOP_MS, MIN_SILENCE_MS)

    def fit(self, records):
        low, high = self.search_range
        self.history_ = []
        for epoch in range(1, self.iterations + 1):
            threshold = (low + high) / 2.0
            errors = []
            maes = []
            for r in records:
                dec = self._predict(r, threshold)
                pred = decisions_to_bounds(dec, r["features"]["starts"], r["duration"], FRAME_MS)
                pred_dur = pred[1] - pred[0]
                truth_dur = r["ground_truth"][1] - r["ground_truth"][0]
                errors.append(pred_dur - truth_dur)
                maes.append(boundary_errors(pred, r["ground_truth"])["mae_ms"])
            signed_err = float(np.mean(errors))
            mae_loss = float(np.mean(maes))
            self.history_.append({
                "epoch": epoch,
                "threshold": threshold,
                "low": low,
                "high": high,
                "signed_error_s": signed_err,
                "mae_loss_ms": mae_loss
            })
            if signed_err > 0:
                low = threshold
            else:
                high = threshold
        self.threshold_ = (low + high) / 2.0
        return self

vad1 = EnergyBinarySearchVAD(iterations=40).fit(train_records)
hist1 = vad1.history_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

epochs = [h["epoch"] for h in hist1]
threshs = [h["threshold"] for h in hist1]
lows = [h["low"] for h in hist1]
highs = [h["high"] for h in hist1]
signed_errs = [h["signed_error_s"] * 1000.0 for h in hist1]  # ms
maes = [h["mae_loss_ms"] for h in hist1]

# Subplot 1: Threshold Bisecting Bounds
ax1.plot(epochs, highs, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85, label="Search Upper Bound")
ax1.plot(epochs, lows, color="#1f77b4", linestyle="--", linewidth=1.8, alpha=0.85, label="Search Lower Bound")
ax1.fill_between(epochs, lows, highs, color="#1f77b4", alpha=0.1)
ax1.plot(epochs, threshs, color="#1f77b4", linewidth=2.5, marker="o", markersize=4.5, label="Midpoint Threshold T")
ax1.axhline(vad1.threshold_, color="#f28e2b", linestyle=":", linewidth=2.5, label=f"Optimal T_opt = {vad1.threshold_:.5f}")
ax1.set_title("A. Bisection Interval Narrowing Over 40 Epochs", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Search Iteration (Epoch)", fontsize=11)
ax1.set_ylabel("Normalized STE Threshold (T)", fontsize=11)
ax1.set_xlim(1, 20)
ax1.grid(True, alpha=0.4)
ax1.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")

# Subplot 2: Signed Duration Error & Boundary MAE
ax2_twin = ax2.twinx()
p1 = ax2.plot(epochs, signed_errs, color="#2ca02c", linewidth=2.2, marker="s", markersize=4.5, label="Signed Duration Error (ms)")
p2 = ax2_twin.plot(epochs, maes, color="#9467bd", linewidth=2.2, linestyle="-.", marker="^", markersize=4.5, label="Training Boundary MAE (ms)")
ax2.axhline(0, color="#666666", linestyle="--", alpha=0.7)

ax2.set_title("B. Objective Function & Boundary MAE Convergence", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel("Search Iteration (Epoch)", fontsize=11)
ax2.set_ylabel("Duration Error vs .lab (ms)", color="#2ca02c", fontsize=11)
ax2_twin.set_ylabel("Boundary MAE (ms)", color="#9467bd", fontsize=11)
ax2.set_xlim(1, 20)
ax2.grid(True, alpha=0.4)

lines = p1 + p2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

fig.suptitle("TT1 INTERMEDIATE TRAINING: Binary Search Threshold Optimization (CS425)", 
             fontsize=14, fontweight="bold", y=0.98, color="#111827")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plot1_path = out_dir / "intermediate_tt1_binary_search.png"
plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[+] Saved: {plot1_path}")

# -------------------------------------------------------------
# 2. TT2: Dual-Peak STE Histograms on 4 Training Files (White Style)
# -------------------------------------------------------------
print("[+] Generating Plot 2: TT2 Dual-Peak Histograms (Academic White Style)...")

fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=200)
axes = axes.flatten()

for idx, r in enumerate(train_records):
    ax = axes[idx]
    ste_norm = r["features"]["ste_norm"]
    counts, edges = np.histogram(ste_norm, bins=100, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    half = 5 // 2
    smooth = np.array([counts[max(0, i-half):min(len(counts), i+half+1)].mean() for i in range(len(counts))])

    # Find peaks
    peaks = []
    if smooth[0] >= smooth[1]:
        peaks.append(0)
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i-1] and smooth[i] >= smooth[i+1]:
            peaks.append(i)
    if len(peaks) < 2:
        first = int(np.argmax(smooth))
        cands = [i for i in range(len(smooth)) if abs(i-first) > 2]
        second = max(cands, key=lambda i: smooth[i]) if cands else min(first+3, len(smooth)-1)
        peaks = sorted([first, second])
    m1, m2 = centers[peaks[0]], centers[peaks[1]]
    weight = 5.0
    threshold = (weight * m1 + m2) / (weight + 1.0)

    # Plot
    ax.bar(centers, counts, width=0.01, color="#cbd5e1", alpha=0.7, edgecolor="#94a3b8", label="Raw Histogram (100 bins)")
    ax.plot(centers, smooth, color="#1f77b4", linewidth=2.4, label="5-point MA Smoothed Envelope")
    ax.scatter([m1], [smooth[peaks[0]]], color="#d62728", s=85, zorder=5, edgecolors="#000000", linewidth=0.8, label=f"M1 (Silence Mode) = {m1:.3f}")
    ax.scatter([m2], [smooth[peaks[1]]], color="#2ca02c", s=85, zorder=5, edgecolors="#000000", linewidth=0.8, label=f"M2 (Speech Mode) = {m2:.3f}")
    ax.axvline(threshold, color="#f28e2b", linestyle="--", linewidth=2.2, 
               label=f"Threshold T = {threshold:.4f}\n[W=5 Formula]")

    # Formatting
    env_type = "Phone (Noisy)" if "phone" in r["name"] else "Studio (Clean)"
    ax.set_title(f"{r['name']}.wav ({env_type})", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Normalized STE", fontsize=10)
    ax.set_ylabel("Frame Count", fontsize=10)
    ax.set_xlim(0, 0.35)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9, edgecolor="#cccccc")

fig.suptitle("TT2 INTERMEDIATE TRAINING: Unsupervised Dual-Peak Histogram Mode Extraction (Giannakopoulos 2014)", 
             fontsize=14, fontweight="bold", y=0.99, color="#111827")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plot2_path = out_dir / "intermediate_tt2_histogram.png"
plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[+] Saved: {plot2_path}")

# -------------------------------------------------------------
# 3. TT3: Gaussian PDFs & Bayes Intersection Plot (White Style)
# -------------------------------------------------------------
print("[+] Generating Plot 3: TT3 Gaussian Distribution & Bayes Intersection (White Style)...")

all_values = np.concatenate([r["features"]["ste_norm"] for r in train_records])
all_targets = np.concatenate([r["targets"] for r in train_records])
silence_vals = all_values[all_targets == 0]
speech_vals = all_values[all_targets == 1]

mu_sil, std_sil = float(silence_vals.mean()), float(silence_vals.std())
mu_sp, std_sp = float(speech_vals.mean()), float(speech_vals.std())

grid = np.linspace(0.0, 0.6, 2000)
pdf_sil = (1.0 / (np.sqrt(2 * np.pi) * std_sil)) * np.exp(-0.5 * ((grid - mu_sil) / std_sil) ** 2)
pdf_sp = (1.0 / (np.sqrt(2 * np.pi) * std_sp)) * np.exp(-0.5 * ((grid - mu_sp) / std_sp) ** 2)

# Optimal Bayes root
diff = np.abs(pdf_sil - pdf_sp)
bayes_idx = np.argmin(diff[:len(grid)//3])
t_bayes = grid[bayes_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=200)

# Linear Scale PDF
ax1.hist(silence_vals, bins=150, range=(0, 0.6), density=True, color="#93c5fd", alpha=0.45, label="Empirical Silence Frames")
ax1.hist(speech_vals, bins=150, range=(0, 0.6), density=True, color="#fdba74", alpha=0.45, label="Empirical Speech Frames")
ax1.plot(grid, pdf_sil, color="#1f77b4", linewidth=2.5, label=f"Silence N(μ={mu_sil:.5f}, σ={std_sil:.5f})")
ax1.plot(grid, pdf_sp, color="#d62728", linewidth=2.5, label=f"Speech N(μ={mu_sp:.4f}, σ={std_sp:.4f})")
ax1.axvline(t_bayes, color="#2ca02c", linestyle="--", linewidth=2.4, 
            label=f"Bayes Threshold T = {t_bayes:.5f}\n[Equal Likelihood Root]")
ax1.fill_between(grid[grid <= t_bayes], pdf_sil[grid <= t_bayes], color="#1f77b4", alpha=0.12)
ax1.fill_between(grid[grid > t_bayes], pdf_sp[grid > t_bayes], color="#d62728", alpha=0.12)

ax1.set_title("A. Linear Scale Probability Density Functions (PDF)", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Normalized STE", fontsize=11)
ax1.set_ylabel("Probability Density", fontsize=11)
ax1.set_xlim(0, 0.05)
ax1.grid(True, alpha=0.4)
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")

# Log Scale PDF (Highlights full separation)
ax2.plot(grid, np.log10(np.maximum(pdf_sil, 1e-6)), color="#1f77b4", linewidth=2.5, label="Log10 Silence PDF")
ax2.plot(grid, np.log10(np.maximum(pdf_sp, 1e-6)), color="#d62728", linewidth=2.5, label="Log10 Speech PDF")
ax2.axvline(t_bayes, color="#2ca02c", linestyle="--", linewidth=2.4, label=f"Bayes Boundary (T = {t_bayes:.5f})")
ax2.annotate(f"Optimal Decision\nT = {t_bayes:.5f}", 
             xy=(t_bayes, np.log10(pdf_sp[bayes_idx])), 
             xytext=(t_bayes + 0.08, np.log10(pdf_sp[bayes_idx]) + 1.0),
             arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=7),
             fontsize=10, fontweight="bold", color="#15803d")

ax2.set_title("B. Log-Likelihood Scale (Full Dynamic Range 0 to 0.5)", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel("Normalized STE", fontsize=11)
ax2.set_ylabel("Log10 Likelihood", fontsize=11)
ax2.set_xlim(0, 0.5)
ax2.grid(True, alpha=0.4)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")

fig.suptitle("TT3 INTERMEDIATE TRAINING: Gaussian Bayes Analytical Modeling from .lab Frames", 
             fontsize=14, fontweight="bold", y=0.98, color="#111827")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plot3_path = out_dir / "intermediate_tt3_gaussian_bayes.png"
plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[+] Saved: {plot3_path}")

# -------------------------------------------------------------
# 4. Preprocessing: 200 ms Minimum Silence Bridging Demonstration (White Style)
# -------------------------------------------------------------
print("[+] Generating Plot 4: 200 ms Minimum Silence Bridging Rule Demonstration (White Style)...")

rec = [r for r in train_records if r["name"] == "phone_F1"][0]
times = rec["features"]["centers"]
ste = rec["features"]["ste_norm"]
thresh = 0.0233
raw_dec = (ste >= thresh).astype(np.int8)
bridged_dec = bridge_short_silences(raw_dec, HOP_MS, MIN_SILENCE_MS)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), dpi=200, sharex=True)

# Audio waveform
t_sig = np.arange(len(rec["signal"])) / rec["sample_rate"]
ax1.plot(t_sig, rec["signal"], color="0.65", linewidth=0.8, alpha=0.9)
ax1.set_title(f"A. Audio Waveform ({rec['name']}.wav) with Ground Truth Labels", fontsize=11, fontweight="bold")
ax1.set_ylabel("Amplitude", fontsize=10)
ax1.grid(True, alpha=0.4)
gt_start, gt_end = rec["ground_truth"]
ax1.axvspan(gt_start, gt_end, color="#2ca02c", alpha=0.18, label="Ground Truth Speech Region")
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")

# STE & Threshold
ax2.plot(times, ste, color="#f28e2b", linewidth=1.6, label="Normalized STE")
ax2.axhline(thresh, color="#d97706", linestyle="--", linewidth=1.8, label=f"Detection Threshold T = {thresh:.4f}")
ax2.set_title("B. Short-Time Energy & Threshold Crossing", fontsize=11, fontweight="bold")
ax2.set_ylabel("Norm STE", fontsize=10)
ax2.grid(True, alpha=0.4)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")

# Decision Comparison: Raw vs Bridged
ax3.step(times, raw_dec * 0.9, where="mid", color="#d62728", linewidth=1.6, label="Raw Energy Decisions (Contains False Short Silences < 200ms)")
ax3.step(times, bridged_dec * 1.0, where="mid", color="#2ca02c", linewidth=2.4, label="Post-Processed Decisions (200 ms Minimum Silence Bridging Rule)")
ax3.set_title("C. Effect of Minimum Silence Bridging Rule (Eliminating False Gaps)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Time (seconds)", fontsize=10)
ax3.set_ylabel("VAD State", fontsize=10)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(["Silence", "Speech"])
ax3.set_ylim(-0.1, 1.2)
ax3.grid(True, alpha=0.4)
ax3.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")

fig.suptitle("PREPROCESSING & POST-PROCESSING RULE: 200 ms Silence Bridging (Rubric Constraint)", 
             fontsize=14, fontweight="bold", y=0.98, color="#111827")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plot4_path = out_dir / "intermediate_preprocessing_200ms.png"
plt.savefig(plot4_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[+] Saved: {plot4_path}")

print("\n" + "=" * 60)
print("  🎉 ALL 4 INTERMEDIATE PLOTS REGENERATED IN ACADEMIC WHITE STYLE!")
print("=" * 60)
