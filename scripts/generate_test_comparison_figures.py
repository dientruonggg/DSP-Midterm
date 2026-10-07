#!/usr/bin/env python3
"""
Generate 4 comparative VAD figures (one for each test file: phone_F2, phone_M2, studio_F2, studio_M2)
displaying the 3 algorithm results (TT1, TT2, TT3) stacked vertically on the exact same time axis.
Clean white academic style, high DPI, 100% English.
"""

from pathlib import Path
import wave
import csv
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "TinHieuKiemThu"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUT_PLOTS_DIR = PROJECT_ROOT / "output_plots"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FRAME_MS = 25.0
HOP_MS = 10.0
MIN_SILENCE_MS = 200.0

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
    "grid.alpha": 0.6,
    "font.size": 9,
    "font.family": "sans-serif",
})

def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        fs = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    max_amp = max(np.max(np.abs(signal)), 1.0)
    return signal / max_amp, fs

def read_lab(path):
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] not in ("F0mean", "F0std"):
            try:
                s, e = float(parts[0]), float(parts[1])
                segments.append((s, e, parts[2].lower()))
            except ValueError:
                continue
    speech_segments = [s for s in segments if s[2] in {"v", "uv"}]
    if not speech_segments:
        return 0.0, 0.0
    return round(min(s[0] for s in speech_segments), 4), round(max(s[1] for s in speech_segments), 4)

def compute_features(signal, fs):
    frame_size = int(round(FRAME_MS * fs / 1000.0))
    hop_size = int(round(HOP_MS * fs / 1000.0))
    n_frames = max(1, int(np.ceil(max(0, len(signal) - frame_size) / hop_size)) + 1)
    bin_index = np.arange(1, frame_size // 2 + 2)
    n_bins = len(bin_index)

    ste = np.zeros(n_frames)
    sc = np.zeros(n_frames)
    frame_starts = np.arange(n_frames) * (hop_size / fs)

    for i in range(n_frames):
        chunk = signal[i * hop_size : i * hop_size + frame_size]
        frame = np.zeros(frame_size)
        frame[:len(chunk)] = chunk
        ste[i] = np.mean(frame ** 2)
        mag = np.abs(np.fft.rfft(frame))
        tot = mag.sum()
        if tot > 1e-12:
            sc[i] = np.sum(bin_index * mag) / (tot * n_bins)

    ste_norm = ste / max(np.max(ste), 1e-12)
    sc_norm = sc / max(np.max(sc), 1e-12)
    return ste_norm, sc_norm, frame_starts

def bridge_silence(labels):
    min_silence_frames = int(round(MIN_SILENCE_MS / HOP_MS))
    active = labels.copy()
    speech_idx = np.flatnonzero(active == 1)
    if len(speech_idx) == 0:
        return active
    idx, last = int(speech_idx[0]), int(speech_idx[-1])
    while idx <= last:
        if active[idx] == 1:
            idx += 1
        else:
            sil_start = idx
            while idx <= last and active[idx] == 0:
                idx += 1
            if idx - sil_start < min_silence_frames:
                active[sil_start:idx] = 1
    return active

# Load TT2 predictions and thresholds from CSV
tt2_csv = PROJECT_ROOT / "src/TT2/output/mine/intermediate_results.csv"
tt2_records = {}
with open(tt2_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] == "test":
            tt2_records[row["file"]] = {
                "pred_start": float(row["pred_start"]),
                "pred_end": float(row["pred_end"]),
                "d_start_ms": float(row["d_start_ms"]),
                "d_end_ms": float(row["d_end_ms"]),
                "mae_ms": float(row["mae_ms"]),
                "T_E": float(row["T_E"]),
                "T_C": float(row["T_C"]),
            }

test_files = ["phone_F2.wav", "phone_M2.wav", "studio_F2.wav", "studio_M2.wav"]
T_TT1 = 0.003478
T_TT3 = 0.002878

for filename in test_files:
    wav_path = TEST_DIR / filename
    signal, fs = read_wav(wav_path)
    gt_start, gt_end = read_lab(wav_path.with_suffix(".lab"))
    duration = len(signal) / fs
    time_wave = np.linspace(0, duration, len(signal))

    ste_norm, sc_norm, frame_starts = compute_features(signal, fs)

    # 1. TT1 inference
    labels_tt1 = bridge_silence((ste_norm >= T_TT1).astype(int))
    sp_tt1 = np.flatnonzero(labels_tt1 == 1)
    pred_tt1_start = float(frame_starts[sp_tt1[0]])
    pred_tt1_end = min(duration, float(frame_starts[sp_tt1[-1]] + FRAME_MS / 1000.0))
    d_st_tt1 = (pred_tt1_start - gt_start) * 1000.0
    d_en_tt1 = (pred_tt1_end - gt_end) * 1000.0
    mae_tt1 = (abs(d_st_tt1) + abs(d_en_tt1)) / 2.0

    # 2. TT2 inference
    tt2_data = tt2_records[filename]
    pred_tt2_start = tt2_data["pred_start"]
    pred_tt2_end = tt2_data["pred_end"]
    d_st_tt2 = tt2_data["d_start_ms"]
    d_en_tt2 = tt2_data["d_end_ms"]
    mae_tt2 = tt2_data["mae_ms"]
    T_E_tt2 = tt2_data["T_E"]
    T_C_tt2 = tt2_data["T_C"]

    # 3. TT3 inference
    labels_tt3 = bridge_silence((ste_norm >= T_TT3).astype(int))
    sp_tt3 = np.flatnonzero(labels_tt3 == 1)
    pred_tt3_start = float(frame_starts[sp_tt3[0]])
    pred_tt3_end = min(duration, float(frame_starts[sp_tt3[-1]] + FRAME_MS / 1000.0))
    d_st_tt3 = (pred_tt3_start - gt_start) * 1000.0
    d_en_tt3 = (pred_tt3_end - gt_end) * 1000.0
    mae_tt3 = (abs(d_st_tt3) + abs(d_en_tt3)) / 2.0

    # Build 3-row stacked comparison figure (widescreen 11.2 x 5.0, ultra-sharp 300 DPI)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11.2, 5.0), sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0.20, left=0.05, right=0.99, top=0.92, bottom=0.08)

    fig.suptitle(
        f"Comparative Test VAD: {filename}  (Ground Truth: [{gt_start:.2f}s, {gt_end:.2f}s])",
        fontsize=13.0, fontweight="bold", color="#0F172A", y=0.98
    )

    # ------------------ PANEL 1: TT1 ------------------
    ax1.plot(time_wave, signal, color="#94A3B8", linewidth=0.7, alpha=0.65, label="Waveform")
    ax1.plot(frame_starts, ste_norm, color="#0284C7", linewidth=1.6, label=f"STE Norm (T={T_TT1:.5f})")
    ax1.axhline(T_TT1, color="#0284C7", linestyle=":", linewidth=1.2, alpha=0.9)
    
    # GT lines
    ax1.axvline(gt_start, color="#DC2626", linewidth=2.0, label=f"GT [{gt_start:.2f}, {gt_end:.2f}]")
    ax1.axvline(gt_end, color="#DC2626", linewidth=2.0)
    
    # TT1 lines
    ax1.axvline(pred_tt1_start, color="#2563EB", linestyle="--", linewidth=2.0, label=f"TT1 [{pred_tt1_start:.2f}, {pred_tt1_end:.2f}]")
    ax1.axvline(pred_tt1_end, color="#2563EB", linestyle="--", linewidth=2.0)
    ax1.axvspan(pred_tt1_start, pred_tt1_end, color="#38BDF8", alpha=0.15, label="TT1 Speech")

    ax1.set_title(
        f"TT1 (Energy Bisection): Pred [{pred_tt1_start:.3f}s, {pred_tt1_end:.3f}s] | ΔStart={d_st_tt1:+.1f}ms, ΔEnd={d_en_tt1:+.1f}ms | MAE = {mae_tt1:.2f} ms",
        fontsize=10.5, fontweight="bold", color="#0369A1", loc="left", pad=3
    )
    ax1.set_ylabel("Amp / STE", fontsize=9.0)
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.92, ncol=4)

    # ------------------ PANEL 2: TT2 ------------------
    ax2.plot(time_wave, signal, color="#94A3B8", linewidth=0.7, alpha=0.65, label="Waveform")
    ax2.plot(frame_starts, ste_norm, color="#D97706", linewidth=1.5, label=f"STE (TE={T_E_tt2:.4f})")
    ax2.plot(frame_starts, sc_norm, color="#7C3AED", linewidth=1.3, alpha=0.85, label=f"SC (TC={T_C_tt2:.4f})")
    ax2.axhline(T_E_tt2, color="#D97706", linestyle=":", linewidth=1.1)
    ax2.axhline(T_C_tt2, color="#7C3AED", linestyle=":", linewidth=1.1)

    # GT lines
    ax2.axvline(gt_start, color="#DC2626", linewidth=2.0, label=f"GT [{gt_start:.2f}, {gt_end:.2f}]")
    ax2.axvline(gt_end, color="#DC2626", linewidth=2.0)

    # TT2 lines
    ax2.axvline(pred_tt2_start, color="#D97706", linestyle="--", linewidth=2.0, label=f"TT2 [{pred_tt2_start:.2f}, {pred_tt2_end:.2f}]")
    ax2.axvline(pred_tt2_end, color="#D97706", linestyle="--", linewidth=2.0)
    ax2.axvspan(pred_tt2_start, pred_tt2_end, color="#F59E0B", alpha=0.15, label="TT2 Speech (+10ms Pad)")

    ax2.set_title(
        f"TT2 (Dual-Feature Histogram): Pred [{pred_tt2_start:.3f}s, {pred_tt2_end:.3f}s] | ΔStart={d_st_tt2:+.1f}ms, ΔEnd={d_en_tt2:+.1f}ms | MAE = {mae_tt2:.2f} ms",
        fontsize=10.5, fontweight="bold", color="#B45309", loc="left", pad=3
    )
    ax2.set_ylabel("Amp / Feats", fontsize=9.0)
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.92, ncol=4)

    # ------------------ PANEL 3: TT3 ------------------
    ax3.plot(time_wave, signal, color="#94A3B8", linewidth=0.7, alpha=0.65, label="Waveform")
    ax3.plot(frame_starts, ste_norm, color="#059669", linewidth=1.6, label=f"STE Norm (T_Bayes={T_TT3:.5f})")
    ax3.axhline(T_TT3, color="#059669", linestyle=":", linewidth=1.2, alpha=0.9)

    # GT lines
    ax3.axvline(gt_start, color="#DC2626", linewidth=2.0, label=f"GT [{gt_start:.2f}, {gt_end:.2f}]")
    ax3.axvline(gt_end, color="#DC2626", linewidth=2.0)

    # TT3 lines
    ax3.axvline(pred_tt3_start, color="#047857", linestyle="--", linewidth=2.0, label=f"TT3 [{pred_tt3_start:.2f}, {pred_tt3_end:.2f}]")
    ax3.axvline(pred_tt3_end, color="#047857", linestyle="--", linewidth=2.0)
    ax3.axvspan(pred_tt3_start, pred_tt3_end, color="#10B981", alpha=0.15, label="TT3 Speech")

    ax3.set_title(
        f"TT3 (Gaussian Bayes): Pred [{pred_tt3_start:.3f}s, {pred_tt3_end:.3f}s] | ΔStart={d_st_tt3:+.1f}ms, ΔEnd={d_en_tt3:+.1f}ms | MAE = {mae_tt3:.2f} ms",
        fontsize=10.5, fontweight="bold", color="#047857", loc="left", pad=3
    )
    ax3.set_ylabel("Amp / STE", fontsize=9.0)
    ax3.set_xlabel("Time (seconds)", fontsize=10.0)
    ax3.set_ylim(-1.05, 1.05)
    ax3.set_xlim(0, duration)
    ax3.grid(True)
    ax3.legend(loc="upper right", fontsize=8.0, framealpha=0.92, ncol=4)

    stem = Path(filename).stem
    out_fig = FIGURES_DIR / f"test_comparison_{stem}.png"
    out_plot = OUTPUT_PLOTS_DIR / f"test_comparison_{stem}.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(out_plot, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"Generated: {out_fig.name}")

print("All 4 test comparison figures generated successfully!")
