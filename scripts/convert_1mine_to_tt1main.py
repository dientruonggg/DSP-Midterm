"""
Script to convert src/TT1/1-mine.ipynb to TT1-main.ipynb,
translating all Vietnamese markdown notes, comments, docstrings, and labels to English,
while strictly preserving notebook architecture and executing it to generate complete outputs.
"""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_tt1_main_notebook():
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12"
        }
    }
    cells = []

    # Cell 0: Title
    cells.append(nbformat.v4.new_markdown_cell(
"""# ALGORITHM 1 (TT1): VOICE ACTIVITY DETECTION (VAD)
## SHORT-TIME ENERGY (STE) OPTIMIZATION VIA BINARY SEARCH"""
    ))

    # Cell 1: Environment & Params
    cells.append(nbformat.v4.new_markdown_cell(
"""### 1. Environment Configuration and System Parameters
Import essential numerical and signal processing libraries (`wave`, `numpy`, `matplotlib`), load `IPython.display.Audio`, and configure framing parameters."""
    ))

    # Cell 2: Config code
    cells.append(nbformat.v4.new_code_cell(
"""from pathlib import Path
import wave
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "TinHieuHuanLuyen").exists() and (candidate / "TinHieuKiemThu").exists():
            return candidate
    return Path.cwd()

PROJECT_ROOT = find_project_root()
TRAIN_DIR = PROJECT_ROOT / "TinHieuHuanLuyen"
TEST_DIR = PROJECT_ROOT / "TinHieuKiemThu"
OUTPUT_DIR = PROJECT_ROOT / "src" / "TT1" / "output" / "main"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standardized DSP parameters
FRAME_MS = 25        # Frame length (ms)
HOP_MS = 10          # Frame shift / hop size (ms)
MIN_SILENCE_MS = 200 # Minimum silence duration (ms) to bridge pseudo-silence gaps"""
    ))

    # Cell 3: Preprocessing markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 2. Preprocessing Utilities: WAV Audio Reading, Ground-Truth (.lab) Parsing, and Short-Time Energy (STE) Computation
Core DSP functions used for preprocessing input data:
- **`read_wav`**: Reads 16-bit PCM WAV audio, normalizes amplitude to $[-1, 1]$, extracts sampling rate $f_s$, and maximum peak amplitude.
- **`read_lab`**: Parses Praat `.lab` annotation files, extracts ground-truth speech boundaries (start and end times) encompassing both voiced (`v`) and unvoiced (`uv`) segments.
- **`compute_ste`**: Frames the signal (frame length: 25 ms, hop size: 10 ms), computes Short-Time Energy (STE), and normalizes it to $[0, 1]$."""
    ))

    # Cell 4: Preprocessing code
    cells.append(nbformat.v4.new_code_cell(
"""# 1. WAV audio reading and preprocessing function (PCM 16-bit)
def read_wav(wav_path):
    \"\"\"Read 16-bit PCM WAV file, normalize amplitude to [-1, 1], return fs and max amplitude.\"\"\"
    with wave.open(str(wav_path), "rb") as wf:
        fs = wf.getframerate()                    # Sampling rate (Hz)
        samples = wf.readframes(wf.getnframes())  # Read all raw audio bytes
    # Convert raw bytes to float numpy array (16-bit PCM)
    signal = np.frombuffer(samples, dtype=np.int16).astype(float)
    max_amp = np.max(np.abs(signal))              # Peak amplitude of the signal
    # Normalize signal amplitude to [-1, 1]
    return signal / max(max_amp, 1.0), fs, max_amp


# 2. Praat .lab ground-truth annotation reading function
def read_lab(lab_path):
    \"\"\"Read .lab label file, return ground-truth speech boundaries (start, end) for voiced (v) and unvoiced (uv).\"\"\"
    speech = []
    # Read each line from Praat format label file
    for line in lab_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        # Filter speech segments labeled with voiced 'v' or unvoiced 'uv'
        if len(parts) == 3 and parts[2].lower() in {"v", "uv"}:
            speech.append((float(parts[0]), float(parts[1])))
    # Determine earliest start and latest end timestamps for speech region
    return round(min(s[0] for s in speech), 4), round(max(s[1] for s in speech), 4)


# 3. Framing and normalized Short-Time Energy (STE) computation
def compute_ste(signal, fs, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    \"\"\"Frame signal and compute normalized Short-Time Energy (STE) in [0, 1].\"\"\"
    # Convert frame and hop durations from milliseconds to sample counts
    frame_size = int(round(frame_ms * fs / 1000.0))
    hop_size = int(round(hop_ms * fs / 1000.0))
    n_frames = max(1, int(np.ceil(max(0, len(signal) - frame_size) / hop_size)) + 1)

    ste = np.zeros(n_frames)
    frame_starts = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        frame = np.zeros(frame_size)
        avail = signal[start : min(start + frame_size, len(signal))]
        frame[:len(avail)] = avail
        # Short-Time Energy: mean squared amplitude of samples within the frame
        ste[i] = np.mean(frame ** 2)
        frame_starts[i] = start / fs              # Frame start time (seconds)

    max_ste = float(np.max(ste))
    # Normalize STE energy to [0, 1] using recording peak value
    ste_norm = ste / max(max_ste, 1e-12)
    return ste_norm, frame_starts, max_ste"""
    ))

    # Cell 5: Model markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 3. Binary Search Energy VAD Model Class (EnergyBinarySearchVAD)
Defines frame classification using threshold $T$, bridges pseudo-silences $< 200\\text{ ms}$, and optimizes threshold via binary search on duration discrepancy."""
    ))

    # Cell 6: Model code
    cells.append(nbformat.v4.new_code_cell(
"""class EnergyBinarySearchVAD:
    \"\"\"Binary search VAD model optimizing STE threshold on signed duration error.\"\"\"
    def __init__(self, epochs=40, search_range=(1e-4, 0.05)):
        self.epochs = epochs
        self.search_range = search_range
        self.threshold = None
        # Lists tracking threshold T and loss metrics across epochs
        self.t_list = []
        self.loss_mae = []
        self.loss_rmse = []
        self.mae_list = self.loss_mae    # alias
        self.rmse_list = self.loss_rmse  # alias

    def predict(self, ste_norm, frame_starts, duration, threshold=None):
        \"\"\"Classify frames using threshold T and bridge pseudo-silence gaps < 200 ms.\"\"\"
        if threshold is None:
            threshold = self.threshold
        labels = (ste_norm >= threshold).astype(int)
        min_silence_frames = int(round(MIN_SILENCE_MS / HOP_MS))
        speech_idx = np.flatnonzero(labels == 1)
        if len(speech_idx) == 0:
            return (0.0, 0.0), labels

        idx, last = int(speech_idx[0]), int(speech_idx[-1])
        while idx <= last:
            if labels[idx] == 1:
                idx += 1
            else:
                sil_start = idx
                while idx <= last and labels[idx] == 0:
                    idx += 1
                if idx - sil_start < min_silence_frames:
                    labels[sil_start:idx] = 1

        speech_idx = np.flatnonzero(labels == 1)
        start_time = float(frame_starts[speech_idx[0]])
        end_time = min(duration, float(frame_starts[speech_idx[-1]] + FRAME_MS / 1000.0))
        return (round(start_time, 4), round(end_time, 4)), labels

    def fit(self, train_data):
        \"\"\"Train threshold bisection across epochs on search interval [low, high].\"\"\"
        low, high = self.search_range
        self.t_list = []
        self.loss_mae = []
        self.loss_rmse = []
        self.mae_list = self.loss_mae    # alias
        self.rmse_list = self.loss_rmse  # alias
        print(f"Starting binary search for STE threshold on [{low:.6f}, {high:.6f}] ({self.epochs} epochs):")
        print("-" * 96)
        for epoch in range(1, self.epochs + 1):
            threshold = (low + high) / 2.0
            duration_errors, maes, rmses = [], [], []
            for item in train_data:
                pred_bounds, _ = self.predict(item["ste_norm"], item["frame_starts"], item["duration"], threshold)
                pred_dur = pred_bounds[1] - pred_bounds[0]
                true_dur = item["gt_bounds"][1] - item["gt_bounds"][0]
                duration_errors.append(pred_dur - true_dur)
                d_st = (pred_bounds[0] - item["gt_bounds"][0]) * 1000.0
                d_en = (pred_bounds[1] - item["gt_bounds"][1]) * 1000.0
                maes.append((abs(d_st) + abs(d_en)) / 2.0)
                rmses.append(np.sqrt((d_st**2 + d_en**2) / 2.0))

            mean_error = float(np.mean(duration_errors))
            epoch_mae = float(np.mean(maes))
            epoch_rmse = float(np.mean(rmses))

            # Store threshold T, MAE loss, and RMSE loss per epoch
            self.t_list.append(threshold)
            self.loss_mae.append(epoch_mae)
            self.loss_rmse.append(epoch_rmse)

            if epoch <= 10 or epoch % 5 == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:02d} | T = {threshold:.8f} | Bias = {mean_error:+.4f} s | Train MAE = {epoch_mae:5.2f} ms | Train RMSE = {epoch_rmse:5.2f} ms")

            if mean_error > 0:
                low = threshold
            else:
                high = threshold

        self.threshold = (low + high) / 2.0
        print("-" * 96)
        print(f">>> OPTIMAL STE THRESHOLD FOUND: T = {self.threshold:.8f}")
        return self"""
    ))

    # Cell 7: Load Train Data markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 4. Training Dataset Ingestion
Load 4 training recordings (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`), compute STE, and parse ground-truth labels."""
    ))

    # Cell 8: Load Train Data code
    cells.append(nbformat.v4.new_code_cell(
"""train_data = []
for wav_file in sorted(TRAIN_DIR.glob("*.wav")):
    signal, fs, max_amp = read_wav(wav_file)
    ste_norm, frame_starts, max_ste = compute_ste(signal, fs)
    gt_bounds = read_lab(wav_file.with_suffix(".lab"))
    train_data.append({
        "name": wav_file.name,
        "ste_norm": ste_norm,
        "frame_starts": frame_starts,
        "duration": len(signal) / fs,
        "gt_bounds": gt_bounds,
        "max_amp": max_amp,
        "max_ste": max_ste,
    })
print(f"Loaded {len(train_data)} training files successfully.")

# print(train_data)"""
    ))

    # Cell 9: Train Model markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 5. Binary Search Model Training for Optimal Threshold T_opt
Execute bisection search training to iteratively narrow the threshold interval across 40 epochs."""
    ))

    # Cell 10: Train Model code
    cells.append(nbformat.v4.new_code_cell(
"""model = EnergyBinarySearchVAD(epochs=40)
model.fit(train_data)"""
    ))

    # Cell 11: Unnormalize threshold markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 6. Physical Scale Threshold Mapping
Map the normalized optimal threshold $T_{opt}$ back to raw STE energy levels and equivalent sample amplitudes for each recording."""
    ))

    # Cell 12: Unnormalize threshold code
    cells.append(nbformat.v4.new_code_cell(
"""print(">>> THRESHOLD MAPPING TO PHYSICAL UNITS:")
for item in train_data:
    t_raw_ste = model.threshold * item["max_ste"]
    t_amp = np.sqrt(model.threshold) * item["max_amp"]
    print(f"  {item['name']:<15} (max_amp = {item['max_amp']:>7.0f}, max_ste = {item['max_ste']:.4f}) -> Raw STE = {t_raw_ste:.6f} | Equivalent Amp = {t_amp:.1f}")"""
    ))

    # Cell 13: Test Benchmark markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 7. Quantitative Test Benchmark Evaluation
Evaluate the VAD algorithm with the optimized threshold on the 4 test recordings and report MAE and RMSE metrics."""
    ))

    # Cell 14: Test Benchmark code
    cells.append(nbformat.v4.new_code_cell(
"""test_results_dict = {}
test_results = []
print("=" * 96)
print(f"{'QUANTITATIVE TEST SET BENCHMARK RESULTS':^96}")
print("=" * 96)
print(f"{'File':<16} | {'Ground Truth (s)':<16} | {'Prediction (s)':<16} | {'ΔStart':<10} | {'ΔEnd':<10} | {'MAE (ms)':<9} | {'RMSE (ms)':<9}")
print("-" * 96)

for wav_file in sorted(TEST_DIR.glob("*.wav")):
    signal, fs, max_amp = read_wav(wav_file)
    duration = len(signal) / fs
    ste_norm, frame_starts, max_ste = compute_ste(signal, fs)
    gt = read_lab(wav_file.with_suffix(".lab"))
    pred, labels = model.predict(ste_norm, frame_starts, duration)

    d_start = (pred[0] - gt[0]) * 1000.0
    d_end = (pred[1] - gt[1]) * 1000.0
    mae = (abs(d_start) + abs(d_end)) / 2.0
    rmse = np.sqrt((d_start**2 + d_end**2) / 2.0)

    res_item = {
        "file": wav_file.name,
        "signal": signal,
        "fs": fs,
        "ste_norm": ste_norm,
        "frame_starts": frame_starts,
        "gt": gt,
        "pred": pred,
        "labels": labels,
        "d_start": d_start,
        "d_end": d_end,
        "mae": mae,
        "rmse": rmse,
    }
    test_results.append(res_item)
    test_results_dict[wav_file.name] = res_item

    gt_str = f"[{gt[0]:.2f}, {gt[1]:.2f}]"
    pred_str = f"[{pred[0]:.2f}, {pred[1]:.2f}]"
    print(f"{wav_file.name:<16} | {gt_str:<16} | {pred_str:<16} | {d_start:+8.1f} ms | {d_end:+8.1f} ms | {mae:9.2f} | {rmse:9.2f}")

mean_mae = np.mean([r["mae"] for r in test_results])
mean_rmse = np.mean([r["rmse"] for r in test_results])
print("-" * 96)
print(f"BENCHMARK SUMMARY:  Mean MAE = {mean_mae:.2f} ms  |  Mean RMSE = {mean_rmse:.2f} ms")
print("=" * 96)"""
    ))

    # Cell 15: Tracking history markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 8. Threshold T Tracking and Loss Curves (MAE / RMSE)
Display tracked histories and visualize the evolution of threshold $T$, MAE loss, and RMSE loss across training epochs."""
    ))

    # Cell 16: Tracking history code
    cells.append(nbformat.v4.new_code_cell(
"""# Display tracked training metrics across epochs
print(f"Total epochs tracked: {len(model.t_list)}")
print(f"1. Threshold T history (first 5 epochs):      {[round(x, 6) for x in model.t_list[:5]]}")
print(f"2. MAE Loss history (first 5 epochs, ms):     {[round(x, 2) for x in model.loss_mae[:5]]}")
print(f"3. RMSE Loss history (first 5 epochs, ms):    {[round(x, 2) for x in model.loss_rmse[:5]]}")

# Visualize the 3 metrics across training epochs
epochs = range(1, len(model.t_list) + 1)
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(epochs, model.t_list, 'b.-', label="Threshold T")
plt.title("Threshold T Evolution across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Threshold T")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, model.loss_mae, 'g.-', label="MAE Loss")
plt.title("MAE Loss Evolution across Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE (ms)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, model.loss_rmse, 'r.-', label="RMSE Loss")
plt.title("RMSE Loss Evolution across Epochs")
plt.xlabel("Epoch")
plt.ylabel("RMSE (ms)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()"""
    ))

    # Cell 17: Plot Helper markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 9. Visualization Helper & Audio Player Integration
Defines detailed plotting function (Waveform, STE, decision threshold, boundaries) alongside dual audio player widgets (full audio and detected speech segment)."""
    ))

    # Cell 18: Plot Helper code
    cells.append(nbformat.v4.new_code_cell(
"""def plot_and_play_test_file(res):
    \"\"\"Plot detailed test results and display audio playback widgets.\"\"\"
    fig, ax = plt.subplots(figsize=(13, 4.5))
    time_signal = np.arange(len(res["signal"])) / res["fs"]
    frame_centers = res["frame_starts"] + (FRAME_MS / 2000.0)

    ax.plot(time_signal, res["signal"], color="gray", linewidth=0.7, label="Normalized Waveform")
    ax.plot(frame_centers, res["ste_norm"], color="red", linewidth=1.1, label="Normalized STE")
    ax.axhline(model.threshold, color="blue", linestyle="--", linewidth=1.2, label=f"Threshold T = {model.threshold:.6f}")

    ax.fill_between(frame_centers, 0, 1, where=res["labels"] == 1, color="lightgreen", alpha=0.25, label="Predicted Speech Region")

    ax.axvline(res["gt"][0], color="red", linestyle="--", linewidth=1.8, label="Ground-truth Boundary (LAB)")
    ax.axvline(res["gt"][1], color="red", linestyle="--", linewidth=1.8)
    ax.axvline(res["pred"][0], color="green", linestyle=":", linewidth=2.2, label="Algorithm Predicted Boundary")
    ax.axvline(res["pred"][1], color="green", linestyle=":", linewidth=2.2)

    ax.set_title(f"{res['file']}  |  MAE = {res['mae']:.2f} ms  |  RMSE = {res['rmse']:.2f} ms", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude / STE")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    info_text = (
        f"Ground-truth: [{res['gt'][0]:.2f}s, {res['gt'][1]:.2f}s]\\n"
        f"Predicted:    [{res['pred'][0]:.2f}s, {res['pred'][1]:.2f}s]\\n"
        f"ΔStart: {res['d_start']:+.1f} ms | ΔEnd: {res['d_end']:+.1f} ms\\n"
        f"MAE: {res['mae']:.2f} ms | RMSE: {res['rmse']:.2f} ms"
    )
    ax.text(0.015, 0.95, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"))

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{Path(res['file']).stem}.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Exported and saved figure: {fig_path}")
    plt.show()

    # Load and display audio player widget for local audio file
    wav_file_path = TEST_DIR / res['file']
    print(f"🎵 Original Audio ({res['file']}):")
    if wav_file_path.exists():
        display(Audio(filename=str(wav_file_path)))
    else:
        display(Audio(data=res['signal'], rate=res['fs']))

    # Display audio player widget for detected speech segment
    st_sample = int(res['pred'][0] * res['fs'])
    en_sample = min(len(res['signal']), int(res['pred'][1] * res['fs']))
    if en_sample > st_sample:
        print(f"🔊 Predicted Speech Segment [{res['pred'][0]:.2f}s - {res['pred'][1]:.2f}s]:")
        display(Audio(data=res['signal'][st_sample:en_sample], rate=res['fs']))"""
    ))

    # Cell 19: Test 1 markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 10. Test Recording 1: `phone_F2.wav`
Telephone channel, female speaker. Display segmentation plot and audio players."""
    ))

    # Cell 20: Test 1 code
    cells.append(nbformat.v4.new_code_cell(
"""plot_and_play_test_file(test_results_dict["phone_F2.wav"])"""
    ))

    # Cell 21: Test 2 markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 11. Test Recording 2: `phone_M2.wav`
Telephone channel, male speaker. Display segmentation plot and audio players."""
    ))

    # Cell 22: Test 2 code
    cells.append(nbformat.v4.new_code_cell(
"""plot_and_play_test_file(test_results_dict["phone_M2.wav"])"""
    ))

    # Cell 23: Test 3 markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 12. Test Recording 3: `studio_F2.wav`
High SNR Studio environment, female speaker. Display segmentation plot and audio players."""
    ))

    # Cell 24: Test 3 code
    cells.append(nbformat.v4.new_code_cell(
"""plot_and_play_test_file(test_results_dict["studio_F2.wav"])"""
    ))

    # Cell 25: Test 4 markdown
    cells.append(nbformat.v4.new_markdown_cell(
"""### 13. Test Recording 4: `studio_M2.wav`
High SNR Studio environment, male speaker. Display segmentation plot and audio players."""
    ))

    # Cell 26: Test 4 code
    cells.append(nbformat.v4.new_code_cell(
"""plot_and_play_test_file(test_results_dict["studio_M2.wav"])"""
    ))

    nb.cells = cells
    return nb

def main():
    print("[*] Generating TT1-main notebook from src/TT1/1-mine.ipynb...")
    nb = build_tt1_main_notebook()

    print("[*] Executing notebook to compute and save outputs...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()

    # Destination paths:
    # 1. In src/TT1/TT1-main.ipynb
    dest_tt1 = PROJECT_ROOT / "src" / "TT1" / "TT1-main.ipynb"
    with open(dest_tt1, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[V] Successfully written: {dest_tt1}")

    # 2. At project root TT1-main.ipynb
    dest_root = PROJECT_ROOT / "TT1-main.ipynb"
    with open(dest_root, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[V] Successfully written: {dest_root}")

if __name__ == "__main__":
    main()
