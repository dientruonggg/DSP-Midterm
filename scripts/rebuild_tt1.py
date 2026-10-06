"""
Script to rebuild TT1 notebooks (1-mine.ipynb and main.ipynb) with granular, decoupled cells
and individual evaluation/audio player cells for each test file.
"""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_tt1_vi():
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    }
    cells = []

    # Title
    cells.append(nbformat.v4.new_markdown_cell("""# THUẬT TOÁN 1 (TT1): PHÂN ĐOẠN TIẾNG NÓI / KHOẢNG LẶNG (VAD)
## TỐI ƯU HÓA NGƯỠNG NĂNG LƯỢNG NGẮN HẠN (STE) BẰNG TÌM KIẾM NHỊ PHÂN"""))

    # Cell 1: Setup & Config
    cells.append(nbformat.v4.new_markdown_cell("""### 1. Cấu hình môi trường và tham số hệ thống
Khai báo thư viện toán học cơ sở (`wave`, `numpy`, `matplotlib`), nạp `IPython.display.Audio` và thiết lập tham số framing."""))
    cells.append(nbformat.v4.new_code_cell("""from pathlib import Path
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
OUTPUT_DIR = PROJECT_ROOT / "src" / "TT1" / "output" / "mine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình DSP chuẩn hóa
FRAME_MS = 25        # Độ dài frame (ms)
HOP_MS = 10          # Bước trượt frame (ms)
MIN_SILENCE_MS = 200 # Khoảng lặng tối thiểu (ms) để lọc khoảng lặng ảo"""))

    # Cell 2: read_wav
    cells.append(nbformat.v4.new_markdown_cell("""### 2. Đọc file âm thanh WAV (PCM 16-bit)
Đọc dữ liệu sóng âm, chuẩn hóa biên độ về khoảng `[-1, 1]`, trích xuất tần số lấy mẫu `fs` và biên độ cực đại."""))
    cells.append(nbformat.v4.new_code_cell("""def read_wav(wav_path):
    \"\"\"Đọc file WAV 16-bit PCM, chuẩn hóa biên độ về [-1, 1], lấy fs và max biên độ.\"\"\"
    with wave.open(str(wav_path), "rb") as wf:
        fs = wf.getframerate()
        samples = wf.readframes(wf.getnframes())
    signal = np.frombuffer(samples, dtype=np.int16).astype(float)
    max_amp = np.max(np.abs(signal))
    return signal / max_amp, fs, max_amp"""))

    # Cell 3: read_lab
    cells.append(nbformat.v4.new_markdown_cell("""### 3. Đọc nhãn phân đoạn Praat .lab (Ground-Truth)
Trích xuất mốc thời gian bắt đầu và kết thúc chuẩn của câu nói từ các nhãn hữu thanh (`v`) và vô thanh (`uv`)."""))
    cells.append(nbformat.v4.new_code_cell("""def read_lab(lab_path):
    \"\"\"Đọc file nhãn .lab, trả về biên chuẩn (start, end) của vùng tiếng nói (v và uv).\"\"\"
    speech = []
    for line in lab_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].lower() in {"v", "uv"}:
            speech.append((float(parts[0]), float(parts[1])))
    return round(min(s[0] for s in speech), 4), round(max(s[1] for s in speech), 4)"""))

    # Cell 4: compute_ste
    cells.append(nbformat.v4.new_markdown_cell("""### 4. Tính toán Năng lượng ngắn hạn STE chuẩn hóa
Phân khung tín hiệu (25 ms, hop 10 ms) và tính trung bình bình phương biên độ cho từng khung, chuẩn hóa về `[0, 1]`."""))
    cells.append(nbformat.v4.new_code_cell("""def compute_ste(signal, fs, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    \"\"\"Phân khung và tính Short-Time Energy (STE) chuẩn hóa [0, 1].\"\"\"
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
        ste[i] = np.mean(frame ** 2)
        frame_starts[i] = start / fs

    max_ste = float(np.max(ste))
    ste_norm = ste / max(max_ste, 1e-12)
    return ste_norm, frame_starts, max_ste"""))

    # Cell 5: Model Class
    cells.append(nbformat.v4.new_markdown_cell("""### 5. Lớp mô hình VAD Tìm kiếm nhị phân (EnergyBinarySearchVAD)
Định nghĩa thuật toán phân loại frame theo ngưỡng $T$, lấp khoảng lặng ảo $< 200\\text{ ms}$ và co hẹp nhị phân theo sai lệch thời lượng."""))
    cells.append(nbformat.v4.new_code_cell("""class EnergyBinarySearchVAD:
    \"\"\"Mô hình VAD tối ưu hóa ngưỡng STE bằng tìm kiếm nhị phân theo sai lệch thời lượng.\"\"\"
    def __init__(self, epochs=40, search_range=(1e-4, 0.05)):
        self.epochs = epochs
        self.search_range = search_range
        self.threshold = None

    def predict(self, ste_norm, frame_starts, duration, threshold=None):
        \"\"\"Phân loại khung theo ngưỡng T và lấp khoảng lặng ảo < 200 ms.\"\"\"
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
        \"\"\"Huấn luyện co hẹp khoảng nhị phân [low, high] qua 40 epochs.\"\"\"
        low, high = self.search_range
        print(f"Bắt đầu tìm kiếm nhị phân ngưỡng STE trên miền [{low:.6f}, {high:.6f}] ({self.epochs} epochs):")
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

            if epoch <= 10 or epoch % 5 == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:02d} | T = {threshold:.8f} | Sai lệch = {mean_error:+.4f} s | Train MAE = {epoch_mae:5.2f} ms | Train RMSE = {epoch_rmse:5.2f} ms")

            if mean_error > 0:
                low = threshold
            else:
                high = threshold

        self.threshold = (low + high) / 2.0
        print("-" * 96)
        print(f">>> NGƯỠNG NĂNG LƯỢNG TỐI ƯU TÌM ĐƯỢC: T = {self.threshold:.8f}")
        return self"""))

    # Cell 6: Load Train Data
    cells.append(nbformat.v4.new_markdown_cell("""### 6. Nạp tập dữ liệu huấn luyện (Training Data)
Nạp 04 file huấn luyện (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`), tính STE và nhãn ground-truth."""))
    cells.append(nbformat.v4.new_code_cell("""train_data = []
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
print(f"Đã nạp {len(train_data)} file huấn luyện thành công.")"""))

    # Cell 7: Train Model
    cells.append(nbformat.v4.new_markdown_cell("""### 7. Huấn luyện tìm kiếm nhị phân tìm ngưỡng tối ưu T_opt
Thực thi quá trình huấn luyện co hẹp khoảng giá trị ngưỡng qua 40 epochs."""))
    cells.append(nbformat.v4.new_code_cell("""model = EnergyBinarySearchVAD(epochs=40)
model.fit(train_data)"""))

    # Cell 8: Unnormalize thresholds
    cells.append(nbformat.v4.new_markdown_cell("""### 8. Giải nén ngưỡng về thang đo thực tế của từng file
Quy đổi ngưỡng chuẩn hóa $T_{opt}$ về giá trị năng lượng STE thô và biên độ mẫu tương đương."""))
    cells.append(nbformat.v4.new_code_cell("""print(">>> GIẢI NÉN NGƯỠNG VỀ THANG ĐO THỰC TẾ:")
for item in train_data:
    t_raw_ste = model.threshold * item["max_ste"]
    t_amp = np.sqrt(model.threshold) * item["max_amp"]
    print(f"  {item['name']:<15} (max_amp = {item['max_amp']:>7.0f}, max_ste = {item['max_ste']:.4f}) -> STE thực = {t_raw_ste:.6f} | Biên độ = {t_amp:.1f}")"""))

    # Cell 9: Test Evaluation
    cells.append(nbformat.v4.new_markdown_cell("""### 9. Đánh giá định lượng trên tập Kiểm thử (Test Data)
Chạy thuật toán với ngưỡng tối ưu tìm được trên 04 file kiểm thử và tổng kết bảng sai số MAE, RMSE."""))
    cells.append(nbformat.v4.new_code_cell("""test_results_dict = {}
test_results = []
print("=" * 96)
print(f"{'BẢNG ĐÁNH GIÁ ĐỊNH LƯỢNG TRÊN TẬP KIỂM THỬ (TEST EVALUATION)':^96}")
print("=" * 96)
print(f"{'File':<16} | {'Biên Chuẩn (s)':<16} | {'Biên Đoán (s)':<16} | {'ΔStart':<10} | {'ΔEnd':<10} | {'MAE (ms)':<9} | {'RMSE (ms)':<9}")
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
print(f"TỔNG KẾT TẬP TEST:  Mean MAE = {mean_mae:.2f} ms  |  Mean RMSE = {mean_rmse:.2f} ms")
print("=" * 96)"""))

    # Cell 10: Plot helper function
    cells.append(nbformat.v4.new_markdown_cell("""### 10. Hàm hiển thị đồ thị và nhúng Audio Player
Định nghĩa hàm vẽ chi tiết (Waveform, STE, ngưỡng, biên phân đoạn) kèm 2 Audio Player (toàn bài và đoạn tiếng nói)."""))
    cells.append(nbformat.v4.new_code_cell("""def plot_and_play_test_file(res):
    \"\"\"Vẽ đồ thị chi tiết cho một file kiểm thử và hiển thị các Audio Player.\"\"\"
    fig, ax = plt.subplots(figsize=(13, 4.5))
    time_signal = np.arange(len(res["signal"])) / res["fs"]
    frame_centers = res["frame_starts"] + (FRAME_MS / 2000.0)

    ax.plot(time_signal, res["signal"], color="gray", linewidth=0.7, label="Waveform chuẩn hóa")
    ax.plot(frame_centers, res["ste_norm"], color="red", linewidth=1.1, label="STE chuẩn hóa")
    ax.axhline(model.threshold, color="blue", linestyle="--", linewidth=1.2, label=f"Ngưỡng T = {model.threshold:.6f}")

    ax.fill_between(frame_centers, 0, 1, where=res["labels"] == 1, color="lightgreen", alpha=0.25, label="Vùng Speech dự đoán")

    ax.axvline(res["gt"][0], color="red", linestyle="--", linewidth=1.8, label="Biên chuẩn Ground-truth (LAB)")
    ax.axvline(res["gt"][1], color="red", linestyle="--", linewidth=1.8)
    ax.axvline(res["pred"][0], color="green", linestyle=":", linewidth=2.2, label="Biên dự đoán thuật toán")
    ax.axvline(res["pred"][1], color="green", linestyle=":", linewidth=2.2)

    ax.set_title(f"{res['file']}  |  MAE = {res['mae']:.2f} ms  |  RMSE = {res['rmse']:.2f} ms", fontsize=12, fontweight="bold")
    ax.set_xlabel("Thời gian (giây)")
    ax.set_ylabel("Biên độ / STE")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    info_text = (
        f"Ground-truth: [{res['gt'][0]:.2f}s, {res['gt'][1]:.2f}s]\\n"
        f"Dự đoán:      [{res['pred'][0]:.2f}s, {res['pred'][1]:.2f}s]\\n"
        f"ΔStart: {res['d_start']:+.1f} ms | ΔEnd: {res['d_end']:+.1f} ms\\n"
        f"MAE: {res['mae']:.2f} ms | RMSE: {res['rmse']:.2f} ms"
    )
    ax.text(0.015, 0.95, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"))

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{Path(res['file']).stem}.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Đã xuất và lưu đồ thị: {fig_path}")
    plt.show()

    # Nạp và hiển thị audio player cho file âm thanh cục bộ
    wav_file_path = TEST_DIR / res['file']
    print(f"🎵 Audio gốc ({res['file']}):")
    if wav_file_path.exists():
        display(Audio(filename=str(wav_file_path)))
    else:
        display(Audio(data=res['signal'], rate=res['fs']))

    # Hiển thị audio player cho phân đoạn tiếng nói nhận diện được
    st_sample = int(res['pred'][0] * res['fs'])
    en_sample = min(len(res['signal']), int(res['pred'][1] * res['fs']))
    if en_sample > st_sample:
        print(f"🔊 Đoạn tiếng nói nhận diện [{res['pred'][0]:.2f}s - {res['pred'][1]:.2f}s]:")
        display(Audio(data=res['signal'][st_sample:en_sample], rate=res['fs']))"""))

    # Cell 11: Test File 1
    cells.append(nbformat.v4.new_markdown_cell("""### 11. Thực nghiệm kiểm thử 1: `phone_F2.wav`
Môi trường điện thoại, giọng nữ. Hiển thị đồ thị phân đoạn và trình phát audio."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_file(test_results_dict["phone_F2.wav"])"""))

    # Cell 12: Test File 2
    cells.append(nbformat.v4.new_markdown_cell("""### 12. Thực nghiệm kiểm thử 2: `phone_M2.wav`
Môi trường điện thoại, giọng nam. Hiển thị đồ thị phân đoạn và trình phát audio."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_file(test_results_dict["phone_M2.wav"])"""))

    # Cell 13: Test File 3
    cells.append(nbformat.v4.new_markdown_cell("""### 13. Thực nghiệm kiểm thử 3: `studio_F2.wav`
Môi trường phòng thu Studio chất lượng cao, giọng nữ. Hiển thị đồ thị phân đoạn và trình phát audio."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_file(test_results_dict["studio_F2.wav"])"""))

    # Cell 14: Test File 4
    cells.append(nbformat.v4.new_markdown_cell("""### 14. Thực nghiệm kiểm thử 4: `studio_M2.wav`
Môi trường phòng thu Studio chất lượng cao, giọng nam. Hiển thị đồ thị phân đoạn và trình phát audio."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_file(test_results_dict["studio_M2.wav"])"""))

    nb.cells = cells
    return nb


def build_tt1_en():
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    }
    cells = []

    # Title
    cells.append(nbformat.v4.new_markdown_cell("""# ALGORITHM 1 (TT1): VOICE ACTIVITY DETECTION (VAD)
## SHORT-TIME ENERGY (STE) OPTIMIZATION VIA BINARY SEARCH"""))

    # Cell 1: Setup & Config
    cells.append(nbformat.v4.new_markdown_cell("""### 1. Environment Configuration and Library Imports
Import standard numerical and audio utilities, initialize `IPython.display.Audio`, and define framing parameters."""))
    cells.append(nbformat.v4.new_code_cell("""from pathlib import Path
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
HOP_MS = 10          # Frame hop (ms)
MIN_SILENCE_MS = 200 # Minimum silence bridging threshold (ms)"""))

    # Cell 2: read_wav
    cells.append(nbformat.v4.new_markdown_cell("""### 2. Audio Reading (16-bit PCM WAV)
Load audio stream, normalize amplitude into `[-1.0, 1.0]`, and retrieve sample rate `fs`."""))
    cells.append(nbformat.v4.new_code_cell("""def read_wav(wav_path):
    \"\"\"Read 16-bit PCM WAV, normalize amplitude to [-1, 1], return fs and max amplitude.\"\"\"
    with wave.open(str(wav_path), "rb") as wf:
        fs = wf.getframerate()
        samples = wf.readframes(wf.getnframes())
    signal = np.frombuffer(samples, dtype=np.int16).astype(float)
    max_amp = np.max(np.abs(signal))
    return signal / max_amp, fs, max_amp"""))

    # Cell 3: read_lab
    cells.append(nbformat.v4.new_markdown_cell("""### 3. Praat Ground-Truth Parsing (.lab)
Parse phonetic time intervals and extract ground-truth speech boundaries from voiced (`v`) and unvoiced (`uv`) segments."""))
    cells.append(nbformat.v4.new_code_cell("""def read_lab(lab_path):
    \"\"\"Read Praat .lab file, return ground-truth speech boundaries (start, end).\"\"\"
    speech = []
    for line in lab_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].lower() in {"v", "uv"}:
            speech.append((float(parts[0]), float(parts[1])))
    return round(min(s[0] for s in speech), 4), round(max(s[1] for s in speech), 4)"""))

    # Cell 4: compute_ste
    cells.append(nbformat.v4.new_markdown_cell("""### 4. Normalized Short-Time Energy (STE) Computation
Frame the signal and compute normalized mean squared energy `STE_norm` $\\in [0, 1]$ per frame."""))
    cells.append(nbformat.v4.new_code_cell("""def compute_ste(signal, fs, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    \"\"\"Frame signal and compute normalized short-time energy [0, 1].\"\"\"
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
        ste[i] = np.mean(frame ** 2)
        frame_starts[i] = start / fs

    max_ste = float(np.max(ste))
    ste_norm = ste / max(max_ste, 1e-12)
    return ste_norm, frame_starts, max_ste"""))

    # Cell 5: Model Class
    cells.append(nbformat.v4.new_markdown_cell("""### 5. Binary Search Energy VAD Model Class
Formulate threshold bisection optimization to eliminate signed speech duration errors on the training set."""))
    cells.append(nbformat.v4.new_code_cell("""class EnergyBinarySearchVAD:
    \"\"\"Binary search VAD model optimizing STE threshold on signed duration error.\"\"\"
    def __init__(self, epochs=40, search_range=(1e-4, 0.05)):
        self.epochs = epochs
        self.search_range = search_range
        self.threshold = None

    def predict(self, ste_norm, frame_starts, duration, threshold=None):
        \"\"\"Classify frames with threshold T and bridge silences < 200 ms.\"\"\"
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
        \"\"\"Train threshold bisection across 40 epochs.\"\"\"
        low, high = self.search_range
        print(f"Starting binary search on range [{low:.6f}, {high:.6f}] ({self.epochs} epochs):")
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

            if epoch <= 10 or epoch % 5 == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:02d} | T = {threshold:.8f} | Bias = {mean_error:+.4f} s | Train MAE = {epoch_mae:5.2f} ms | Train RMSE = {epoch_rmse:5.2f} ms")

            if mean_error > 0:
                low = threshold
            else:
                high = threshold

        self.threshold = (low + high) / 2.0
        print("-" * 96)
        print(f">>> OPTIMAL STE THRESHOLD ACQUIRED: T = {self.threshold:.8f}")
        return self"""))

    # Cell 6: Load Train Data
    cells.append(nbformat.v4.new_markdown_cell("""### 6. Training Dataset Ingestion
Load 4 training recordings (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`) with ground-truth boundaries."""))
    cells.append(nbformat.v4.new_code_cell("""train_data = []
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
print(f"Loaded {len(train_data)} training files successfully.")"""))

    # Cell 7: Train Model
    cells.append(nbformat.v4.new_markdown_cell("""### 7. Binary Search Model Training
Execute 40-epoch bisection training to derive the global optimal threshold."""))
    cells.append(nbformat.v4.new_code_cell("""model = EnergyBinarySearchVAD(epochs=40)
model.fit(train_data)"""))

    # Cell 8: Unnormalize thresholds
    cells.append(nbformat.v4.new_markdown_cell("""### 8. Physical Scale Threshold Mapping
Map normalized $T_{opt}$ back to unnormalized energy levels and amplitude scales."""))
    cells.append(nbformat.v4.new_code_cell("""print(">>> THRESHOLD MAPPING TO PHYSICAL UNITS:")
for item in train_data:
    t_raw_ste = model.threshold * item["max_ste"]
    t_amp = np.sqrt(model.threshold) * item["max_amp"]
    print(f"  {item['name']:<15} (max_amp = {item['max_amp']:>7.0f}, max_ste = {item['max_ste']:.4f}) -> Raw STE = {t_raw_ste:.6f} | Equivalent Amp = {t_amp:.1f}")"""))

    # Cell 9: Test Evaluation
    cells.append(nbformat.v4.new_markdown_cell("""### 9. Quantitative Test Benchmark Evaluation
Apply the optimized threshold to the 4 test recordings and report MAE and RMSE metrics."""))
    cells.append(nbformat.v4.new_code_cell("""test_results_dict = {}
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
print("=" * 96)"""))

    # Cell 10: Plot helper function
    cells.append(nbformat.v4.new_markdown_cell("""### 10. Visualization Helper & Audio Player Integration
Plot detailed test waveforms, STE curves, decision boundaries, and render audio playback widgets."""))
    cells.append(nbformat.v4.new_code_cell("""def plot_and_play_test_file(res):
    \"\"\"Plot test results and display embedded audio players.\"\"\"
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

    # Load and display local audio player widget
    wav_file_path = TEST_DIR / res['file']
    print(f"🎵 Original Audio ({res['file']}):")
    if wav_file_path.exists():
        display(Audio(filename=str(wav_file_path)))
    else:
        display(Audio(data=res['signal'], rate=res['fs']))

    # Display audio player for predicted speech segment
    st_sample = int(res['pred'][0] * res['fs'])
    en_sample = min(len(res['signal']), int(res['pred'][1] * res['fs']))
    if en_sample > st_sample:
        print(f"🔊 Predicted Speech Segment [{res['pred'][0]:.2f}s - {res['pred'][1]:.2f}s]:")
        display(Audio(data=res['signal'][st_sample:en_sample], rate=res['fs']))"""))

    # Cell 11: Test File 1
    cells.append(nbformat.v4.new_markdown_cell("""### 11. Test Recording 1: `phone_F2.wav`
Telephone channel, female speaker. Waveform, STE overlay, and audio players."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_file(test_results_dict["phone_F2.wav"])"""))

    # Cell 12: Test File 2
    cells.append(nbformat.v4.new_markdown_cell("""### 12. Test Recording 2: `phone_M2.wav`
Telephone channel, male speaker. Waveform, STE overlay, and audio players."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_file(test_results_dict["phone_M2.wav"])"""))

    # Cell 13: Test File 3
    cells.append(nbformat.v4.new_markdown_cell("""### 13. Test Recording 3: `studio_F2.wav`
High SNR Studio environment, female speaker. Waveform, STE overlay, and audio players."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_file(test_results_dict["studio_F2.wav"])"""))

    # Cell 14: Test File 4
    cells.append(nbformat.v4.new_markdown_cell("""### 14. Test Recording 4: `studio_M2.wav`
High SNR Studio environment, male speaker. Waveform, STE overlay, and audio players."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_file(test_results_dict["studio_M2.wav"])"""))

    nb.cells = cells
    return nb


def run():
    p_vi = PROJECT_ROOT / "src" / "TT1" / "1-mine.ipynb"
    print(f"[*] Building and executing {p_vi}...")
    nb_vi = build_tt1_vi()
    client_vi = NotebookClient(nb_vi, timeout=60, kernel_name="python3")
    client_vi.execute()
    with open(p_vi, "w", encoding="utf-8") as f:
        nbformat.write(nb_vi, f)
    print(f"[V] Successfully saved {p_vi} with {len(nb_vi.cells)} cells!")

    p_en = PROJECT_ROOT / "src" / "TT1" / "main.ipynb"
    print(f"[*] Building and executing {p_en}...")
    nb_en = build_tt1_en()
    client_en = NotebookClient(nb_en, timeout=60, kernel_name="python3")
    client_en.execute()
    with open(p_en, "w", encoding="utf-8") as f:
        nbformat.write(nb_en, f)
    print(f"[V] Successfully saved {p_en} with {len(nb_en.cells)} cells!")

if __name__ == "__main__":
    run()
