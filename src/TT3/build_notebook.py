"""
Script: TT3/build_notebook.py
Mục đích: Tự động khởi tạo và chạy (execute) toàn bộ 2 notebook:
          1. TT3_Gaussian_VAD.ipynb (Tiếng Việt)
          2. TT3_Gaussian_VAD_EN.ipynb (English)
          Lưu vào thư mục TT3/notebook/
          Các cell được module hóa cao độ, markdown ngắn gọn, súc tích,
          đầy đủ biểu đồ và widget nghe âm thanh độc lập cho từng file kiểm thử.
"""

import os
import shutil
from pathlib import Path
import nbformat
from nbclient import NotebookClient

# Tìm kiếm thư mục gốc dự án linh hoạt
def find_project_root() -> Path:
    curr = Path(__file__).resolve()
    for candidate in [curr, *curr.parents]:
        if (candidate / "TinHieuHuanLuyen").exists() and (candidate / "TinHieuKiemThu").exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy thư mục gốc của dự án DSP_MidTerm.")

PROJECT_ROOT = find_project_root()


def build_tt3_notebook_vi():
    """Tạo notebook tiếng Việt tuân thủ cấu trúc slide và yêu cầu giữa kỳ."""
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    }

    cells = []

    # SLIDE 1: COVER SLIDE
    cells.append(nbformat.v4.new_markdown_cell(r"""# BÁO CÁO THỰC NGHIỆM GIỮA KỲ - XỬ LÝ TÍN HIỆU SỐ
## THUẬT TOÁN 3 (TT3): PHÂN ĐOẠN TIẾNG NÓI / KHOẢNG LẶNG (VAD) BẰNG PHÂN BỐ GAUSS

- **Cấu hình DSP**: Độ dài khung 25 ms, bước trượt 10 ms, lọc khoảng lặng ảo < 200 ms.
- **Phương pháp**: Ước lượng phân bố Gauss ($\mu, \sigma$), giải phương trình Bayes xác định ngưỡng $T_{opt}$.
- **Đánh giá**: Định lượng MAE, RMSE (ms) và đồ thị phân đoạn trên 04 tệp kiểm thử."""))

    # CELL 1: SETUP & IMPORTS
    cells.append(nbformat.v4.new_markdown_cell("""### 1. Cấu hình môi trường và nạp thư viện
Khai báo thư viện chuẩn (`wave`, `math`, `os`, `glob`), `numpy`, `matplotlib` và `IPython.display.Audio`."""))

    cells.append(nbformat.v4.new_code_cell("""import os
import glob
import math
import wave
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Cấu hình hiển thị đồ thị chuẩn báo cáo học thuật
plt.rcParams.update({
    "font.size": 10,
    "figure.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": ":"
})
print("Môi trường thực thi đã sẵn sàng.")"""))

    # CELL 2: PATHS
    cells.append(nbformat.v4.new_markdown_cell("""### 2. Thiết lập đường dẫn thư mục dự án
Xác định thư mục dữ liệu `TinHieuHuanLuyen`, `TinHieuKiemThu` và thư mục lưu trữ kết quả đầu ra."""))

    cells.append(nbformat.v4.new_code_cell("""def find_project_root() -> Path:
    curr = Path.cwd()
    for candidate in [curr, *curr.parents]:
        if (candidate / "TinHieuHuanLuyen").exists() and (candidate / "TinHieuKiemThu").exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy thư mục gốc của dự án.")

PROJECT_ROOT = find_project_root()
TRAIN_DIR = PROJECT_ROOT / "TinHieuHuanLuyen"
TEST_DIR = PROJECT_ROOT / "TinHieuKiemThu"
OUTPUT_DIR = PROJECT_ROOT / "src" / "TT3" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Thư mục huấn luyện : {TRAIN_DIR}")
print(f"Thư mục kiểm thử   : {TEST_DIR}")
print(f"Thư mục xuất ảnh   : {OUTPUT_DIR}")"""))

    # CELL 3: READ WAV
    cells.append(nbformat.v4.new_markdown_cell("""### 3. Đọc dữ liệu âm thanh từ file WAV (PCM 16-bit)
Giải mã dữ liệu WAV 16-bit PCM và chuẩn hóa biên độ về đoạn `[-1.0, 1.0]`."""))

    cells.append(nbformat.v4.new_code_cell("""def read_wav(file_path: str) -> Tuple[np.ndarray, int]:
    \"\"\"Đọc file âm thanh WAV định dạng PCM 16-bit, chuẩn hóa về [-1.0, 1.0].\"\"\"
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Chỉ hỗ trợ file WAV 16-bit PCM (sample_width=2), hiện tại: {sample_width}")

    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    if n_channels > 1:
        audio_int16 = audio_int16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    signal = audio_int16.astype(np.float64) / 32768.0
    return signal, sample_rate

print("Hàm read_wav đã được định nghĩa.")"""))

    # CELL 4: READ LAB
    cells.append(nbformat.v4.new_markdown_cell("""### 4. Đọc nhãn Praat .lab và trích xuất mốc tiếng nói chuẩn (Ground Truth)
Trích xuất mốc thời gian bắt đầu và kết thúc của câu nói dựa trên các nhãn `v` và `uv`."""))

    cells.append(nbformat.v4.new_code_cell("""def read_lab(lab_path: str) -> List[Tuple[float, float, str]]:
    \"\"\"Đọc và phân tích file nhãn mốc thời gian phân đoạn (.lab) của Praat.\"\"\"
    segments: List[Tuple[float, float, str]] = []
    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] not in ("F0mean", "F0std"):
                try:
                    segments.append((float(parts[0]), float(parts[1]), parts[2].lower()))
                except ValueError:
                    continue
    return segments


def get_speech_groundtruth(lab_segments: List[Tuple[float, float, str]]) -> Tuple[float, float]:
    \"\"\"Xác định mốc bắt đầu và kết thúc toàn bộ câu nói chuẩn từ danh sách các đoạn nhãn.\"\"\"
    speech_segments = [seg for seg in lab_segments if seg[2] in ("v", "uv")]
    if not speech_segments:
        return 0.0, 0.0
    t_start = min(seg[0] for seg in speech_segments)
    t_end = max(seg[1] for seg in speech_segments)
    return t_start, t_end

print("Hàm read_lab và get_speech_groundtruth đã được định nghĩa.")"""))

    # CELL 5: FRAMING
    cells.append(nbformat.v4.new_markdown_cell("""### 5. Cài đặt hàm phân khung tín hiệu (Signal Framing)
Chia tín hiệu thành các khung 25 ms với bước nhảy 10 ms (độ chồng chập 60%)."""))

    cells.append(nbformat.v4.new_code_cell("""def frame_signal(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Phân chia tín hiệu âm thanh thành các khung (frames) chồng lấp bằng chỉ số Numpy cơ bản.\"\"\"
    frame_len = int(round(frame_size_ms * sample_rate / 1000.0))
    hop_len = int(round(hop_size_ms * sample_rate / 1000.0))

    num_samples = len(signal)
    if num_samples < frame_len:
        pad_len = frame_len - num_samples
        padded = np.zeros(frame_len, dtype=signal.dtype)
        padded[:num_samples] = signal
        signal = padded
        num_samples = len(signal)

    num_frames = 1 + int(np.floor((num_samples - frame_len) / hop_len))
    frames = np.zeros((num_frames, frame_len), dtype=np.float64)
    frame_times = np.zeros(num_frames, dtype=np.float64)

    for m in range(num_frames):
        start_idx = m * hop_len
        end_idx = start_idx + frame_len
        frames[m, :] = signal[start_idx:end_idx]
        frame_times[m] = (start_idx + end_idx) / (2.0 * sample_rate)

    return frames, frame_times

print("Hàm frame_signal đã được định nghĩa.")"""))

    # CELL 6: STE
    cells.append(nbformat.v4.new_markdown_cell(r"""### 6. Tính toán Năng lượng ngắn hạn STE chuẩn hóa
Tính năng lượng ngắn hạn $STE[m] = \sum x^2[m \cdot H + n]$ và chuẩn hóa về `[0.0, 1.0]`."""))

    cells.append(nbformat.v4.new_code_cell("""def compute_ste(frames: np.ndarray) -> np.ndarray:
    \"\"\"Tính năng lượng ngắn hạn (STE) bằng tổng bình phương biên độ mẫu trong khung.\"\"\"
    return np.sum(np.square(frames), axis=1)


def normalize_ste(ste: np.ndarray) -> np.ndarray:
    \"\"\"Chuẩn hóa vector năng lượng ngắn hạn về đoạn [0.0, 1.0].\"\"\"
    max_energy = np.max(ste)
    if max_energy <= 1e-12:
        return np.zeros_like(ste)
    return ste / max_energy


def extract_ste_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Quy trình tích hợp: Phân khung (25ms, hop 10ms) -> Tính STE -> Chuẩn hóa STE về [0, 1].\"\"\"
    frames, frame_times = frame_signal(signal, sample_rate, frame_size_ms, hop_size_ms)
    ste_raw = compute_ste(frames)
    ste_norm = normalize_ste(ste_raw)
    return ste_norm, frame_times

print("Các hàm trích xuất đặc trưng STE đã được định nghĩa.")"""))

    # SLIDE 2: TRAINING DATA SURVEY
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 2: KHẢO SÁT DỮ LIỆU HUẤN LUYỆN (TRAINING DATA SURVEY)
Trích xuất phân đoạn tiếng nói và khoảng lặng trên 04 file huấn luyện từ nhãn ground-truth .lab."""))

    cells.append(nbformat.v4.new_code_cell("""def survey_training_data(training_dir: str, frame_size_ms: float = 25.0, hop_size_ms: float = 10.0):
    \"\"\"Khảo sát toàn bộ các khung tín hiệu trên 4 file huấn luyện, phân chia thành silence và speech.\"\"\"
    wav_files = sorted(glob.glob(os.path.join(training_dir, "*.wav")))
    all_silence_ste: List[float] = []
    all_speech_ste: List[float] = []
    per_file_stats: Dict[str, Dict[str, float]] = {}

    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        signal, sample_rate = read_wav(wav_path)
        ste_norm, frame_times = extract_ste_features(signal, sample_rate, frame_size_ms, hop_size_ms)
        lab_segments = read_lab(lab_path)

        file_sil_ste = []
        file_sp_ste = []

        for ste_val, t_center in zip(ste_norm, frame_times):
            label = "sil"
            for seg in lab_segments:
                if seg[0] <= t_center <= seg[1]:
                    label = seg[2]
                    break
            if label in ("v", "uv"):
                file_sp_ste.append(ste_val)
                all_speech_ste.append(ste_val)
            else:
                file_sil_ste.append(ste_val)
                all_silence_ste.append(ste_val)

        per_file_stats[fname] = {
            "sil_count": len(file_sil_ste),
            "sil_mean": float(np.mean(file_sil_ste)) if file_sil_ste else 0.0,
            "sil_std": float(np.std(file_sil_ste)) if file_sil_ste else 0.0,
            "sp_count": len(file_sp_ste),
            "sp_mean": float(np.mean(file_sp_ste)) if file_sp_ste else 0.0,
            "sp_std": float(np.std(file_sp_ste)) if file_sp_ste else 0.0
        }

    return np.array(all_silence_ste), np.array(all_speech_ste), per_file_stats

silence_ste, speech_ste, train_stats = survey_training_data(str(TRAIN_DIR))
print(f"Tổng số khung khoảng lặng (Silence Frames): {len(silence_ste)}")
print(f"Tổng số khung tiếng nói     (Speech Frames) : {len(speech_ste)}")"""))

    # SLIDE 3: GAUSSIAN PARAMETERS ESTIMATION
    cells.append(nbformat.v4.new_markdown_cell(r"""## SLIDE 3: THỐNG KÊ PHÂN BỐ GAUSS (meanSil, stdSil, meanSp, stdSp)
Ước lượng kỳ vọng $\mu$ và độ lệch chuẩn $\sigma$ của STE cho khoảng lặng và tiếng nói."""))

    cells.append(nbformat.v4.new_code_cell("""def estimate_gaussian_parameters(silence_ste: np.ndarray, speech_ste: np.ndarray):
    \"\"\"Ước lượng kỳ vọng (mu) và độ lệch chuẩn (sigma) cho hai phân bố Gauss.\"\"\"
    mu_sil = float(np.mean(silence_ste))
    sigma_sil = float(np.std(silence_ste))
    mu_sp = float(np.mean(speech_ste))
    sigma_sp = float(np.std(speech_ste))
    return mu_sil, sigma_sil, mu_sp, sigma_sp

mu_sil, sigma_sil, mu_sp, sigma_sp = estimate_gaussian_parameters(silence_ste, speech_ste)

print("=" * 60)
print(f"{'Thông số':<28} | {'Khoảng lặng (Sil)':<15} | {'Tiếng nói (Sp)':<15}")
print("-" * 60)
print(f"{'Kỳ vọng (Mean mu)':<28} | {mu_sil:<15.6f} | {mu_sp:<15.6f}")
print(f"{'Độ lệch chuẩn (Std sigma)':<28} | {sigma_sil:<15.6f} | {sigma_sp:<15.6f}")
print("=" * 60)"""))

    # SLIDE 4: BAYES OPTIMAL THRESHOLD DERIVATION
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 4: GIẢI PHƯƠNG TRÌNH BAYES XÁC ĐỊNH NGƯỠNG TỐI ƯU
Giải phương trình giao điểm phân bố Gauss $p(x|\text{silence}) = p(x|\text{speech})$ để tìm ngưỡng $T_{opt}$."""))

    cells.append(nbformat.v4.new_code_cell("""def solve_bayes_threshold(mu_sil: float, sigma_sil: float, mu_sp: float, sigma_sp: float) -> float:
    \"\"\"Giải phương trình bậc hai xác định giao điểm tối ưu giữa hai hàm mật độ Gauss.\"\"\"
    A = (1.0 / (sigma_sil ** 2)) - (1.0 / (sigma_sp ** 2))
    B = -2.0 * ((mu_sil / (sigma_sil ** 2)) - (mu_sp / (sigma_sp ** 2)))
    C = ((mu_sil ** 2) / (sigma_sil ** 2)) - ((mu_sp ** 2) / (sigma_sp ** 2)) - 2.0 * math.log(sigma_sp / sigma_sil)

    discriminant = B ** 2 - 4.0 * A * C
    if discriminant < 0:
        raise ValueError("Phương trình không có nghiệm thực.")

    sqrt_disc = math.sqrt(discriminant)
    root1 = (-B - sqrt_disc) / (2.0 * A)
    root2 = (-B + sqrt_disc) / (2.0 * A)

    candidates = [r for r in (root1, root2) if mu_sil < r < mu_sp]
    if candidates:
        return candidates[0]
    return min((root1, root2), key=lambda r: abs(r - (mu_sil + mu_sp) / 2.0))

T_opt = solve_bayes_threshold(mu_sil, sigma_sil, mu_sp, sigma_sp)
print(f"[*] NGƯỠNG NĂNG LƯỢNG TỐI ƯU BAYES CHUNG: T_opt = {T_opt:.6f}")"""))

    # SLIDE 5: GAUSSIAN DISTRIBUTIONS VISUALIZATION
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 5: ĐỒ THỊ MINH HỌA PHÂN BỐ GAUSS VÀ NGƯỠNG CẮT PHÂN BIỆT
Trực quan hóa hàm mật độ xác suất (PDF) của khoảng lặng, tiếng nói và vị trí ngưỡng cắt $T_{opt}$."""))

    cells.append(nbformat.v4.new_code_cell("""x_range = np.linspace(0.0, 0.015, 2000)
pdf_sil = (1.0 / (sigma_sil * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu_sil) / sigma_sil) ** 2)
pdf_sp = (1.0 / (sigma_sp * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu_sp) / sigma_sp) ** 2)

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
ax.plot(x_range, pdf_sil, label=f"Khoảng lặng N(μ={mu_sil:.5f}, σ={sigma_sil:.5f})", color="#e53e3e", lw=2.0)
ax.plot(x_range, pdf_sp, label=f"Tiếng nói N(μ={mu_sp:.5f}, σ={sigma_sp:.5f})", color="#3182ce", lw=2.0)
ax.axvline(T_opt, color="#2b6cb0", linestyle="--", lw=2.2, label=f"Ngưỡng tối ưu Bayes T_opt = {T_opt:.6f}")
ax.set_title("Phân bố Gauss của STE và Ngưỡng phân tách tối ưu Bayes (TT3)", fontweight="bold")
ax.set_xlabel("Năng lượng ngắn hạn (STE) chuẩn hóa")
ax.set_ylabel("Mật độ xác suất (PDF)")
ax.set_xlim(0, 0.012)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()"""))

    # CELL 7a: apply_threshold
    cells.append(nbformat.v4.new_markdown_cell("""### 7. Phân loại nhị phân từng khung theo ngưỡng
So sánh năng lượng STE của từng khung với ngưỡng $T_{opt}$ để gán nhãn 1 (tiếng nói) hoặc 0 (khoảng lặng)."""))

    cells.append(nbformat.v4.new_code_cell("""def apply_threshold(ste_norm: np.ndarray, threshold: float) -> np.ndarray:
    \"\"\"Phân loại nhị phân từng khung: 1 nếu STE >= threshold, ngược lại 0.\"\"\"
    return (ste_norm >= threshold).astype(np.int32)"""))

    # CELL 7b: remove_short_silences
    cells.append(nbformat.v4.new_markdown_cell("""### 8. Lọc khoảng lặng ảo ngắn < 200 ms
Lấp đầy các khoảng lặng ảo $< 200\\text{ ms}$ (tương ứng 20 khung với bước trượt 10 ms) nằm giữa câu nói."""))

    cells.append(nbformat.v4.new_code_cell("""def remove_short_silences(
    frame_decisions: np.ndarray,
    hop_size_ms: float = 10.0,
    min_silence_ms: float = 200.0
) -> np.ndarray:
    \"\"\"Loại bỏ các khoảng lặng ảo < 200 ms (tương ứng 20 khung với hop 10 ms).\"\"\"
    min_frames = int(round(min_silence_ms / hop_size_ms))
    smoothed = frame_decisions.copy()
    speech_indices = np.where(smoothed == 1)[0]
    if len(speech_indices) == 0:
        return smoothed

    first_speech, last_speech = speech_indices[0], speech_indices[-1]
    idx = first_speech
    while idx <= last_speech:
        if smoothed[idx] == 0:
            zero_start = idx
            while idx <= last_speech and smoothed[idx] == 0:
                idx += 1
            if (idx - zero_start) < min_frames:
                smoothed[zero_start:idx] = 1
        else:
            idx += 1
    return smoothed"""))

    # CELL 7c: extract_speech_boundaries
    cells.append(nbformat.v4.new_markdown_cell("""### 9. Trích xuất mốc biên câu nói
Xác định mốc bắt đầu $T_{start}$ và kết thúc $T_{end}$ từ chuỗi quyết định khung đã làm mịn."""))

    cells.append(nbformat.v4.new_code_cell("""def extract_speech_boundaries(
    frame_decisions: np.ndarray,
    hop_size_ms: float = 10.0,
    frame_size_ms: float = 25.0
) -> Tuple[float, float]:
    \"\"\"Trích xuất mốc thời gian bắt đầu T_start và kết thúc T_end của câu nói.\"\"\"
    speech_indices = np.where(frame_decisions == 1)[0]
    if len(speech_indices) == 0:
        return 0.0, 0.0

    start_idx, end_idx = speech_indices[0], speech_indices[-1]
    hop_sec = hop_size_ms / 1000.0
    frame_sec = frame_size_ms / 1000.0

    t_start = round(float(start_idx * hop_sec), 4)
    t_end = round(float(end_idx * hop_sec + frame_sec), 4)
    return t_start, t_end"""))

    # CELL 7d: predict_vad
    cells.append(nbformat.v4.new_markdown_cell("""### 10. Hàm tích hợp chu trình phân đoạn VAD
Hàm dự đoán phân đoạn từ sóng âm thô đến các mốc thời gian $[T_{start}, T_{end}]$."""))

    cells.append(nbformat.v4.new_code_cell("""def predict_vad(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
):
    \"\"\"Chu trình phân đoạn hoàn chỉnh từ tín hiệu thô đến mốc thời gian [T_start, T_end].\"\"\"
    ste_norm, frame_times = extract_ste_features(
        signal=signal,
        sample_rate=sample_rate,
        frame_size_ms=frame_size_ms,
        hop_size_ms=hop_size_ms
    )
    raw_decisions = apply_threshold(ste_norm, threshold)
    smoothed = remove_short_silences(raw_decisions, hop_size_ms=hop_size_ms)
    t_start, t_end = extract_speech_boundaries(
        smoothed,
        hop_size_ms=hop_size_ms,
        frame_size_ms=frame_size_ms
    )
    return t_start, t_end, ste_norm, frame_times, smoothed

print("Chu trình predict_vad đã sẵn sàng.")"""))

    # CELL 8: METRICS
    cells.append(nbformat.v4.new_markdown_cell("""### 11. Hàm tính toán sai số định lượng MAE và RMSE
Đo lường sai số tuyệt đối trung bình (MAE) và căn bậc hai sai số bình phương trung bình (RMSE) theo miligiây."""))

    cells.append(nbformat.v4.new_code_cell("""def calculate_mae_rmse(pred_bounds: Tuple[float, float], gt_bounds: Tuple[float, float]) -> Tuple[float, float]:
    \"\"\"Tính MAE và RMSE (miligiây) theo đúng công thức quy định.\"\"\"
    pred_start, pred_end = pred_bounds
    gt_start, gt_end = gt_bounds

    diff_start_ms = abs(pred_start - gt_start) * 1000.0
    diff_end_ms = abs(pred_end - gt_end) * 1000.0

    mae_ms = (diff_start_ms + diff_end_ms) / 2.0
    rmse_ms = math.sqrt((diff_start_ms ** 2 + diff_end_ms ** 2) / 2.0)
    return mae_ms, rmse_ms"""))

    # SLIDE 6: TEST BENCHMARK
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 6: KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG TRÊN TẬP KIỂM THỬ (TEST DATA)
Đánh giá sai số phân đoạn (MAE, RMSE ms) trên 04 tệp kiểm thử độc lập."""))

    cells.append(nbformat.v4.new_code_cell("""test_files = sorted(glob.glob(os.path.join(str(TEST_DIR), "*.wav")))
eval_results = []
test_data_dict = {}

for wav_path in test_files:
    fname = os.path.basename(wav_path)
    lab_path = os.path.splitext(wav_path)[0] + ".lab"

    signal, sample_rate = read_wav(wav_path)
    lab_segments = read_lab(lab_path)
    gt_bounds = get_speech_groundtruth(lab_segments)

    pred_start, pred_end, ste_norm, frame_times, smoothed = predict_vad(
        signal=signal,
        sample_rate=sample_rate,
        threshold=T_opt,
        frame_size_ms=25.0,
        hop_size_ms=10.0
    )
    pred_bounds = (pred_start, pred_end)

    mae_ms, rmse_ms = calculate_mae_rmse(pred_bounds, gt_bounds)
    d_start_ms = abs(pred_start - gt_bounds[0]) * 1000.0
    d_end_ms = abs(pred_end - gt_bounds[1]) * 1000.0

    eval_results.append({
        "file": fname,
        "gt": gt_bounds,
        "pred": pred_bounds,
        "d_start": d_start_ms,
        "d_end": d_end_ms,
        "mae": mae_ms,
        "rmse": rmse_ms
    })
    test_data_dict[fname] = (fname, signal, sample_rate, ste_norm, frame_times, gt_bounds, pred_bounds, mae_ms, rmse_ms)

print("=" * 96)
print(f"{'Tên file kiểm thử':<16} | {'GT [s]':<14} | {'Dự đoán [s]':<14} | {'ΔStart (ms)':<11} | {'ΔEnd (ms)':<10} | {'MAE (ms)':<9} | {'RMSE (ms)':<9}")
print("-" * 96)
for r in eval_results:
    gt_str = f"[{r['gt'][0]:.2f}, {r['gt'][1]:.2f}]"
    pred_str = f"[{r['pred'][0]:.2f}, {r['pred'][1]:.2f}]"
    print(f"{r['file']:<16} | {gt_str:<14} | {pred_str:<14} | {r['d_start']:<11.1f} | {r['d_end']:<10.1f} | {r['mae']:<9.1f} | {r['rmse']:<9.1f}")
print("=" * 96)

avg_mae = np.mean([r["mae"] for r in eval_results])
avg_rmse = np.mean([r["rmse"] for r in eval_results])
print(f"[*] SAI SỐ TUYỆT ĐỐI TRUNG BÌNH TOÀN TẬP (AVERAGE MAE) : {avg_mae:.2f} ms")
print(f"[*] SAI SỐ BÌNH PHƯƠNG TRUNG BÌNH TOÀN TẬP (AVERAGE RMSE): {avg_rmse:.2f} ms")
print("=" * 96)"""))

    # SLIDE 7: Helper function for individual test display
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 7: MINH HỌA FIGURE VÀ AUDIO PLAYER CÁC FILE KIỂM THỬ
Hàm trực quan hóa dạng sóng, đường STE kèm biên phân đoạn và hiển thị 2 audio player cho từng file."""))

    cells.append(nbformat.v4.new_code_cell("""def plot_and_play_test_recording(item):
    \"\"\"Vẽ đồ thị dạng sóng, STE và hiển thị Audio Players cho từng file kiểm thử.\"\"\"
    fname, sig, sr, ste, ft, gt, pred, mae, rmse = item
    time_wave = np.arange(len(sig)) / float(sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, dpi=120)

    # Đồ thị con 1: Dạng sóng âm thanh
    ax1.plot(time_wave, sig, color="#4a5568", lw=0.7, alpha=0.85, label="Tín hiệu âm thanh")
    ax1.axvline(gt[0], color="red", linestyle="-", lw=2.0, label=f"Biên chuẩn GT: [{gt[0]:.2f}s, {gt[1]:.2f}s]")
    ax1.axvline(gt[1], color="red", linestyle="-", lw=2.0)
    ax1.axvline(pred[0], color="blue", linestyle="--", lw=2.0, label=f"Biên TT3 tìm được: [{pred[0]:.2f}s, {pred[1]:.2f}s]")
    ax1.axvline(pred[1], color="blue", linestyle="--", lw=2.0)
    ax1.set_title(f"Kết quả phân đoạn file kiểm thử: {fname} | MAE = {mae:.1f} ms | RMSE = {rmse:.1f} ms", fontweight="bold")
    ax1.set_ylabel("Biên độ")
    ax1.legend(loc="upper right")

    # Đồ thị con 2: STE chuẩn hóa
    ax2.plot(ft, ste, color="#2b6cb0", lw=1.2, label="Đặc trưng STE chuẩn hóa")
    ax2.axhline(T_opt, color="#dd6b20", linestyle=":", lw=2.0, label=f"Ngưỡng T_Bayes = {T_opt:.6f}")
    ax2.axvline(gt[0], color="red", linestyle="-", lw=1.8)
    ax2.axvline(gt[1], color="red", linestyle="-", lw=1.8)
    ax2.axvline(pred[0], color="blue", linestyle="--", lw=1.8)
    ax2.axvline(pred[1], color="blue", linestyle="--", lw=1.8)
    ax2.set_xlabel("Thời gian (giây)")
    ax2.set_ylabel("STE chuẩn hóa")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{Path(fname).stem}.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Đã lưu đồ thị: {fig_path}")
    plt.show()

    # Nạp và hiển thị audio player cho file kiểm thử cục bộ
    test_wav_path = TEST_DIR / fname
    print(f"🎵 Original Audio ({fname}):")
    if test_wav_path.exists():
        display(Audio(filename=str(test_wav_path)))
    else:
        display(Audio(data=sig, rate=sr))

    # Hiển thị audio player cho phân đoạn tiếng nói nhận diện được
    st_idx = int(pred[0] * sr)
    en_idx = min(len(sig), int(pred[1] * sr))
    if en_idx > st_idx:
        print(f"🔊 Predicted Speech Segment TT3 [{pred[0]:.2f}s - {pred[1]:.2f}s]:")
        display(Audio(data=sig[st_idx:en_idx], rate=sr))"""))

    # Test File 1: phone_F2
    cells.append(nbformat.v4.new_markdown_cell("""### 12. Kiểm thử file 1: `phone_F2.wav`
Kênh điện thoại, người nói nữ. Hiển thị đồ thị dạng sóng, STE và 2 audio player."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_recording(test_data_dict["phone_F2.wav"])"""))

    # Test File 2: phone_M2
    cells.append(nbformat.v4.new_markdown_cell("""### 13. Kiểm thử file 2: `phone_M2.wav`
Kênh điện thoại, người nói nam. Hiển thị đồ thị dạng sóng, STE và 2 audio player."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_recording(test_data_dict["phone_M2.wav"])"""))

    # Test File 3: studio_F2
    cells.append(nbformat.v4.new_markdown_cell("""### 14. Kiểm thử file 3: `studio_F2.wav`
Môi trường phòng thu Studio SNR cao, người nói nữ. Hiển thị đồ thị dạng sóng, STE và 2 audio player."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_recording(test_data_dict["studio_F2.wav"])"""))

    # Test File 4: studio_M2
    cells.append(nbformat.v4.new_markdown_cell("""### 15. Kiểm thử file 4: `studio_M2.wav`
Môi trường phòng thu Studio SNR cao, người nói nam. Hiển thị đồ thị dạng sóng, STE và 2 audio player."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_recording(test_data_dict["studio_M2.wav"])"""))

    # SLIDE 8: DISCUSSION & CONCLUSION
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 8: TỔNG KẾT VÀ KẾT LUẬN THỰC NGHIỆM

| Tệp kiểm thử | Môi trường | SNR | Biên GT (s) | Biên TT3 (s) | ΔStart (ms) | ΔEnd (ms) | MAE (ms) | RMSE (ms) |
|---|---|---|---|---|---|---|---|---|
| `phone_F2.wav` | Điện thoại | Thấp | [1.02, 4.04] | [1.03, 4.09] | +10.0 | +45.0 | 27.5 | 32.6 |
| `phone_M2.wav` | Điện thoại | Thấp | [0.53, 2.52] | [0.54, 2.52] | +10.0 | 0.0 | 5.0 | 7.1 |
| `studio_F2.wav` | Studio | Cao | [0.77, 2.37] | [0.78, 2.37] | +10.0 | 0.0 | 5.0 | 7.1 |
| `studio_M2.wav` | Studio | Cao | [0.45, 1.93] | [0.46, 1.95] | +10.0 | +15.0 | 12.5 | 12.7 |
| **Trung bình** | - | - | - | - | **+10.0** | **+15.0** | **12.5** | **14.9** |

- **Độ chính xác**: MAE trung bình đạt $12.5\\text{ ms}$ (xấp xỉ 1 bước nhảy khung 10 ms); sai số mốc đầu $\\Delta_{Start} = 10\\text{ ms}$ đồng nhất trên 4 file.
- **Tính thích nghi**: Ngưỡng Bayes $T_{opt} \\approx 0.002878$ tự động tạo khoảng an toàn chống nhiễu hiệu quả trên cả kênh điện thoại và phòng thu."""))

    nb.cells = cells
    return nb


def build_tt3_notebook_en():
    """Tạo notebook tiếng Anh hoàn chỉnh chuẩn học thuật quốc tế."""
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    }

    cells = []

    # SLIDE 1: COVER SLIDE
    cells.append(nbformat.v4.new_markdown_cell(r"""# MIDTERM EXPERIMENTAL REPORT - DIGITAL SIGNAL PROCESSING
## ALGORITHM 3 (TT3): VOICE ACTIVITY DETECTION (VAD) BASED ON GAUSSIAN SHORT-TIME ENERGY

- **DSP Configuration**: Frame size 25 ms, hop size 10 ms, virtual silence bridging < 200 ms.
- **Methodology**: Estimate Gaussian parameters ($\mu, \sigma$), solve Bayes decision boundary for $T_{opt}$.
- **Evaluation**: Quantitative metrics (MAE, RMSE ms) and waveform/STE figures across 04 test recordings."""))

    # CELL 1: SETUP & IMPORTS
    cells.append(nbformat.v4.new_markdown_cell("""### 1. Environment Configuration and Library Imports
Import standard Python modules (`wave`, `math`, `os`, `glob`), `numpy`, `matplotlib`, and `IPython.display.Audio`."""))

    cells.append(nbformat.v4.new_code_cell("""import os
import glob
import math
import wave
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Configure matplotlib formatting for academic presentation
plt.rcParams.update({
    "font.size": 10,
    "figure.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": ":"
})
print("Environment configured successfully.")"""))

    # CELL 2: PATHS
    cells.append(nbformat.v4.new_markdown_cell("""### 2. Project Directory Path Setup
Locate repository directories for `TinHieuHuanLuyen` (training data), `TinHieuKiemThu` (testing data), and output plots."""))

    cells.append(nbformat.v4.new_code_cell("""def find_project_root() -> Path:
    curr = Path.cwd()
    for candidate in [curr, *curr.parents]:
        if (candidate / "TinHieuHuanLuyen").exists() and (candidate / "TinHieuKiemThu").exists():
            return candidate
    raise FileNotFoundError("Could not find project root directory.")

PROJECT_ROOT = find_project_root()
TRAIN_DIR = PROJECT_ROOT / "TinHieuHuanLuyen"
TEST_DIR = PROJECT_ROOT / "TinHieuKiemThu"
OUTPUT_DIR = PROJECT_ROOT / "src" / "TT3" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Training Directory : {TRAIN_DIR}")
print(f"Testing Directory  : {TEST_DIR}")
print(f"Output Directory   : {OUTPUT_DIR}")"""))

    # CELL 3: READ WAV
    cells.append(nbformat.v4.new_markdown_cell("""### 3. Read 16-bit PCM Audio from WAV Files
Decode raw binary audio sample stream into 16-bit signed integers and normalize amplitudes to float values in `[-1.0, 1.0]`."""))

    cells.append(nbformat.v4.new_code_cell("""def read_wav(file_path: str) -> Tuple[np.ndarray, int]:
    \"\"\"Read standard 16-bit PCM WAV audio using Python's built-in wave module.\"\"\"
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM audio (sample_width=2) is supported, got: {sample_width}")

    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    if n_channels > 1:
        audio_int16 = audio_int16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    signal = audio_int16.astype(np.float64) / 32768.0
    return signal, sample_rate

print("Function read_wav defined.")"""))

    # CELL 4: READ LAB
    cells.append(nbformat.v4.new_markdown_cell("""### 4. Parse Praat .lab Files and Extract Speech Ground Truth
Extract continuous speech boundary `[t_start, t_end]` based on phoneme labels 'v' (voiced) and 'uv' (unvoiced)."""))

    cells.append(nbformat.v4.new_code_cell("""def read_lab(lab_path: str) -> List[Tuple[float, float, str]]:
    \"\"\"Parse Praat .lab segmentation file, filtering out metadata lines.\"\"\"
    segments: List[Tuple[float, float, str]] = []
    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] not in ("F0mean", "F0std"):
                try:
                    segments.append((float(parts[0]), float(parts[1]), parts[2].lower()))
                except ValueError:
                    continue
    return segments


def get_speech_groundtruth(lab_segments: List[Tuple[float, float, str]]) -> Tuple[float, float]:
    \"\"\"Identify earliest start and latest end timestamps for speech ('v' or 'uv').\"\"\"
    speech_segments = [seg for seg in lab_segments if seg[2] in ("v", "uv")]
    if not speech_segments:
        return 0.0, 0.0
    t_start = min(seg[0] for seg in speech_segments)
    t_end = max(seg[1] for seg in speech_segments)
    return t_start, t_end

print("Functions read_lab and get_speech_groundtruth defined.")"""))

    # CELL 5: FRAMING
    cells.append(nbformat.v4.new_markdown_cell("""### 5. Manual Signal Framing Implementation
Partition the 1D continuous audio array into overlapping frames with duration 25 ms and hop size 10 ms (60% overlap)."""))

    cells.append(nbformat.v4.new_code_cell("""def frame_signal(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Partition 1D audio signal into overlapping 2D frames using basic NumPy indexing.\"\"\"
    frame_len = int(round(frame_size_ms * sample_rate / 1000.0))
    hop_len = int(round(hop_size_ms * sample_rate / 1000.0))

    num_samples = len(signal)
    if num_samples < frame_len:
        pad_len = frame_len - num_samples
        padded = np.zeros(frame_len, dtype=signal.dtype)
        padded[:num_samples] = signal
        signal = padded
        num_samples = len(signal)

    num_frames = 1 + int(np.floor((num_samples - frame_len) / hop_len))
    frames = np.zeros((num_frames, frame_len), dtype=np.float64)
    frame_times = np.zeros(num_frames, dtype=np.float64)

    for m in range(num_frames):
        start_idx = m * hop_len
        end_idx = start_idx + frame_len
        frames[m, :] = signal[start_idx:end_idx]
        frame_times[m] = (start_idx + end_idx) / (2.0 * sample_rate)

    return frames, frame_times

print("Function frame_signal defined.")"""))

    # CELL 6: STE
    cells.append(nbformat.v4.new_markdown_cell(r"""### 6. Compute Normalized Short-Time Energy (STE)
Calculate frame energy $STE[m] = \sum x^2[m \cdot H + n]$ and normalize into `[0.0, 1.0]`."""))

    cells.append(nbformat.v4.new_code_cell("""def compute_ste(frames: np.ndarray) -> np.ndarray:
    \"\"\"Compute Short-Time Energy (STE) via sum of squared amplitudes across frames.\"\"\"
    return np.sum(np.square(frames), axis=1)


def normalize_ste(ste: np.ndarray) -> np.ndarray:
    \"\"\"Normalize short-time energy array to [0.0, 1.0].\"\"\"
    max_energy = np.max(ste)
    if max_energy <= 1e-12:
        return np.zeros_like(ste)
    return ste / max_energy


def extract_ste_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Integrated feature extraction pipeline: Framing -> Raw STE -> Normalized STE.\"\"\"
    frames, frame_times = frame_signal(signal, sample_rate, frame_size_ms, hop_size_ms)
    ste_raw = compute_ste(frames)
    ste_norm = normalize_ste(ste_raw)
    return ste_norm, frame_times

print("Functions for STE feature extraction defined.")"""))

    # SLIDE 2: TRAINING DATA SURVEY
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 2: TRAINING DATA SURVEY (SILENCE VS. SPEECH FRAMES)
Survey all speech and silence frames across 4 training audio recordings using Praat ground-truth labels."""))

    cells.append(nbformat.v4.new_code_cell("""def survey_training_data(training_dir: str, frame_size_ms: float = 25.0, hop_size_ms: float = 10.0):
    \"\"\"Survey all frames across training recordings, segregating into silence and speech sets.\"\"\"
    wav_files = sorted(glob.glob(os.path.join(training_dir, "*.wav")))
    all_silence_ste: List[float] = []
    all_speech_ste: List[float] = []
    per_file_stats: Dict[str, Dict[str, float]] = {}

    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        signal, sample_rate = read_wav(wav_path)
        ste_norm, frame_times = extract_ste_features(signal, sample_rate, frame_size_ms, hop_size_ms)
        lab_segments = read_lab(lab_path)

        file_sil_ste = []
        file_sp_ste = []

        for ste_val, t_center in zip(ste_norm, frame_times):
            label = "sil"
            for seg in lab_segments:
                if seg[0] <= t_center <= seg[1]:
                    label = seg[2]
                    break
            if label in ("v", "uv"):
                file_sp_ste.append(ste_val)
                all_speech_ste.append(ste_val)
            else:
                file_sil_ste.append(ste_val)
                all_silence_ste.append(ste_val)

        per_file_stats[fname] = {
            "sil_count": len(file_sil_ste),
            "sil_mean": float(np.mean(file_sil_ste)) if file_sil_ste else 0.0,
            "sil_std": float(np.std(file_sil_ste)) if file_sil_ste else 0.0,
            "sp_count": len(file_sp_ste),
            "sp_mean": float(np.mean(file_sp_ste)) if file_sp_ste else 0.0,
            "sp_std": float(np.std(file_sp_ste)) if file_sp_ste else 0.0
        }

    return np.array(all_silence_ste), np.array(all_speech_ste), per_file_stats

silence_ste, speech_ste, train_stats = survey_training_data(str(TRAIN_DIR))
print(f"Total Silence Frames Surveyed: {len(silence_ste)}")
print(f"Total Speech Frames Surveyed : {len(speech_ste)}")"""))

    # SLIDE 3: GAUSSIAN PARAMETERS ESTIMATION
    cells.append(nbformat.v4.new_markdown_cell(r"""## SLIDE 3: GAUSSIAN PARAMETERS ESTIMATION (meanSil, stdSil, meanSp, stdSp)
Estimate empirical Gaussian mean $\mu$ and standard deviation $\sigma$ for silence and speech classes."""))

    cells.append(nbformat.v4.new_code_cell("""def estimate_gaussian_parameters(silence_ste: np.ndarray, speech_ste: np.ndarray):
    \"\"\"Estimate empirical Gaussian mean and standard deviation for silence and speech.\"\"\"
    mu_sil = float(np.mean(silence_ste))
    sigma_sil = float(np.std(silence_ste))
    mu_sp = float(np.mean(speech_ste))
    sigma_sp = float(np.std(speech_ste))
    return mu_sil, sigma_sil, mu_sp, sigma_sp

mu_sil, sigma_sil, mu_sp, sigma_sp = estimate_gaussian_parameters(silence_ste, speech_ste)

print("=" * 60)
print(f"{'Parameter':<28} | {'Silence (Sil)':<15} | {'Speech (Sp)':<15}")
print("-" * 60)
print(f"{'Mean (mu)':<28} | {mu_sil:<15.6f} | {mu_sp:<15.6f}")
print(f"{'Standard Deviation (sigma)':<28} | {sigma_sil:<15.6f} | {sigma_sp:<15.6f}")
print("=" * 60)"""))

    # SLIDE 4: OPTIMAL BAYES DECISION THRESHOLD DERIVATION
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 4: OPTIMAL BAYES DECISION THRESHOLD DERIVATION
Solve quadratic intersection equation $p(x|\text{silence}) = p(x|\text{speech})$ for optimal decision threshold $T_{opt}$."""))

    cells.append(nbformat.v4.new_code_cell("""def solve_bayes_threshold(mu_sil: float, sigma_sil: float, mu_sp: float, sigma_sp: float) -> float:
    \"\"\"Solve quadratic intersection equation for equal-prior Gaussian Bayes decision boundary.\"\"\"
    A = (1.0 / (sigma_sil ** 2)) - (1.0 / (sigma_sp ** 2))
    B = -2.0 * ((mu_sil / (sigma_sil ** 2)) - (mu_sp / (sigma_sp ** 2)))
    C = ((mu_sil ** 2) / (sigma_sil ** 2)) - ((mu_sp ** 2) / (sigma_sp ** 2)) - 2.0 * math.log(sigma_sp / sigma_sil)

    discriminant = B ** 2 - 4.0 * A * C
    if discriminant < 0:
        raise ValueError("Discriminant is negative; no real roots exist.")

    sqrt_disc = math.sqrt(discriminant)
    root1 = (-B - sqrt_disc) / (2.0 * A)
    root2 = (-B + sqrt_disc) / (2.0 * A)

    candidates = [r for r in (root1, root2) if mu_sil < r < mu_sp]
    if candidates:
        return candidates[0]
    return min((root1, root2), key=lambda r: abs(r - (mu_sil + mu_sp) / 2.0))

T_opt = solve_bayes_threshold(mu_sil, sigma_sil, mu_sp, sigma_sp)
print(f"[*] UNIFIED OPTIMAL BAYES THRESHOLD: T_opt = {T_opt:.6f}")"""))

    # SLIDE 5: GAUSSIAN DISTRIBUTIONS AND BAYES THRESHOLD VISUALIZATION
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 5: GAUSSIAN DISTRIBUTIONS AND BAYES THRESHOLD VISUALIZATION
Overlay Gaussian probability density functions (PDFs) and display the Bayes decision threshold $T_{opt}$."""))

    cells.append(nbformat.v4.new_code_cell("""x_range = np.linspace(0.0, 0.015, 2000)
pdf_sil = (1.0 / (sigma_sil * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu_sil) / sigma_sil) ** 2)
pdf_sp = (1.0 / (sigma_sp * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu_sp) / sigma_sp) ** 2)

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
ax.plot(x_range, pdf_sil, label=f"Silence N(μ={mu_sil:.5f}, σ={sigma_sil:.5f})", color="#e53e3e", lw=2.0)
ax.plot(x_range, pdf_sp, label=f"Speech N(μ={mu_sp:.5f}, σ={sigma_sp:.5f})", color="#3182ce", lw=2.0)
ax.axvline(T_opt, color="#2b6cb0", linestyle="--", lw=2.2, label=f"Bayes Threshold T_opt = {T_opt:.6f}")
ax.set_title("Gaussian Probability Density Functions and Optimal Bayes Threshold (TT3)", fontweight="bold")
ax.set_xlabel("Normalized Short-Time Energy (STE)")
ax.set_ylabel("Probability Density (PDF)")
ax.set_xlim(0, 0.012)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()"""))

    # CELL 7a: apply_threshold
    cells.append(nbformat.v4.new_markdown_cell("""### 7. Binary Classification Against Threshold
Classify each frame into speech (1) if STE exceeds $T_{opt}$, otherwise silence (0)."""))

    cells.append(nbformat.v4.new_code_cell("""def apply_threshold(ste_norm: np.ndarray, threshold: float) -> np.ndarray:
    \"\"\"Binary frame classification: 1 if STE >= threshold, else 0.\"\"\"
    return (ste_norm >= threshold).astype(np.int32)"""))

    # CELL 7b: remove_short_silences
    cells.append(nbformat.v4.new_markdown_cell("""### 8. Virtual Silence Bridging (< 200 ms)
Bridge internal pause gaps shorter than 200 ms (20 frames at 10 ms hop) into continuous speech."""))

    cells.append(nbformat.v4.new_code_cell("""def remove_short_silences(
    frame_decisions: np.ndarray,
    hop_size_ms: float = 10.0,
    min_silence_ms: float = 200.0
) -> np.ndarray:
    \"\"\"Eliminate virtual silence gaps < 200 ms between speech segments.\"\"\"
    min_frames = int(round(min_silence_ms / hop_size_ms))
    smoothed = frame_decisions.copy()
    speech_indices = np.where(smoothed == 1)[0]
    if len(speech_indices) == 0:
        return smoothed

    first_speech, last_speech = speech_indices[0], speech_indices[-1]
    idx = first_speech
    while idx <= last_speech:
        if smoothed[idx] == 0:
            zero_start = idx
            while idx <= last_speech and smoothed[idx] == 0:
                idx += 1
            if (idx - zero_start) < min_frames:
                smoothed[zero_start:idx] = 1
        else:
            idx += 1
    return smoothed"""))

    # CELL 7c: extract_speech_boundaries
    cells.append(nbformat.v4.new_markdown_cell("""### 9. Speech Boundary Timestamp Extraction
Extract speech start $T_{start}$ and end $T_{end}$ timestamps from the smoothed frame sequence."""))

    cells.append(nbformat.v4.new_code_cell("""def extract_speech_boundaries(
    frame_decisions: np.ndarray,
    hop_size_ms: float = 10.0,
    frame_size_ms: float = 25.0
) -> Tuple[float, float]:
    \"\"\"Extract physical start T_start and end T_end boundaries in seconds.\"\"\"
    speech_indices = np.where(frame_decisions == 1)[0]
    if len(speech_indices) == 0:
        return 0.0, 0.0

    start_idx, end_idx = speech_indices[0], speech_indices[-1]
    hop_sec = hop_size_ms / 1000.0
    frame_sec = frame_size_ms / 1000.0

    t_start = round(float(start_idx * hop_sec), 4)
    t_end = round(float(end_idx * hop_sec + frame_sec), 4)
    return t_start, t_end"""))

    # CELL 7d: predict_vad
    cells.append(nbformat.v4.new_markdown_cell("""### 10. Integrated VAD Prediction Pipeline
End-to-end VAD detection function mapping raw audio waveform to timestamps $[T_{start}, T_{end}]$."""))

    cells.append(nbformat.v4.new_code_cell("""def predict_vad(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
):
    \"\"\"End-to-end VAD detection pipeline from raw waveform to [t_start, t_end].\"\"\"
    ste_norm, frame_times = extract_ste_features(
        signal=signal,
        sample_rate=sample_rate,
        frame_size_ms=frame_size_ms,
        hop_size_ms=hop_size_ms
    )
    raw_decisions = apply_threshold(ste_norm, threshold)
    smoothed = remove_short_silences(raw_decisions, hop_size_ms=hop_size_ms)
    t_start, t_end = extract_speech_boundaries(
        smoothed,
        hop_size_ms=hop_size_ms,
        frame_size_ms=frame_size_ms
    )
    return t_start, t_end, ste_norm, frame_times, smoothed

print("predict_vad pipeline defined.")"""))

    # CELL 8: METRICS
    cells.append(nbformat.v4.new_markdown_cell("""### 11. Quantitative Evaluation Metrics (MAE and RMSE in Milliseconds)
Compute Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) against Praat ground-truth boundaries."""))

    cells.append(nbformat.v4.new_code_cell("""def calculate_mae_rmse(pred_bounds: Tuple[float, float], gt_bounds: Tuple[float, float]) -> Tuple[float, float]:
    \"\"\"Compute MAE and RMSE in milliseconds between predicted and ground-truth bounds.\"\"\"
    pred_start, pred_end = pred_bounds
    gt_start, gt_end = gt_bounds

    diff_start_ms = abs(pred_start - gt_start) * 1000.0
    diff_end_ms = abs(pred_end - gt_end) * 1000.0

    mae_ms = (diff_start_ms + diff_end_ms) / 2.0
    rmse_ms = math.sqrt((diff_start_ms ** 2 + diff_end_ms ** 2) / 2.0)
    return mae_ms, rmse_ms"""))

    # SLIDE 6: TEST BENCHMARK
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 6: QUANTITATIVE BENCHMARK ON 04 TEST RECORDINGS
Execute VAD with 25 ms frame size, 10 ms hop size, and threshold $T_{opt} = 0.002878$ on the 04 test audio files."""))

    cells.append(nbformat.v4.new_code_cell("""test_files = sorted(glob.glob(os.path.join(str(TEST_DIR), "*.wav")))
eval_results = []
test_data_dict = {}

for wav_path in test_files:
    fname = os.path.basename(wav_path)
    lab_path = os.path.splitext(wav_path)[0] + ".lab"

    signal, sample_rate = read_wav(wav_path)
    lab_segments = read_lab(lab_path)
    gt_bounds = get_speech_groundtruth(lab_segments)

    pred_start, pred_end, ste_norm, frame_times, smoothed = predict_vad(
        signal=signal,
        sample_rate=sample_rate,
        threshold=T_opt,
        frame_size_ms=25.0,
        hop_size_ms=10.0
    )
    pred_bounds = (pred_start, pred_end)

    mae_ms, rmse_ms = calculate_mae_rmse(pred_bounds, gt_bounds)
    d_start_ms = abs(pred_start - gt_bounds[0]) * 1000.0
    d_end_ms = abs(pred_end - gt_bounds[1]) * 1000.0

    eval_results.append({
        "file": fname,
        "gt": gt_bounds,
        "pred": pred_bounds,
        "d_start": d_start_ms,
        "d_end": d_end_ms,
        "mae": mae_ms,
        "rmse": rmse_ms
    })
    test_data_dict[fname] = (fname, signal, sample_rate, ste_norm, frame_times, gt_bounds, pred_bounds, mae_ms, rmse_ms)

print("=" * 96)
print(f"{'Test Recording':<16} | {'GT [s]':<14} | {'Predicted [s]':<14} | {'ΔStart (ms)':<11} | {'ΔEnd (ms)':<10} | {'MAE (ms)':<9} | {'RMSE (ms)':<9}")
print("-" * 96)
for r in eval_results:
    gt_str = f"[{r['gt'][0]:.2f}, {r['gt'][1]:.2f}]"
    pred_str = f"[{r['pred'][0]:.2f}, {r['pred'][1]:.2f}]"
    print(f"{r['file']:<16} | {gt_str:<14} | {pred_str:<14} | {r['d_start']:<11.1f} | {r['d_end']:<10.1f} | {r['mae']:<9.1f} | {r['rmse']:<9.1f}")
print("=" * 96)

avg_mae = np.mean([r["mae"] for r in eval_results])
avg_rmse = np.mean([r["rmse"] for r in eval_results])
print(f"[*] AGGREGATE BENCHMARK AVERAGE MAE  : {avg_mae:.2f} ms")
print(f"[*] AGGREGATE BENCHMARK AVERAGE RMSE : {avg_rmse:.2f} ms")
print("=" * 96)"""))

    # SLIDE 7: Helper function for individual test display
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 7: TEST VISUALIZATIONS AND AUDIO PLAYERS
Plot waveform and STE overlay with predicted boundaries and display 2 audio players per test recording."""))

    cells.append(nbformat.v4.new_code_cell("""def plot_and_play_test_recording(item):
    \"\"\"Plot waveform, normalized STE and render interactive audio players.\"\"\"
    fname, sig, sr, ste, ft, gt, pred, mae, rmse = item
    time_wave = np.arange(len(sig)) / float(sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, dpi=120)

    # Subplot 1: Audio Waveform
    ax1.plot(time_wave, sig, color="#4a5568", lw=0.7, alpha=0.85, label="Audio Waveform")
    ax1.axvline(gt[0], color="red", linestyle="-", lw=2.0, label=f"Ground Truth: [{gt[0]:.2f}s, {gt[1]:.2f}s]")
    ax1.axvline(gt[1], color="red", linestyle="-", lw=2.0)
    ax1.axvline(pred[0], color="blue", linestyle="--", lw=2.0, label=f"TT3 Predicted: [{pred[0]:.2f}s, {pred[1]:.2f}s]")
    ax1.axvline(pred[1], color="blue", linestyle="--", lw=2.0)
    ax1.set_title(f"VAD Segmentation Result: {fname} | MAE = {mae:.1f} ms | RMSE = {rmse:.1f} ms", fontweight="bold")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")

    # Subplot 2: Normalized STE Curve
    ax2.plot(ft, ste, color="#2b6cb0", lw=1.2, label="Normalized STE")
    ax2.axhline(T_opt, color="#dd6b20", linestyle=":", lw=2.0, label=f"Bayes Threshold T_opt = {T_opt:.6f}")
    ax2.axvline(gt[0], color="red", linestyle="-", lw=1.8)
    ax2.axvline(gt[1], color="red", linestyle="-", lw=1.8)
    ax2.axvline(pred[0], color="blue", linestyle="--", lw=1.8)
    ax2.axvline(pred[1], color="blue", linestyle="--", lw=1.8)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Normalized STE")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{Path(fname).stem}.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure: {fig_path}")
    plt.show()

    # Load and display local audio player widget
    test_wav_path = TEST_DIR / fname
    print(f"🎵 Original Audio ({fname}):")
    if test_wav_path.exists():
        display(Audio(filename=str(test_wav_path)))
    else:
        display(Audio(data=sig, rate=sr))

    # Display audio player for predicted speech segment
    st_idx = int(pred[0] * sr)
    en_idx = min(len(sig), int(pred[1] * sr))
    if en_idx > st_idx:
        print(f"🔊 Predicted Speech Segment TT3 [{pred[0]:.2f}s - {pred[1]:.2f}s]:")
        display(Audio(data=sig[st_idx:en_idx], rate=sr))"""))

    # Test File 1: phone_F2
    cells.append(nbformat.v4.new_markdown_cell("""### 12. Test Recording 1: `phone_F2.wav`
Telephone channel, female speaker. Waveform, STE overlay, and audio players."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_recording(test_data_dict["phone_F2.wav"])"""))

    # Test File 2: phone_M2
    cells.append(nbformat.v4.new_markdown_cell("""### 13. Test Recording 2: `phone_M2.wav`
Telephone channel, male speaker. Waveform, STE overlay, and audio players."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_recording(test_data_dict["phone_M2.wav"])"""))

    # Test File 3: studio_F2
    cells.append(nbformat.v4.new_markdown_cell("""### 14. Test Recording 3: `studio_F2.wav`
High SNR Studio environment, female speaker. Waveform, STE overlay, and audio players."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_recording(test_data_dict["studio_F2.wav"])"""))

    # Test File 4: studio_M2
    cells.append(nbformat.v4.new_markdown_cell("""### 15. Test Recording 4: `studio_M2.wav`
High SNR Studio environment, male speaker. Waveform, STE overlay, and audio players."""))
    cells.append(nbformat.v4.new_code_cell("""plot_and_play_test_recording(test_data_dict["studio_M2.wav"])"""))

    # SLIDE 8: DISCUSSION & CONCLUSION
    cells.append(nbformat.v4.new_markdown_cell("""## SLIDE 8: EXPERIMENTAL SUMMARY AND CONCLUSION

| Test Recording | Environment | SNR | Ground Truth [s] | Predicted [s] | ΔStart (ms) | ΔEnd (ms) | MAE (ms) | RMSE (ms) |
|---|---|---|---|---|---|---|---|---|
| `phone_F2.wav` | Telephone | Low | [1.02, 4.04] | [1.03, 4.09] | +10.0 | +45.0 | 27.5 | 32.6 |
| `phone_M2.wav` | Telephone | Low | [0.53, 2.52] | [0.54, 2.52] | +10.0 | 0.0 | 5.0 | 7.1 |
| `studio_F2.wav` | Studio | High | [0.77, 2.37] | [0.78, 2.37] | +10.0 | 0.0 | 5.0 | 7.1 |
| `studio_M2.wav` | Studio | High | [0.45, 1.93] | [0.46, 1.95] | +10.0 | +15.0 | 12.5 | 12.7 |
| **Average** | - | - | - | - | **+10.0** | **+15.0** | **12.5** | **14.9** |

- **Boundary Accuracy**: Average MAE reaches $12.5\\text{ ms}$ (~1 frame hop of 10 ms); start boundary error $\\Delta_{Start} = 10\\text{ ms}$ is consistent across all files.
- **Robustness**: Bayes threshold $T_{opt} \\approx 0.002878$ automatically establishes a strong noise margin across both telephone and studio channels."""))

    nb.cells = cells
    return nb


def run_build():
    tt3_dir = PROJECT_ROOT / "src" / "TT3"
    notebook_dir = tt3_dir / "notebook"
    notebook_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Generating Vietnamese notebook (TT3_Gaussian_VAD.ipynb)...")
    nb_vi = build_tt3_notebook_vi()
    vi_path = notebook_dir / "TT3_Gaussian_VAD.ipynb"
    with open(vi_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_vi, f)

    print("[*] Executing Vietnamese notebook...")
    client_vi = NotebookClient(nb_vi, timeout=60, kernel_name="python3")
    client_vi.execute()
    with open(vi_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_vi, f)
    print(f"[V] Vietnamese notebook saved and executed at: {vi_path} with {len(nb_vi.cells)} cells!")

    print("[*] Generating English notebook (TT3_Gaussian_VAD_EN.ipynb)...")
    nb_en = build_tt3_notebook_en()
    en_path = notebook_dir / "TT3_Gaussian_VAD_EN.ipynb"
    with open(en_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_en, f)

    print("[*] Executing English notebook...")
    client_en = NotebookClient(nb_en, timeout=60, kernel_name="python3")
    client_en.execute()
    with open(en_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_en, f)
    print(f"[V] English notebook saved and executed at: {en_path} with {len(nb_en.cells)} cells!")

    # Alias TT3.ipynb in notebook directory
    shutil.copy2(vi_path, notebook_dir / "TT3.ipynb")

    # Copy baseline notebooks 1.ipynb and 2.ipynb into notebook directory for completeness
    for base_nb in ["1.ipynb", "2.ipynb"]:
        src_nb = tt3_dir / base_nb
        if src_nb.exists():
            shutil.copy2(src_nb, notebook_dir / base_nb)

    print(f"[V] Synchronized all notebooks in: {notebook_dir}")


if __name__ == "__main__":
    run_build()
