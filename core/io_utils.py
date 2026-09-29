"""
Module: core.io_utils
Description: Handles loading audio WAV files (using standard library wave)
             and parsing Praat ground-truth .lab label files.
Strict Requirement: No specialized audio toolboxes (librosa, scipy.signal) allowed.
"""

import wave
import numpy as np
from typing import List, Tuple


def read_wav(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Read a 16-bit PCM mono WAV file using Python's standard wave library.

    Parameters:
        file_path (str): Path to the .wav audio file.

    Returns:
        Tuple[np.ndarray, int]:
            - signal (np.ndarray): 1D float64 array of samples normalized to [-1.0, 1.0].
            - sample_rate (int): Sampling frequency (e.g., 16000 or 44100 Hz).
    """
    # [DONE]: Implement WAV file reading using standard wave library, parse byte stream to float array
    # print("[CALL] read_wav")

    # Mở file WAV bằng thư viện wave chuẩn của Python
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Kiểm tra định dạng 16-bit PCM chuẩn
    if sample_width != 2:
        raise ValueError(f"Chỉ hỗ trợ âm thanh 16-bit PCM (sample_width=2), file có sample_width={sample_width}")

    # Chuyển đổi byte stream thành mảng numpy số nguyên int16
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

    # Nếu file stereo (2 kênh), lấy trung bình cộng để đưa về mono
    if n_channels > 1:
        audio_int16 = audio_int16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    # Chuẩn hóa biên độ tín hiệu về dải giá trị float [-1.0, 1.0] bằng cách chia cho 32768.0
    signal = audio_int16.astype(np.float64) / 32768.0

    return signal, sample_rate


def read_lab(lab_path: str) -> List[Tuple[float, float, str]]:
    """
    Read and parse a Praat .lab label file.
    Format per line: <start_time> <end_time> <label>
    Note: Ignores metadata lines like F0mean and F0std at the end of the file.

    Parameters:
        lab_path (str): Path to the .lab label file.

    Returns:
        List[Tuple[float, float, str]]:
            List of parsed segments, each containing (t_start, t_end, label).
    """
    # [DONE]: Parse lines of Praat .lab file into (t_start, t_end, label), filtering out F0 lines
    # print("[CALL] read_lab")

    segments: List[Tuple[float, float, str]] = []

    # Đọc từng dòng trong file nhãn .lab
    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            # Bỏ qua các dòng chú thích hoặc siêu dữ liệu F0mean, F0std
            if len(parts) >= 3 and parts[0] not in ("F0mean", "F0std"):
                try:
                    t_start = float(parts[0])
                    t_end = float(parts[1])
                    label = parts[2].lower()
                    segments.append((t_start, t_end, label))
                except ValueError:
                    # Bỏ qua dòng nếu không ép kiểu float được
                    continue

    return segments


def get_speech_groundtruth(lab_segments: List[Tuple[float, float, str]]) -> Tuple[float, float]:
    """
    Extract the global continuous speech boundary [T_start, T_end] from label segments.
    Speech is defined as any segment with label 'v' (voiced) or 'uv' (unvoiced).
    Silence is labeled as 'sil'.

    Parameters:
        lab_segments (List[Tuple[float, float, str]]): List of parsed segments.

    Returns:
        Tuple[float, float]: (T_start, T_end) ground-truth speech boundaries in seconds.
    """
    # [DONE]: Locate earliest speech start time and latest speech end time across all 'v'/'uv' segments
    # print("[CALL] get_speech_groundtruth")

    speech_segments = [seg for seg in lab_segments if seg[2] in ("v", "uv")]

    # Nếu không có đoạn tiếng nói nào trong file, mặc định trả về (0.0, 0.0)
    if not speech_segments:
        return 0.0, 0.0

    # Thời điểm bắt đầu câu nói là mốc đầu tiên có 'v' hoặc 'uv'
    t_start = min(seg[0] for seg in speech_segments)
    # Thời điểm kết thúc câu nói là mốc cuối cùng có 'v' hoặc 'uv'
    t_end = max(seg[1] for seg in speech_segments)

    return t_start, t_end
