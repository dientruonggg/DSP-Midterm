"""
Module: core.features
Description: Implements speech signal framing and Short-Time Energy (STE) feature extraction.
             Strictly implemented using basic mathematical operations without external signal toolboxes.
"""

import numpy as np
from typing import Tuple


def frame_signal(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 20.0,
    hop_size_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide a 1D audio signal into overlapping frames.

    Parameters:
        signal (np.ndarray): 1D float array of audio samples.
        sample_rate (int): Sampling frequency (Hz).
        frame_size_ms (float): Frame duration in milliseconds (default: 20 ms).
        hop_size_ms (float): Hop/shift duration in milliseconds (default: 10 ms).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - frames (np.ndarray): 2D array of shape (num_frames, frame_len_samples).
            - frame_times (np.ndarray): 1D array of center timestamps in seconds for each frame.
    """
    # [DONE]: Implement signal framing with 20ms frame length and 10ms hop size using basic indexing
    # print("[CALL] frame_signal")

    # Tính độ dài khung N và bước nhảy H theo số mẫu
    frame_len = int(round(frame_size_ms * sample_rate / 1000.0))
    hop_len = int(round(hop_size_ms * sample_rate / 1000.0))

    num_samples = len(signal)
    if num_samples < frame_len:
        # Trường hợp tín hiệu quá ngắn, đệm thêm số 0
        pad_len = frame_len - num_samples
        signal = np.pad(signal, (0, pad_len), mode="constant")
        num_samples = len(signal)

    # Tính tổng số khung có thể trích xuất
    num_frames = 1 + int(np.floor((num_samples - frame_len) / hop_len))

    # Khởi tạo mảng chứa các khung tín hiệu
    frames = np.zeros((num_frames, frame_len), dtype=np.float64)
    frame_times = np.zeros(num_frames, dtype=np.float64)

    # Cắt từng khung tín hiệu và tính mốc thời gian trung tâm của khung
    for m in range(num_frames):
        start_idx = m * hop_len
        end_idx = start_idx + frame_len
        frames[m, :] = signal[start_idx:end_idx]
        # Thời điểm đại diện cho khung m là điểm giữa của khung
        frame_times[m] = (start_idx + end_idx) / (2.0 * sample_rate)

    return frames, frame_times


def compute_ste(frames: np.ndarray) -> np.ndarray:
    """
    Compute Short-Time Energy (STE) for each frame.
    Formula: STE[m] = sum_{n=0}^{N-1} x^2[m * H + n]

    Parameters:
        frames (np.ndarray): 2D array of shape (num_frames, frame_len).

    Returns:
        np.ndarray: 1D array of energy values for each frame.
    """
    # [DONE]: Calculate Short-Time Energy (STE) as the sum of squared samples per frame
    # print("[CALL] compute_ste")

    # Bình phương từng mẫu trong mỗi khung
    squared_frames = np.square(frames)

    # Tính tổng năng lượng trên mỗi dòng (tương ứng từng khung)
    ste = np.sum(squared_frames, axis=1)

    return ste


def normalize_ste(ste: np.ndarray) -> np.ndarray:
    """
    Normalize Short-Time Energy values to the range [0.0, 1.0].
    Formula: STE_norm[m] = STE[m] / max_k(STE[k])

    Parameters:
        ste (np.ndarray): 1D array of raw energy values.

    Returns:
        np.ndarray: 1D array of normalized energy values in [0.0, 1.0].
    """
    # [DONE]: Normalize STE by dividing by its maximum value across all frames
    # print("[CALL] normalize_ste")

    max_energy = np.max(ste)

    # Tránh chia cho 0 nếu tín hiệu hoàn toàn im lặng
    if max_energy <= 1e-12:
        return np.zeros_like(ste)

    ste_norm = ste / max_energy
    return ste_norm


def extract_ste_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 20.0,
    hop_size_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level helper to extract normalized STE and corresponding timestamps.

    Parameters:
        signal (np.ndarray): 1D float array of audio samples.
        sample_rate (int): Sampling frequency (Hz).
        frame_size_ms (float): Frame duration in milliseconds (default: 20 ms).
        hop_size_ms (float): Hop duration in milliseconds (default: 10 ms).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - ste_norm (np.ndarray): Normalized Short-Time Energy.
            - frame_times (np.ndarray): Center timestamp of each frame in seconds.
    """
    # [DONE]: Execute framing, calculate STE, and normalize energy in a single unified pipeline
    # print("[CALL] extract_ste_features")

    # Bước 1: Chia khung tín hiệu
    frames, frame_times = frame_signal(
        signal=signal,
        sample_rate=sample_rate,
        frame_size_ms=frame_size_ms,
        hop_size_ms=hop_size_ms
    )

    # Bước 2: Tính năng lượng ngắn hạn thô
    ste_raw = compute_ste(frames)

    # Bước 3: Chuẩn hóa năng lượng về khoảng [0, 1]
    ste_norm = normalize_ste(ste_raw)

    return ste_norm, frame_times
