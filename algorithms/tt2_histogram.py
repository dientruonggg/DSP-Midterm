"""
Module: algorithms.tt2_histogram
Author: Sinh viên 2 (Student 2)
Description: Thuật toán 2 (TT2) - Ngưỡng tự thích nghi theo Histogram (Giannakopoulos 2014).
             Tính toán ngưỡng T riêng biệt cho từng tệp âm thanh dựa trên lược đồ tần suất
             của năng lượng ngắn hạn chuẩn hóa STE_norm, làm mịn bằng Moving Average
             và tìm hai cực đại địa phương M1 (Silence) và M2 (Speech).
Strict Requirement: CẤM dùng scipy.signal để làm mịn hoặc tìm peak; tự viết bằng toán cơ bản.
"""

import numpy as np
from typing import Tuple

from core.features import extract_ste_features
from core.postprocess import apply_threshold, remove_short_silences_200ms, extract_speech_boundaries


def compute_histogram_100bins(
    ste_norm: np.ndarray,
    num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a 100-bin histogram of normalized STE values within [0.0, 1.0].

    Parameters:
        ste_norm (np.ndarray): 1D array of normalized STE values.
        num_bins (int): Number of bins (default: 100).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - counts (np.ndarray): Frequency count for each bin.
            - bin_centers (np.ndarray): Center value of each bin in [0, 1].
    """
    # [DONE]: Construct 100-bin histogram for STE_norm values in range [0, 1] using basic array math
    # print("[CALL] compute_histogram_100bins")

    # Tạo 101 mốc biên chia đều đoạn [0.0, 1.0] thành 100 bins (tự viết, không dùng np.linspace)
    bin_edges = np.array([i / num_bins for i in range(num_bins + 1)], dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.float64)

    # Đếm số lượng mẫu STE_norm rơi vào từng khoảng bin (tự viết, không dùng np.digitize/np.clip)
    bin_width = 1.0 / num_bins
    for val in ste_norm:
        idx = int(val / bin_width)
        # Đảm bảo giá trị tại biên 1.0 nằm vào bin cuối cùng (index num_bins - 1)
        if idx < 0:
            idx = 0
        elif idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1.0

    # Tính tâm của từng bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return counts, bin_centers


def moving_average_smooth(
    histogram: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Smooth the histogram using a Moving Average filter with a 5-bin window.
    Strictly uses basic array summation without scipy.signal.

    Parameters:
        histogram (np.ndarray): 1D array of raw bin counts.
        window_size (int): Size of the moving average window (default: 5 bins).

    Returns:
        np.ndarray: Smoothed histogram array of same length.
    """
    # [DONE]: Smooth 1D array using basic moving average window of size 5 with edge padding
    # print("[CALL] moving_average_smooth")

    n = len(histogram)
    smoothed = np.zeros(n, dtype=np.float64)
    half_w = window_size // 2

    # Lặp qua từng bin và tính trung bình cộng trong cửa sổ lân cận [-half_w, +half_w]
    for i in range(n):
        start_idx = max(0, i - half_w)
        end_idx = min(n, i + half_w + 1)
        smoothed[i] = np.mean(histogram[start_idx:end_idx])

    return smoothed


def find_histogram_peaks(
    smoothed_hist: np.ndarray,
    bin_centers: np.ndarray
) -> Tuple[float, float]:
    """
    Locate two local maxima:
      - M1: Silence peak (low energy background noise).
      - M2: Speech peak (higher energy region).
    Supports detection of plateaus (flat-top maxima).

    Parameters:
        smoothed_hist (np.ndarray): 1D array of smoothed histogram counts.
        bin_centers (np.ndarray): Center values for each bin.

    Returns:
        Tuple[float, float]: (M1, M2) corresponding to the energy levels of silence and speech peaks.
    """
    # [DONE]: Find first local peak M1 (silence) and second significant local peak M2 (speech)
    # print("[CALL] find_histogram_peaks")

    n = len(smoothed_hist)
    peaks = []

    # Kiểm tra bin 0 có phải cực đại địa phương hay không
    # (Năng lượng khoảng lặng ~0.0003 nằm hết trong bin 0 và thường là đỉnh cao nhất)
    if n > 1 and smoothed_hist[0] >= smoothed_hist[1]:
        peaks.append(0)

    i = 1
    # Quét tìm các điểm cực đại địa phương, hỗ trợ vùng đỉnh bằng phẳng (plateau)
    while i < n - 1:
        if smoothed_hist[i] > smoothed_hist[i - 1]:
            j = i
            while j < n - 1 and smoothed_hist[j] == smoothed_hist[j + 1]:
                j += 1
            if j < n - 1 and smoothed_hist[j] > smoothed_hist[j + 1]:
                peaks.append((i + j) // 2)
                i = j + 1
            else:
                i = j + 1
        else:
            i += 1

    # Nếu không tìm thấy đủ 2 cực đại địa phương rõ ràng, dự phòng logic thích nghi
    if len(peaks) == 0:
        return 0.065, 0.115
    elif len(peaks) == 1:
        m1_idx = peaks[0]
        other_indices = [idx for idx in range(n) if abs(idx - m1_idx) > 2]
        m2_idx = max(other_indices, key=lambda idx: smoothed_hist[idx]) if other_indices else min(n - 1, m1_idx + 2)
    else:
        # 2 đỉnh cực đại địa phương đầu tiên theo đúng mô tả Giannakopoulos
        m1_idx = peaks[0]
        m2_idx = peaks[1]

    m1_val = float(bin_centers[m1_idx])
    m2_val = float(bin_centers[m2_idx])

    return m1_val, m2_val


def compute_adaptive_threshold_tt2(
    ste_norm: np.ndarray,
    weight_w: float = 5.0
) -> float:
    """
    Calculate the file-specific adaptive threshold T = (W * M1 + M2) / (W + 1).

    Parameters:
        ste_norm (np.ndarray): Normalized STE values.
        weight_w (float): Weight assigned to silence peak (default: 5.0).

    Returns:
        float: Computed adaptive threshold T.
    """
    # [DONE]: Extract histogram, smooth with MA filter, find M1 & M2, compute T = (W*M1 + M2)/(W+1)
    # print("[CALL] compute_adaptive_threshold_tt2")

    # Bước 1: Tính histogram 100 bins
    counts, bin_centers = compute_histogram_100bins(ste_norm, num_bins=100)

    # Bước 2: Làm mịn histogram bằng Moving Average 5 bins
    smoothed = moving_average_smooth(counts, window_size=5)

    # Bước 3: Tìm 2 đỉnh cực đại địa phương M1 và M2
    m1, m2 = find_histogram_peaks(smoothed, bin_centers)

    # Bước 4: Tính ngưỡng tự thích nghi theo công thức Giannakopoulos
    threshold = (weight_w * m1 + m2) / (weight_w + 1.0)

    return float(threshold)


def predict_vad_tt2(
    signal: np.ndarray,
    sample_rate: int,
    weight_w: float = 5.0
) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    """
    Run full VAD pipeline with Giannakopoulos adaptive histogram threshold.

    Parameters:
        signal (np.ndarray): 1D audio sample array.
        sample_rate (int): Sampling rate (Hz).
        weight_w (float): Weight W for formula (default: 5.0).

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray, float]:
            - t_start (float): Speech start boundary (s).
            - t_end (float): Speech end boundary (s).
            - ste_norm (np.ndarray): Normalized STE array.
            - frame_times (np.ndarray): Center timestamp of each frame (s).
            - computed_threshold (float): Adaptive threshold T computed for this file.
    """
    # [DONE]: Run end-to-end adaptive histogram VAD, return boundaries, features, and threshold
    # print("[CALL] predict_vad_tt2")

    # Trích xuất STE chuẩn hóa
    ste_norm, frame_times = extract_ste_features(signal, sample_rate)

    # Tính ngưỡng thích nghi riêng cho tệp âm thanh này
    threshold = compute_adaptive_threshold_tt2(ste_norm, weight_w=weight_w)

    # Áp dụng ngưỡng và nối khoảng lặng ảo < 200 ms
    decisions = apply_threshold(ste_norm, threshold)
    smoothed = remove_short_silences_200ms(decisions)

    # Xác định biên phân đoạn
    t_start, t_end = extract_speech_boundaries(smoothed, frame_times)

    return t_start, t_end, ste_norm, frame_times, threshold
