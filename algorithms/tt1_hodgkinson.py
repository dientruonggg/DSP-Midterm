"""
Module: algorithms.tt1_hodgkinson
Author: Sinh viên 1 (Student 1)
Description: Thuật toán 1 (TT1) - Tối ưu hóa ngưỡng toàn cục cố định theo Hodgkinson (2012).
             Tìm kiếm giá trị ngưỡng T_opt duy nhất trên tập huấn luyện (TinHieuHuanLuyen)
             bằng phương pháp quét lưới (Grid Search) hoặc tìm kiếm tam phân (Ternary Search)
             để cực tiểu hóa sai số MAE.
"""

import os
import glob
import numpy as np
from typing import Tuple, List, Dict, Any

from core.io_utils import read_wav, read_lab, get_speech_groundtruth
from core.features import extract_ste_features
from core.postprocess import apply_threshold, remove_short_silences_200ms, extract_speech_boundaries
from core.metrics import calculate_mae_rmse


def hodgkinson_cost_function(
    threshold: float,
    train_data: List[Dict[str, Any]]
) -> float:
    """
    Compute average MAE loss over training files for a given threshold T.
    Formula: E(T) = (1/K) * sum_{k=1}^K MAE_k(T)

    Parameters:
        threshold (float): Candidate normalized STE threshold.
        train_data (List[Dict[str, Any]]): Pre-extracted features containing 'ste_norm',
                                           'frame_times', and 'gt_boundaries'.

    Returns:
        float: Average MAE in milliseconds across training audio files.
    """
    # [DONE]: Evaluate candidate threshold T across training files and return average MAE
    # print("[CALL] hodgkinson_cost_function")

    total_mae = 0.0
    for item in train_data:
        ste_norm = item["ste_norm"]
        frame_times = item["frame_times"]
        gt_bounds = item["gt_boundaries"]

        # Bước 1: Áp dụng ngưỡng năng lượng
        decisions = apply_threshold(ste_norm, threshold)

        # Bước 2: Hậu xử lý gộp khoảng lặng ảo < 200ms (20 khung)
        smoothed = remove_short_silences_200ms(decisions)

        # Bước 3: Tìm biên thời gian [T_start, T_end]
        pred_bounds = extract_speech_boundaries(smoothed, frame_times)

        # Bước 4: Tính sai số MAE so với nhãn chuẩn
        mae_ms, _ = calculate_mae_rmse(pred_bounds, gt_bounds)
        total_mae += mae_ms

    return total_mae / len(train_data)


def signed_duration_error(
    threshold: float,
    train_data: List[Dict[str, Any]]
) -> float:
    """
    Compute average signed duration error: (predicted_duration - gt_duration).
    Đại lượng này giảm đơn điệu khi T tăng (ngưỡng cao hơn -> phát hiện ít hơn -> thời lượng ngắn hơn),
    nên phù hợp để áp dụng tìm kiếm nhị phân tìm nghiệm = 0.

    Parameters:
        threshold (float): Candidate normalized STE threshold.
        train_data (List[Dict[str, Any]]): Pre-extracted features.

    Returns:
        float: Average signed duration error in seconds.
    """
    total = 0.0
    for item in train_data:
        ste_norm = item["ste_norm"]
        frame_times = item["frame_times"]
        gt_bounds = item["gt_boundaries"]

        decisions = apply_threshold(ste_norm, threshold)
        smoothed = remove_short_silences_200ms(decisions)
        pred_bounds = extract_speech_boundaries(smoothed, frame_times)

        gt_dur = gt_bounds[1] - gt_bounds[0]
        pred_dur = pred_bounds[1] - pred_bounds[0]
        total += (pred_dur - gt_dur)

    return total / len(train_data)


def binary_search_optimal_threshold_tt1(
    training_dir: str,
    search_range: Tuple[float, float] = (0.0001, 0.05),
    max_iter: int = 50
) -> float:
    """
    Search for the optimal global threshold T_opt using true binary search.
    Tìm kiếm nhị phân trên đại lượng sai số thời lượng có dấu (signed duration error),
    đại lượng này đơn điệu giảm theo T nên binary search hợp lệ.
    Theo tài liệu tham khảo [1] Hodgkinson (2012):
    'Energy-based Speech/Silence discrimination (thuật toán dùng tìm kiếm nhị phân)'.

    Parameters:
        training_dir (str): Directory path containing training .wav and .lab files.
        search_range (Tuple[float, float]): (min_threshold, max_threshold) interval.
        max_iter (int): Maximum search iterations (default: 50).

    Returns:
        float: Best threshold T_opt found via binary search.
    """
    # [DONE]: Implement binary search to find optimal threshold T_opt
    # print("[CALL] binary_search_optimal_threshold_tt1")

    wav_files = sorted(glob.glob(os.path.join(training_dir, "*.wav")))
    if not wav_files:
        return 0.0025

    train_data = []
    for wav_path in wav_files:
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        signal, fs = read_wav(wav_path)
        ste_norm, frame_times = extract_ste_features(signal, fs)
        lab_segments = read_lab(lab_path)
        gt_bounds = get_speech_groundtruth(lab_segments)
        train_data.append({
            "wav_path": wav_path,
            "ste_norm": ste_norm,
            "frame_times": frame_times,
            "gt_boundaries": gt_bounds
        })

    # Tìm kiếm nhị phân: sai số thời lượng có dấu giảm đơn điệu theo T
    # Khi T nhỏ -> phát hiện nhiều -> duration lớn -> error > 0
    # Khi T lớn -> phát hiện ít -> duration nhỏ -> error < 0
    # Tìm T sao cho signed_duration_error ≈ 0
    low, high = search_range
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        err = signed_duration_error(mid, train_data)
        if err > 0:
            low = mid
        else:
            high = mid

    return float((low + high) / 2.0)


def train_optimal_threshold_tt1(
    training_dir: str,
    search_range: Tuple[float, float] = (0.0001, 0.05),
    num_steps: int = 500,
    method: str = "grid"
) -> float:
    """
    Search for the optimal global threshold T_opt that minimizes MAE over the 4 training files.
    Supports both Grid Search and Binary Search (Hodgkinson 2012).
    Optimal result from benchmark: T_opt ≈ 0.0025.

    Parameters:
        training_dir (str): Directory path containing training .wav and .lab files.
        search_range (Tuple[float, float]): (min_threshold, max_threshold) search interval.
        num_steps (int): Number of grid search evaluation points.
        method (str): 'grid' for Grid Search, 'binary' for Binary Search.

    Returns:
        float: Best threshold T_opt found.
    """
    # [DONE]: Perform Grid Search or Ternary Search over search_range to find T_opt minimizing MAE
    # print("[CALL] train_optimal_threshold_tt1")

    if method == "binary":
        return binary_search_optimal_threshold_tt1(training_dir, search_range)

    # Thu thập tất cả các file WAV trong thư mục huấn luyện
    wav_files = sorted(glob.glob(os.path.join(training_dir, "*.wav")))
    if not wav_files:
        print("Cảnh báo: Không tìm thấy file wav huấn luyện, trả về T_opt mặc định = 0.0025")
        return 0.0025

    # Đọc trước tín hiệu và trích xuất đặc trưng STE cho toàn bộ tập huấn luyện để tối ưu tốc độ
    train_data = []
    for wav_path in wav_files:
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        signal, fs = read_wav(wav_path)
        ste_norm, frame_times = extract_ste_features(signal, fs)
        lab_segments = read_lab(lab_path)
        gt_bounds = get_speech_groundtruth(lab_segments)

        train_data.append({
            "wav_path": wav_path,
            "ste_norm": ste_norm,
            "frame_times": frame_times,
            "gt_boundaries": gt_bounds
        })

    # Tiến hành quét lưới (Grid Search) trong khoảng quy định
    # Tạo danh sách ứng viên thủ công (không dùng np.linspace)
    step_size = (search_range[1] - search_range[0]) / (num_steps - 1) if num_steps > 1 else 0
    candidates = [search_range[0] + i * step_size for i in range(num_steps)]
    best_t = 0.0025
    best_mae = float("inf")

    for t_cand in candidates:
        mae = hodgkinson_cost_function(t_cand, train_data)
        if mae < best_mae:
            best_mae = mae
            best_t = float(t_cand)

    return best_t


def predict_vad_tt1(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float = 0.0025
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run Voice Activity Detection (VAD) using fixed global threshold TT1.

    Parameters:
        signal (np.ndarray): 1D audio sample array.
        sample_rate (int): Sampling rate (Hz).
        threshold (float): Fixed threshold T_opt (default: 0.0025).

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]:
            - t_start (float): Speech start boundary in seconds.
            - t_end (float): Speech end boundary in seconds.
            - ste_norm (np.ndarray): Normalized STE curve.
            - frame_times (np.ndarray): Frame timestamps in seconds.
    """
    # [DONE]: Extract STE_norm, apply fixed threshold T_opt, bridge <200ms silences, return boundaries
    # print("[CALL] predict_vad_tt1")

    # Bước 1: Trích xuất đặc trưng STE chuẩn hóa và mốc thời gian
    ste_norm, frame_times = extract_ste_features(signal, sample_rate)

    # Bước 2: So sánh năng lượng với ngưỡng cố định
    decisions = apply_threshold(ste_norm, threshold)

    # Bước 3: Nối khoảng lặng ngắn dưới 200 ms (20 khung)
    smoothed = remove_short_silences_200ms(decisions)

    # Bước 4: Trích xuất điểm bắt đầu và kết thúc câu nói
    t_start, t_end = extract_speech_boundaries(smoothed, frame_times)

    return t_start, t_end, ste_norm, frame_times
