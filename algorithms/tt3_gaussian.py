"""
Module: algorithms.tt3_gaussian
Author: Sinh viên 3 (Student 3 & Pipeline Manager)
Description: Thuật toán 3 (TT3) - Phân đoạn dựa trên Phân bố xác suất Gauss Bayes (Gaussian Bayes).
             Khảo sát toàn bộ các khung Silence và Speech trên 4 file huấn luyện,
             ước lượng các tham số kỳ vọng và độ lệch chuẩn (mu_sil, sigma_sil, mu_sp, sigma_sp),
             và giải phương trình phân bố Gauss xác suất Bayes p(x|sil) = p(x|sp)
             để xác định ngưỡng tối ưu lý thuyết T_Bayes ≈ 0.002864.
"""

import os
import glob
import math
import numpy as np
from typing import Tuple, List, Dict

from core.io_utils import read_wav, read_lab, get_speech_groundtruth
from core.features import extract_ste_features
from core.postprocess import apply_threshold, remove_short_silences_200ms, extract_speech_boundaries


def extract_speech_silence_ste_frames(
    training_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract normalized STE values for all frames across training files,
    partitioning them into silence frames and speech frames using ground-truth timestamps.

    Parameters:
        training_dir (str): Directory containing training WAV and LAB files.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - silence_ste (np.ndarray): 1D array of STE_norm values for silence frames.
            - speech_ste (np.ndarray): 1D array of STE_norm values for speech frames.
    """
    # [DONE]: Partition training frames into silence vs speech collections based on ground truth
    # print("[CALL] extract_speech_silence_ste_frames")

    wav_files = sorted(glob.glob(os.path.join(training_dir, "*.wav")))
    silence_list: List[float] = []
    speech_list: List[float] = []

    for wav_path in wav_files:
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        signal, fs = read_wav(wav_path)
        ste_norm, frame_times = extract_ste_features(signal, fs)
        lab_segments = read_lab(lab_path)
        t_start_gt, t_end_gt = get_speech_groundtruth(lab_segments)

        # Phân loại từng khung dựa trên mốc thời gian ground truth
        for t, energy in zip(frame_times, ste_norm):
            if t_start_gt <= t <= t_end_gt:
                speech_list.append(float(energy))
            else:
                silence_list.append(float(energy))

    return np.array(silence_list, dtype=np.float64), np.array(speech_list, dtype=np.float64)


def estimate_gaussian_parameters(
    silence_ste: np.ndarray,
    speech_ste: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Estimate Gaussian distribution parameters (mean and standard deviation)
    for silence and speech frames using basic numpy arithmetic.

    Parameters:
        silence_ste (np.ndarray): Array of silence STE values.
        speech_ste (np.ndarray): Array of speech STE values.

    Returns:
        Tuple[float, float, float, float]: (mu_sil, sigma_sil, mu_sp, sigma_sp)
    """
    # [DONE]: Compute mean and std for silence and speech distributions using basic numpy functions
    # print("[CALL] estimate_gaussian_parameters")

    # Tính kỳ vọng (mean) và độ lệch chuẩn (std) của khoảng lặng
    mu_sil = float(np.mean(silence_ste))
    sigma_sil = float(np.std(silence_ste))

    # Tính kỳ vọng (mean) và độ lệch chuẩn (std) của tiếng nói
    mu_sp = float(np.mean(speech_ste))
    sigma_sp = float(np.std(speech_ste))

    return mu_sil, sigma_sil, mu_sp, sigma_sp


def solve_bayes_decision_threshold(
    mu_sil: float,
    sigma_sil: float,
    mu_sp: float,
    sigma_sp: float
) -> float:
    """
    Solve the Bayes minimum error decision boundary equation p(x|sil) = p(x|sp).
    Assuming equal prior probabilities P(sil) = P(sp):
        (1 / (sqrt(2*pi)*sigma_sil)) * exp(-(x - mu_sil)^2 / (2*sigma_sil^2))
      = (1 / (sqrt(2*pi)*sigma_sp))  * exp(-(x - mu_sp)^2 / (2*sigma_sp^2))

    Takes logarithm and converts to quadratic equation:
        A * x^2 + B * x + C = 0

    Returns:
        float: Optimal Bayes threshold T_Bayes (approximately 0.002864).
    """
    # [DONE]: Solve quadratic equation for Bayes decision boundary where Gaussian PDFs intersect
    # print("[CALL] solve_bayes_decision_threshold")

    var_sil = sigma_sil ** 2
    var_sp = sigma_sp ** 2

    # Các hệ số của phương trình bậc hai A*x^2 + B*x + C = 0
    coef_a = (1.0 / var_sil) - (1.0 / var_sp)
    coef_b = -2.0 * ((mu_sil / var_sil) - (mu_sp / var_sp))
    coef_c = (mu_sil ** 2 / var_sil) - (mu_sp ** 2 / var_sp) + 2.0 * math.log(sigma_sil / sigma_sp)

    # Xử lý trường hợp cận biên: sigma_sil ≈ sigma_sp dẫn đến A ≈ 0 (tránh ZeroDivisionError)
    if abs(coef_a) < 1e-12:
        if abs(coef_b) > 1e-12:
            linear_root = -coef_c / coef_b
            if mu_sil <= linear_root <= mu_sp:
                return float(linear_root)
        return float((mu_sil + mu_sp) / 2.0)

    # Tính biệt thức Delta = B^2 - 4*A*C
    delta = coef_b ** 2 - 4.0 * coef_a * coef_c

    if delta < 0:
        # Dự phòng trường hợp delta âm: dùng điểm trung bình có trọng số Z-score
        return float((mu_sil * sigma_sp + mu_sp * sigma_sil) / (sigma_sil + sigma_sp))

    sqrt_delta = math.sqrt(delta)
    root1 = (-coef_b + sqrt_delta) / (2.0 * coef_a)
    root2 = (-coef_b - sqrt_delta) / (2.0 * coef_a)

    # Lấy nghiệm nằm giữa mu_sil và mu_sp
    valid_roots = [r for r in (root1, root2) if mu_sil <= r <= mu_sp]
    if valid_roots:
        return float(valid_roots[0])

    # Nếu cả 2 nghiệm ngoài khoảng, chọn nghiệm gần mu_sil nhất
    return float(min((root1, root2), key=lambda r: abs(r - mu_sil)))


def predict_vad_tt3(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float = 0.002864
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run Voice Activity Detection (VAD) using Gaussian Bayes decision threshold TT3.

    Parameters:
        signal (np.ndarray): 1D audio sample array.
        sample_rate (int): Sampling rate (Hz).
        threshold (float): Bayes threshold T_Bayes (default: 0.002864).

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]:
            - t_start (float): Speech start boundary in seconds.
            - t_end (float): Speech end boundary in seconds.
            - ste_norm (np.ndarray): Normalized STE curve.
            - frame_times (np.ndarray): Frame timestamps in seconds.
    """
    # [DONE]: Run full VAD pipeline with Gaussian Bayes threshold, return boundaries and STE curve
    # print("[CALL] predict_vad_tt3")

    # Bước 1: Trích xuất đặc trưng STE chuẩn hóa
    ste_norm, frame_times = extract_ste_features(signal, sample_rate)

    # Bước 2: Phân loại theo ngưỡng Gauss Bayes
    decisions = apply_threshold(ste_norm, threshold)

    # Bước 3: Hậu xử lý gộp khoảng lặng ảo < 200 ms
    smoothed = remove_short_silences_200ms(decisions)

    # Bước 4: Tìm biên tiếng nói [T_start, T_end]
    t_start, t_end = extract_speech_boundaries(smoothed, frame_times)

    return t_start, t_end, ste_norm, frame_times
