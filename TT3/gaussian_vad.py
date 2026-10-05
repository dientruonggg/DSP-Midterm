"""
Module: TT3.gaussian_vad
Mô tả: Thuật toán 3 (TT3) - Phân đoạn tiếng nói và khoảng lặng dựa trên mô hình phân bố Gauss của STE.
       Khảo sát toàn bộ các khung Silence và Speech trên dữ liệu huấn luyện,
       thống kê kỳ vọng và độ lệch chuẩn (meanSil, stdSil, meanSp, stdSp),
       và giải phương trình phân bố xác suất Gauss Bayes để xác định ngưỡng phân tách tối ưu.
Quy định: Tự code toàn bộ các hàm giải phương trình và xử lý logic,
          tuân thủ cấu trúc hàm rõ ràng, chú thích cho từng block 5-10 dòng code.
"""

import os
import glob
import math
import numpy as np
from typing import Tuple, List, Dict, Any

from TT3.io_utils import read_wav, read_lab, get_speech_groundtruth
from TT3.features import extract_ste_features


def survey_training_data(
    training_dir: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Khảo sát tất cả các khung tín hiệu trên các file huấn luyện (*.wav và *.lab tương ứng),
    phân chia năng lượng STE chuẩn hóa của từng khung thành 2 tập: silence frames và speech frames.

    Tham số đầu vào:
        training_dir (str): Đường dẫn đến thư mục chứa các file huấn luyện (TinHieuHuanLuyen).

    Giá trị trả lại:
        Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
            - all_silence_ste (np.ndarray): Mảng 1 chiều chứa STE chuẩn hóa của tất cả các khung khoảng lặng.
            - all_speech_ste (np.ndarray): Mảng 1 chiều chứa STE chuẩn hóa của tất cả các khung tiếng nói.
            - per_file_stats (Dict): Bảng thống kê chi tiết mean/std theo từng file huấn luyện đơn lẻ.
    """
    # Tìm kiếm toàn bộ các file âm thanh .wav trong thư mục huấn luyện
    wav_files = sorted(glob.glob(os.path.join(training_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"Không tìm thấy file WAV nào trong thư mục: {training_dir}")

    all_silence_ste_list: List[float] = []
    all_speech_ste_list: List[float] = []
    per_file_stats: Dict[str, Dict[str, float]] = {}

    # Duyệt qua từng file huấn luyện để trích xuất đặc trưng và đối chiếu nhãn chuẩn
    for wav_path in wav_files:
        file_name = os.path.basename(wav_path)
        lab_path = os.path.splitext(wav_path)[0] + ".lab"

        # Đọc tín hiệu âm thanh và nhãn phân đoạn chuẩn Praat
        signal, sample_rate = read_wav(wav_path)
        ste_norm, frame_times = extract_ste_features(signal, sample_rate)
        lab_segments = read_lab(lab_path)
        gt_start, gt_end = get_speech_groundtruth(lab_segments)

        # Phân loại từng khung dựa trên mốc thời gian trung tâm của khung so với biên ground truth
        file_silence: List[float] = []
        file_speech: List[float] = []
        for t, energy in zip(frame_times, ste_norm):
            if gt_start <= t <= gt_end:
                file_speech.append(float(energy))
            else:
                file_silence.append(float(energy))

        # Lưu trữ các giá trị năng lượng của file hiện tại vào danh sách tổng
        all_silence_ste_list.extend(file_silence)
        all_speech_ste_list.extend(file_speech)

        # Thống kê mean và std riêng lẻ cho file huấn luyện này
        per_file_stats[file_name] = {
            "mean_sil": float(np.mean(file_silence)),
            "std_sil": float(np.std(file_silence)),
            "mean_sp": float(np.mean(file_speech)),
            "std_sp": float(np.std(file_speech)),
            "num_sil_frames": len(file_silence),
            "num_sp_frames": len(file_speech),
        }

    return (
        np.array(all_silence_ste_list, dtype=np.float64),
        np.array(all_speech_ste_list, dtype=np.float64),
        per_file_stats
    )


def estimate_gaussian_parameters(
    silence_ste: np.ndarray,
    speech_ste: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Ước lượng các tham số kỳ vọng (mean) và độ lệch chuẩn (std) của phân bố Gauss
    cho hai lớp khoảng lặng (Silence) và tiếng nói (Speech).

    Tham số đầu vào:
        silence_ste (np.ndarray): Mảng chứa giá trị STE của các khung khoảng lặng.
        speech_ste (np.ndarray): Mảng chứa giá trị STE của các khung tiếng nói.

    Giá trị trả lại:
        Tuple[float, float, float, float]: (meanSil, stdSil, meanSp, stdSp)
    """
    # Tính kỳ vọng và độ lệch chuẩn của khoảng lặng bằng các hàm built-in của numpy
    mu_sil = float(np.mean(silence_ste))
    sigma_sil = float(np.std(silence_ste))

    # Tính kỳ vọng và độ lệch chuẩn của tiếng nói bằng các hàm built-in của numpy
    mu_sp = float(np.mean(speech_ste))
    sigma_sp = float(np.std(speech_ste))

    return mu_sil, sigma_sil, mu_sp, sigma_sp


def solve_bayes_threshold(
    mu_sil: float,
    sigma_sil: float,
    mu_sp: float,
    sigma_sp: float
) -> float:
    """
    Giải phương trình tìm ngưỡng tối ưu Bayes p(x|sil) = p(x|sp) giữa 2 phân bố Gauss:
        (1 / (sqrt(2*pi)*sigma_sil)) * exp(-(x - mu_sil)^2 / (2*sigma_sil^2))
      = (1 / (sqrt(2*pi)*sigma_sp))  * exp(-(x - mu_sp)^2 / (2*sigma_sp^2))

    Chuyển đổi sang phương trình bậc hai: A*x^2 + B*x + C = 0
    và chọn nghiệm nằm giữa mu_sil và mu_sp.

    Tham số đầu vào:
        mu_sil (float): Kỳ vọng STE của khoảng lặng.
        sigma_sil (float): Độ lệch chuẩn STE của khoảng lặng.
        mu_sp (float): Kỳ vọng STE của tiếng nói.
        sigma_sp (float): Độ lệch chuẩn STE của tiếng nói.

    Giá trị trả lại:
        float: Ngưỡng năng lượng tối ưu T_Bayes dùng chung cho các tín hiệu.
    """
    # Tính phương sai (variance) từ độ lệch chuẩn
    var_sil = sigma_sil ** 2
    var_sp = sigma_sp ** 2

    # Xác định các hệ số A, B, C của phương trình bậc hai sau khi lấy logarit tự nhiên 2 vế
    coef_a = (1.0 / var_sil) - (1.0 / var_sp)
    coef_b = -2.0 * ((mu_sil / var_sil) - (mu_sp / var_sp))
    coef_c = (mu_sil ** 2 / var_sil) - (mu_sp ** 2 / var_sp) + 2.0 * math.log(sigma_sil / sigma_sp)

    # Xử lý trường hợp phương sai xấp xỉ bằng nhau dẫn đến phương trình suy biến thành bậc nhất
    if abs(coef_a) < 1e-12:
        if abs(coef_b) > 1e-12:
            linear_root = -coef_c / coef_b
            if mu_sil <= linear_root <= mu_sp:
                return float(linear_root)
        return float((mu_sil + mu_sp) / 2.0)

    # Tính biệt thức Delta = B^2 - 4*A*C của phương trình bậc hai
    delta = coef_b ** 2 - 4.0 * coef_a * coef_c

    # Nếu delta âm (không có giao điểm thực), sử dụng điểm chia tỷ lệ theo độ lệch chuẩn
    if delta < 0:
        return float((mu_sil * sigma_sp + mu_sp * sigma_sil) / (sigma_sil + sigma_sp))

    # Tính 2 nghiệm thực của phương trình
    sqrt_delta = math.sqrt(delta)
    root1 = (-coef_b + sqrt_delta) / (2.0 * coef_a)
    root2 = (-coef_b - sqrt_delta) / (2.0 * coef_a)

    # Lựa chọn nghiệm phân tách hợp lệ nằm giữa khoảng kỳ vọng của khoảng lặng và tiếng nói
    valid_roots = [r for r in (root1, root2) if mu_sil <= r <= mu_sp]
    if valid_roots:
        return float(valid_roots[0])

    # Nếu cả 2 nghiệm đều nằm ngoài, lấy nghiệm dương gần mu_sil nhất
    positive_roots = [r for r in (root1, root2) if r > 0]
    if positive_roots:
        return float(min(positive_roots, key=lambda r: abs(r - mu_sil)))

    return float((mu_sil + mu_sp) / 2.0)


def apply_threshold(ste_norm: np.ndarray, threshold: float) -> np.ndarray:
    """
    Phân loại nhị phân từng khung: 1 (Speech) nếu STE >= ngưỡng, ngược lại 0 (Silence).

    Tham số đầu vào:
        ste_norm (np.ndarray): Mảng 1 chiều chứa năng lượng STE chuẩn hóa [0, 1].
        threshold (float): Ngưỡng năng lượng phân loại.

    Giá trị trả lại:
        np.ndarray: Mảng nhị phân các quyết định cho từng khung (1=tiếng nói, 0=khoảng lặng).
    """
    # So sánh trực tiếp mảng năng lượng với giá trị ngưỡng phân loại
    decisions = (ste_norm >= threshold).astype(np.int32)
    return decisions


def remove_short_silences(
    frame_decisions: np.ndarray,
    hop_size_ms: float = 10.0,
    min_silence_ms: float = 200.0
) -> np.ndarray:
    """
    Loại bỏ các khoảng lặng 'ảo' ngắn hơn 200 ms nằm xen giữa các đoạn tiếng nói.
    Quy định bài tập: Độ dài tối thiểu của một khoảng lặng thực sự là 200 ms.
    Nếu khoảng lặng giữa các đoạn tiếng nói < 200 ms (20 khung), gộp thành tiếng nói.

    Tham số đầu vào:
        frame_decisions (np.ndarray): Mảng nhị phân quyết định phân khung (0 hoặc 1).
        hop_size_ms (float): Độ dời khung tính theo ms (mặc định 10 ms).
        min_silence_ms (float): Độ dài khoảng lặng tối thiểu cần giữ lại (mặc định 200 ms).

    Giá trị trả lại:
        np.ndarray: Mảng nhị phân sau khi đã loại bỏ và làm mượt các khoảng lặng ảo.
    """
    # Số lượng khung tối thiểu tương ứng với 200 ms (200 ms / 10 ms = 20 khung)
    min_frames = int(round(min_silence_ms / hop_size_ms))
    smoothed = frame_decisions.copy()

    # Tìm chỉ số của khung tiếng nói đầu tiên và cuối cùng
    speech_indices = np.where(smoothed == 1)[0]
    if len(speech_indices) == 0:
        return smoothed

    first_speech = speech_indices[0]
    last_speech = speech_indices[-1]

    # Duyệt qua các đoạn khoảng lặng (dãy số 0) nằm giữa first_speech và last_speech
    idx = first_speech
    while idx <= last_speech:
        if smoothed[idx] == 0:
            zero_start = idx
            while idx <= last_speech and smoothed[idx] == 0:
                idx += 1
            zero_len = idx - zero_start

            # Nếu độ dài chuỗi 0 nhỏ hơn 20 khung (< 200 ms), lấp đầy chuỗi này thành tiếng nói (1)
            if zero_len < min_frames:
                smoothed[zero_start:idx] = 1
        else:
            idx += 1

    return smoothed


def extract_speech_boundaries(
    frame_decisions: np.ndarray,
    hop_size_ms: float = 10.0,
    frame_size_ms: float = 20.0
) -> Tuple[float, float]:
    """
    Xác định mốc thời gian bắt đầu (T_start) và kết thúc (T_end) của câu nói từ chuỗi quyết định khung.
    T_start tính từ đầu khung tiếng nói đầu tiên (start_idx * hop_size).
    T_end tính đến hết khung tiếng nói cuối cùng (end_idx * hop_size + frame_size).

    Tham số đầu vào:
        frame_decisions (np.ndarray): Mảng nhị phân quyết định đã qua xử lý hậu kỳ.
        hop_size_ms (float): Bước nhảy khung (10 ms).
        frame_size_ms (float): Độ rộng khung (20 ms).

    Giá trị trả lại:
        Tuple[float, float]: (T_start, T_end) tính bằng giây.
    """
    # Lấy toàn bộ chỉ số của các khung được phân loại là tiếng nói (giá trị 1)
    speech_indices = np.where(frame_decisions == 1)[0]

    # Trường hợp không phát hiện bất kỳ khung tiếng nói nào
    if len(speech_indices) == 0:
        return 0.0, 0.0

    start_idx = speech_indices[0]
    end_idx = speech_indices[-1]

    hop_sec = hop_size_ms / 1000.0
    frame_sec = frame_size_ms / 1000.0

    # Điểm bắt đầu tính từ rìa trước của khung tiếng nói đầu tiên
    t_start = round(float(start_idx * hop_sec), 4)

    # Điểm kết thúc tính đến hết rìa sau của khung tiếng nói cuối cùng
    t_end = round(float(end_idx * hop_sec + frame_sec), 4)

    return t_start, t_end


def predict_vad(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hàm thực thi phân đoạn tiếng nói / khoảng lặng hoàn chỉnh cho một file âm thanh.

    Tham số đầu vào:
        signal (np.ndarray): Mảng tín hiệu âm thanh đầu vào.
        sample_rate (int): Tần số lấy mẫu (Hz).
        threshold (float): Ngưỡng năng lượng STE phân loại.

    Giá trị trả lại:
        Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
            - t_start (float): Mốc bắt đầu tiếng nói tìm được (giây).
            - t_end (float): Mốc kết thúc tiếng nói tìm được (giây).
            - ste_norm (np.ndarray): Năng lượng STE chuẩn hóa.
            - frame_times (np.ndarray): Mốc thời gian trung tâm của các khung.
            - smoothed_decisions (np.ndarray): Mảng nhị phân sau hậu xử lý 200ms.
    """
    # Bước 1: Trích xuất đặc trưng STE chuẩn hóa và thời gian khung
    ste_norm, frame_times = extract_ste_features(signal, sample_rate)

    # Bước 2: So sánh năng lượng với ngưỡng Bayes T_opt
    raw_decisions = apply_threshold(ste_norm, threshold)

    # Bước 3: Hậu xử lý gộp các khoảng lặng ngắn dưới 200 ms
    smoothed_decisions = remove_short_silences(raw_decisions)

    # Bước 4: Trích xuất biên thời gian bắt đầu và kết thúc của câu nói
    t_start, t_end = extract_speech_boundaries(smoothed_decisions)

    return t_start, t_end, ste_norm, frame_times, smoothed_decisions
