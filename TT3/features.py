"""
Module: TT3.features
Mô tả: Tự cài đặt các hàm xử lý tín hiệu cơ sở: phân khung tín hiệu (framing)
       và tính Năng lượng ngắn hạn (Short-Time Energy - STE) chuẩn hóa.
Quy định: Tự code bằng toán học cơ bản và built-in của Numpy (sum, max, min, square,...),
          tuyệt đối không sử dụng toolbox xử lý tín hiệu của bên thứ ba.
"""

import numpy as np
from typing import Tuple


def frame_signal(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phân chia tín hiệu âm thanh 1 chiều thành các khung (frames) có độ chồng lấp.

    Tham số đầu vào:
        signal (np.ndarray): Mảng 1 chiều chứa các mẫu biên độ âm thanh.
        sample_rate (int): Tần số lấy mẫu (Hz).
        frame_size_ms (float): Độ dài một khung tính bằng miligiây (mặc định: 25 ms).
        hop_size_ms (float): Bước nhảy giữa hai khung tính bằng miligiây (mặc định: 10 ms).

    Giá trị trả lại:
        Tuple[np.ndarray, np.ndarray]:
            - frames (np.ndarray): Mảng 2 chiều kích thước (num_frames, frame_len_samples).
            - frame_times (np.ndarray): Mảng 1 chiều chứa mốc thời gian trung tâm (giây) của từng khung.
    """
    # Tính độ dài khung N và độ dời khung H quy đổi ra số lượng mẫu tín hiệu
    frame_len = int(round(frame_size_ms * sample_rate / 1000.0))
    hop_len = int(round(hop_size_ms * sample_rate / 1000.0))

    # Xử lý trường hợp độ dài tín hiệu ngắn hơn một khung đơn lẻ
    num_samples = len(signal)
    if num_samples < frame_len:
        pad_len = frame_len - num_samples
        padded = np.zeros(frame_len, dtype=signal.dtype)
        padded[:num_samples] = signal
        signal = padded
        num_samples = len(signal)

    # Tính tổng số khung có thể tạo được từ tín hiệu
    num_frames = 1 + int(np.floor((num_samples - frame_len) / hop_len))

    # Khởi tạo mảng bộ nhớ chứa các khung tín hiệu và vector mốc thời gian
    frames = np.zeros((num_frames, frame_len), dtype=np.float64)
    frame_times = np.zeros(num_frames, dtype=np.float64)

    # Cắt từng khung dữ liệu dựa trên chỉ số mẫu và xác định mốc thời gian tâm khung
    for m in range(num_frames):
        start_idx = m * hop_len
        end_idx = start_idx + frame_len
        frames[m, :] = signal[start_idx:end_idx]
        # Thời điểm tâm khung được tính bằng trung bình cộng giữa mẫu đầu và mẫu cuối
        frame_times[m] = (start_idx + end_idx) / (2.0 * sample_rate)

    return frames, frame_times


def compute_ste(frames: np.ndarray) -> np.ndarray:
    """
    Tính Năng lượng ngắn hạn (Short-Time Energy - STE) cho từng khung tín hiệu.
    Công thức: STE[m] = sum_{n=0}^{N-1} x^2[m * H + n]

    Tham số đầu vào:
        frames (np.ndarray): Mảng 2 chiều kích thước (num_frames, frame_len).

    Giá trị trả lại:
        np.ndarray: Mảng 1 chiều chứa giá trị năng lượng STE của từng khung.
    """
    # Bình phương biên độ của tất cả các mẫu trong từng khung tín hiệu
    squared_frames = np.square(frames)

    # Tính tổng năng lượng theo từng hàng (tương ứng với từng khung đơn lẻ)
    ste = np.sum(squared_frames, axis=1)

    return ste


def normalize_ste(ste: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa giá trị năng lượng ngắn hạn STE về đoạn dải giá trị [0.0, 1.0].
    Công thức: STE_norm[m] = STE[m] / max(STE)

    Tham số đầu vào:
        ste (np.ndarray): Mảng 1 chiều chứa năng lượng STE thô.

    Giá trị trả lại:
        np.ndarray: Mảng 1 chiều chứa năng lượng STE đã chuẩn hóa trong khoảng [0.0, 1.0].
    """
    # Tìm giá trị năng lượng lớn nhất trong toàn bộ các khung tín hiệu
    max_energy = np.max(ste)

    # Kiểm tra tránh chia cho 0 trong trường hợp tín hiệu hoàn toàn là khoảng lặng
    if max_energy <= 1e-12:
        return np.zeros_like(ste)

    # Chia toàn bộ các giá trị năng lượng cho giá trị cực đại để chuẩn hóa
    ste_norm = ste / max_energy

    return ste_norm


def extract_ste_features(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hàm tích hợp thực hiện toàn bộ chu trình trích xuất đặc trưng STE chuẩn hóa.

    Tham số đầu vào:
        signal (np.ndarray): Mảng 1 chiều chứa tín hiệu âm thanh đầu vào.
        sample_rate (int): Tần số lấy mẫu (Hz).
        frame_size_ms (float): Độ dài khung tính theo miligiây (25 ms).
        hop_size_ms (float): Bước nhảy khung tính theo miligiây (10 ms).

    Giá trị trả lại:
        Tuple[np.ndarray, np.ndarray]:
            - ste_norm (np.ndarray): Vector năng lượng ngắn hạn STE chuẩn hóa [0, 1].
            - frame_times (np.ndarray): Mốc thời gian trung tâm của từng khung (giây).
    """
    # Bước 1: Chia tín hiệu thành các khung có độ chồng lấp (khung 25ms, bước nhảy 10ms)
    frames, frame_times = frame_signal(
        signal=signal,
        sample_rate=sample_rate,
        frame_size_ms=frame_size_ms,
        hop_size_ms=hop_size_ms
    )

    # Bước 2: Tính tổng năng lượng bình phương cho từng khung tín hiệu
    ste_raw = compute_ste(frames)

    # Bước 3: Chuẩn hóa vector năng lượng STE về khoảng cực đại bằng 1.0
    ste_norm = normalize_ste(ste_raw)

    return ste_norm, frame_times
