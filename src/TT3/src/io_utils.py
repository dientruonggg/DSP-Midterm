"""
Module: TT3.io_utils
Mô tả: Chứa các hàm đọc dữ liệu âm thanh định dạng WAV (chuẩn PCM 16-bit)
       và đọc file nhãn phân đoạn tiếng nói (.lab) từ phần mềm Praat.
Quy định: Không sử dụng thư viện chuyên dụng xử lý tín hiệu âm thanh (như librosa, scipy.signal).
"""

import wave
import numpy as np
from typing import List, Tuple


def read_wav(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Đọc file âm thanh WAV chuẩn 16-bit PCM sử dụng thư viện wave tích hợp sẵn của Python.

    Tham số đầu vào:
        file_path (str): Đường dẫn đến file âm thanh .wav.

    Giá trị trả lại:
        Tuple[np.ndarray, int]:
            - signal (np.ndarray): Mảng 1 chiều chứa các mẫu tín hiệu đã chuẩn hóa về [-1.0, 1.0].
            - sample_rate (int): Tần số lấy mẫu của file âm thanh (Hz).
    """
    # Mở file WAV bằng thư viện wave chuẩn của Python để lấy thông số kỹ thuật
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Kiểm tra kích thước mẫu để đảm bảo đúng định dạng PCM 16-bit (2 bytes/mẫu)
    if sample_width != 2:
        raise ValueError(f"Chỉ hỗ trợ file WAV 16-bit PCM (sample_width=2), hiện tại là: {sample_width}")

    # Chuyển đổi chuỗi byte nhị phân thành mảng số nguyên int16 của numpy
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

    # Nếu âm thanh là stereo (2 kênh), tính trung bình các kênh để chuyển về mono
    if n_channels > 1:
        audio_int16 = audio_int16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    # Chuẩn hóa biên độ tín hiệu về dải số thực [-1.0, 1.0] bằng cách chia cho 32768.0
    signal = audio_int16.astype(np.float64) / 32768.0

    return signal, sample_rate


def read_lab(lab_path: str) -> List[Tuple[float, float, str]]:
    """
    Đọc và phân tích file nhãn mốc thời gian phân đoạn (.lab) của Praat.
    Cấu trúc từng dòng dữ liệu: <thời_điểm_bắt_đầu> <thời_điểm_kết_thúc> <nhãn>

    Tham số đầu vào:
        lab_path (str): Đường dẫn đến file nhãn .lab.

    Giá trị trả lại:
        List[Tuple[float, float, str]]:
            Danh sách các phân đoạn gồm (t_start, t_end, label).
    """
    segments: List[Tuple[float, float, str]] = []

    # Mở file nhãn và duyệt qua từng dòng dữ liệu
    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # Bỏ qua các dòng trống không chứa thông tin
            if not stripped:
                continue

            parts = stripped.split()
            # Bỏ qua các dòng metadata như F0mean, F0std ở cuối file
            if len(parts) >= 3 and parts[0] not in ("F0mean", "F0std"):
                try:
                    t_start = float(parts[0])
                    t_end = float(parts[1])
                    label = parts[2].lower()
                    segments.append((t_start, t_end, label))
                except ValueError:
                    # Bỏ qua dòng nếu xảy ra lỗi chuyển đổi kiểu dữ liệu sang float
                    continue

    return segments


def get_speech_groundtruth(lab_segments: List[Tuple[float, float, str]]) -> Tuple[float, float]:
    """
    Xác định mốc thời gian bắt đầu và kết thúc toàn bộ câu nói chuẩn (Ground Truth) từ file .lab.
    Tiếng nói bao gồm các đoạn có nhãn 'v' (voiced - hữu thanh) hoặc 'uv' (unvoiced - vô thanh).
    Khoảng lặng được gán nhãn 'sil' (silence).

    Tham số đầu vào:
        lab_segments (List[Tuple[float, float, str]]): Danh sách các phân đoạn đọc từ file .lab.

    Giá trị trả lại:
        Tuple[float, float]:
            - t_start (float): Mốc bắt đầu tiếng nói chuẩn (giây).
            - t_end (float): Mốc kết thúc tiếng nói chuẩn (giây).
    """
    # Lọc các phân đoạn là tiếng nói (hữu thanh 'v' hoặc vô thanh 'uv')
    speech_segments = [seg for seg in lab_segments if seg[2] in ("v", "uv")]

    # Nếu file âm thanh hoàn toàn im lặng, trả về mốc mặc định (0.0, 0.0)
    if not speech_segments:
        return 0.0, 0.0

    # Mốc bắt đầu là điểm bắt đầu của phân đoạn tiếng nói xuất hiện sớm nhất
    t_start = min(seg[0] for seg in speech_segments)

    # Mốc kết thúc là điểm kết thúc của phân đoạn tiếng nói xuất hiện muộn nhất
    t_end = max(seg[1] for seg in speech_segments)

    return t_start, t_end
