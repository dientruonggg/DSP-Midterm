"""
Module: core.postprocess
Description: Implements thresholding, morphological smoothing, and elimination of
             virtual silences shorter than 200ms (< 20 frames) between speech chunks.
"""

import numpy as np
from typing import Tuple, List


def apply_threshold(ste_norm: np.ndarray, threshold: float) -> np.ndarray:
    """
    Classify each frame into speech (1) or silence (0) based on an energy threshold.

    Parameters:
        ste_norm (np.ndarray): Normalized STE values in [0.0, 1.0].
        threshold (float): Energy threshold.

    Returns:
        np.ndarray: Binary array (1 for speech, 0 for silence).
    """
    # [DONE]: Classify frames into binary states: 1 if ste_norm >= threshold else 0
    # print("[CALL] apply_threshold")

    # So sánh từng giá trị năng lượng với ngưỡng quy định
    decisions = (ste_norm >= threshold).astype(np.int32)
    return decisions


def remove_short_silences_200ms(
    frame_decisions: np.ndarray,
    hop_size_ms: float = 10.0,
    min_silence_ms: float = 200.0
) -> np.ndarray:
    """
    Bridge 'virtual' silence intervals that are shorter than 200 ms between speech segments.
    According to the project requirement:
    Any gap of silence < 200 ms (equivalent to 20 frames at 10 ms hop size)
    surrounded by speech is converted into speech.

    Parameters:
        frame_decisions (np.ndarray): 1D binary array of frame labels (1=speech, 0=silence).
        hop_size_ms (float): Frame hop duration in ms (default: 10 ms).
        min_silence_ms (float): Minimum true silence duration in ms (default: 200 ms).

    Returns:
        np.ndarray: Smoothed binary array with short silences bridged.
    """
    # [DONE]: Scan for runs of 0s bounded by 1s; if length < min_silence_frames (20), convert to 1s
    # print("[CALL] remove_short_silences_200ms")

    min_frames = int(round(min_silence_ms / hop_size_ms))
    smoothed = frame_decisions.copy()
    n = len(smoothed)

    # Tìm vị trí khung tiếng nói đầu tiên và cuối cùng
    speech_indices = np.where(smoothed == 1)[0]
    if len(speech_indices) == 0:
        return smoothed

    first_speech = speech_indices[0]
    last_speech = speech_indices[-1]

    # Quét các chuỗi số 0 (silence) nằm giữa first_speech và last_speech
    idx = first_speech
    while idx <= last_speech:
        if smoothed[idx] == 0:
            zero_start = idx
            while idx <= last_speech and smoothed[idx] == 0:
                idx += 1
            zero_len = idx - zero_start

            # Nếu khoảng lặng ngắn hơn ngưỡng tối thiểu (20 frames = 200 ms), nối liền thành tiếng nói
            if zero_len < min_frames:
                smoothed[zero_start:idx] = 1
        else:
            idx += 1

    return smoothed


def extract_speech_boundaries(
    frame_decisions: np.ndarray,
    frame_times: np.ndarray,
    hop_size_ms: float = 10.0,
    frame_size_ms: float = 20.0
) -> Tuple[float, float]:
    """
    Determine the global start time T_start and end time T_end of the speech segment.
    Speech segment starts at the beginning of the first speech frame (start_idx * hop_size)
    and ends at the end of the last speech frame (end_idx * hop_size + frame_size).

    Parameters:
        frame_decisions (np.ndarray): 1D binary array of smoothed frame decisions.
        frame_times (np.ndarray): Center timestamp of each frame in seconds.
        hop_size_ms (float): Frame hop duration in ms (default: 10.0 ms).
        frame_size_ms (float): Frame duration in ms (default: 20.0 ms).

    Returns:
        Tuple[float, float]: (T_start, T_end) in seconds.
    """
    # [DONE]: Locate index of first speech frame (T_start) and last speech frame (T_end)
    # print("[CALL] extract_speech_boundaries")

    speech_indices = np.where(frame_decisions == 1)[0]

    # Nếu không phát hiện tiếng nói nào, trả về (0.0, 0.0)
    if len(speech_indices) == 0:
        return 0.0, 0.0

    start_idx = speech_indices[0]
    end_idx = speech_indices[-1]

    hop_sec = hop_size_ms / 1000.0
    frame_sec = frame_size_ms / 1000.0

    # Điểm bắt đầu tính từ đầu khung đầu tiên (start_idx * hop_sec)
    t_start = round(float(start_idx * hop_sec), 4)
    # Điểm kết thúc tính đến hết khung cuối cùng (end_idx * hop_sec + frame_sec)
    t_end = round(float(end_idx * hop_sec + frame_sec), 4)

    return t_start, t_end
