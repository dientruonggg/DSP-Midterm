"""
Module: TT3.metrics
Mô tả: Đánh giá định lượng độ chính xác phân đoạn VAD bằng hai chỉ số:
       - MAE (Mean Absolute Error) tính bằng miligiây (ms).
       - RMSE (Root Mean Squared Error) tính bằng miligiây (ms).
Quy định: Tự code bằng toán học cơ sở, tuân thủ hướng dẫn đề tài giữa kỳ.
"""

import math
from typing import Tuple, Dict, Any, List


def calculate_mae_rmse(
    pred_boundaries: Tuple[float, float],
    gt_boundaries: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Tính toán sai số MAE và RMSE (đơn vị: milliseconds) giữa biên dự đoán và biên chuẩn.

    Công thức:
        MAE  = (|T_hat_start - T_start| + |T_hat_end - T_end|) / 2 * 1000 (ms)
        RMSE = sqrt(((T_hat_start - T_start)^2 + (T_hat_end - T_end)^2) / 2) * 1000 (ms)

    Tham số đầu vào:
        pred_boundaries (Tuple[float, float]): (pred_start, pred_end) tính bằng giây.
        gt_boundaries (Tuple[float, float]): (gt_start, gt_end) tính bằng giây.

    Giá trị trả lại:
        Tuple[float, float]: (mae_ms, rmse_ms) sai số tính bằng miligiây.
    """
    pred_start, pred_end = pred_boundaries
    gt_start, gt_end = gt_boundaries

    # Tính độ lệch thời gian mốc bắt đầu và mốc kết thúc (quy đổi ra miligiây)
    diff_start_ms = abs(pred_start - gt_start) * 1000.0
    diff_end_ms = abs(pred_end - gt_end) * 1000.0

    # Sai số tuyệt đối trung bình (MAE)
    mae_ms = (diff_start_ms + diff_end_ms) / 2.0

    # Sai số bình phương trung bình (RMSE)
    squared_error = (diff_start_ms ** 2 + diff_end_ms ** 2) / 2.0
    rmse_ms = math.sqrt(squared_error)

    return mae_ms, rmse_ms


def evaluate_vad_file(
    file_name: str,
    pred_boundaries: Tuple[float, float],
    gt_boundaries: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Đóng gói kết quả đánh giá chi tiết cho từng file tín hiệu vào một từ điển (dictionary).

    Tham số đầu vào:
        file_name (str): Tên file âm thanh (ví dụ 'phone_F2.wav').
        pred_boundaries (Tuple[float, float]): (pred_start, pred_end) tính bằng giây.
        gt_boundaries (Tuple[float, float]): (gt_start, gt_end) tính bằng giây.

    Giá trị trả lại:
        Dict[str, Any]: Bảng chứa các trường thông tin mốc biên và các loại sai số.
    """
    pred_start, pred_end = pred_boundaries
    gt_start, gt_end = gt_boundaries

    # Sai số riêng rẽ cho từng mốc biên
    delta_start_ms = abs(pred_start - gt_start) * 1000.0
    delta_end_ms = abs(pred_end - gt_end) * 1000.0

    # Tính MAE và RMSE cho file hiện tại
    mae_ms, rmse_ms = calculate_mae_rmse(pred_boundaries, gt_boundaries)

    return {
        "file_name": file_name,
        "gt_start": gt_start,
        "gt_end": gt_end,
        "pred_start": pred_start,
        "pred_end": pred_end,
        "delta_start_ms": delta_start_ms,
        "delta_end_ms": delta_end_ms,
        "mae_ms": mae_ms,
        "rmse_ms": rmse_ms,
    }


def summarize_evaluation(
    eval_list: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Tính toán trung bình cộng các chỉ số sai số MAE và RMSE trên toàn bộ tập file kiểm thử.

    Tham số đầu vào:
        eval_list (List[Dict[str, Any]]): Danh sách các kết quả đánh giá theo từng file.

    Giá trị trả lại:
        Dict[str, float]: Trung bình MAE (avg_mae_ms) và trung bình RMSE (avg_rmse_ms).
    """
    # Xử lý trường hợp danh sách rỗng
    if not eval_list:
        return {"avg_mae_ms": 0.0, "avg_rmse_ms": 0.0}

    # Tính tổng sai số trên toàn bộ tập kiểm thử
    total_mae = sum(item["mae_ms"] for item in eval_list)
    total_rmse = sum(item["rmse_ms"] for item in eval_list)
    count = len(eval_list)

    # Chia cho số lượng file để ra giá trị sai số trung bình
    return {
        "avg_mae_ms": total_mae / count,
        "avg_rmse_ms": total_rmse / count,
    }
