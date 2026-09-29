"""
Module: core.metrics
Description: Computes quantitative performance metrics (MAE and RMSE in milliseconds)
             comparing detected speech boundaries with ground truth.
"""

import math
from typing import Tuple, Dict, Any, List, Union


def calculate_mae_rmse(
    pred_boundaries: Tuple[float, float],
    gt_boundaries: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in milliseconds.

    Formula:
        MAE = (|T_hat_start - T_start| + |T_hat_end - T_end|) / 2 * 1000 (ms)
        RMSE = sqrt(((T_hat_start - T_start)^2 + (T_hat_end - T_end)^2) / 2) * 1000 (ms)

    Parameters:
        pred_boundaries (Tuple[float, float]): (pred_start, pred_end) in seconds.
        gt_boundaries (Tuple[float, float]): (gt_start, gt_end) in seconds.

    Returns:
        Tuple[float, float]: (mae_ms, rmse_ms) in milliseconds.
    """
    # [DONE]: Compute MAE and RMSE in milliseconds between predicted and ground-truth boundaries
    # print("[CALL] calculate_mae_rmse")

    pred_start, pred_end = pred_boundaries
    gt_start, gt_end = gt_boundaries

    # Tính độ lệch thời gian tại 2 mốc biên (đổi ra miligiây)
    diff_start_ms = abs(pred_start - gt_start) * 1000.0
    diff_end_ms = abs(pred_end - gt_end) * 1000.0

    # Sai số tuyệt đối trung bình (MAE)
    mae_ms = (diff_start_ms + diff_end_ms) / 2.0

    # Căn bậc hai của trung bình sai số bình phương (RMSE)
    squared_error = (diff_start_ms ** 2 + diff_end_ms ** 2) / 2.0
    rmse_ms = math.sqrt(squared_error)

    return mae_ms, rmse_ms


def evaluate_file_performance(
    file_id: str,
    pred_bounds: Tuple[float, float],
    gt_bounds: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Package boundary comparison metrics into a dictionary for reporting.

    Parameters:
        file_id (str): Identifier of the audio file (e.g., 'phone_F2').
        pred_bounds (Tuple[float, float]): Predicted [start, end].
        gt_bounds (Tuple[float, float]): Ground-truth [start, end].

    Returns:
        Dict[str, Any]: Dictionary containing file_id, deltas, MAE, RMSE.
    """
    # [DONE]: Calculate deltas, MAE, and RMSE and format into structured result dictionary
    # print("[CALL] evaluate_file_performance")

    pred_s, pred_e = pred_bounds
    gt_s, gt_e = gt_bounds

    delta_start_ms = abs(pred_s - gt_s) * 1000.0
    delta_end_ms = abs(pred_e - gt_e) * 1000.0
    mae_ms, rmse_ms = calculate_mae_rmse(pred_bounds, gt_bounds)

    return {
        "file_id": file_id,
        "gt_start": gt_s,
        "gt_end": gt_e,
        "pred_start": pred_s,
        "pred_end": pred_e,
        "delta_start_ms": delta_start_ms,
        "delta_end_ms": delta_end_ms,
        "mae_ms": mae_ms,
        "rmse_ms": rmse_ms,
    }


def summarize_benchmark(
    results_list: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute aggregate average MAE and average RMSE across all tested files.

    Parameters:
        results_list (List[Dict[str, Any]]): List of per-file evaluation dictionaries.

    Returns:
        Dict[str, float]: Aggregate summary containing avg_mae_ms and avg_rmse_ms.
    """
    # [DONE]: Aggregate per-file errors to compute overall benchmark MAE and RMSE
    # print("[CALL] summarize_benchmark")

    if not results_list:
        return {"avg_mae_ms": 0.0, "avg_rmse_ms": 0.0}

    # Tính trung bình cộng MAE và RMSE trên tất cả các file
    total_mae = sum(item["mae_ms"] for item in results_list)
    total_rmse = sum(item["rmse_ms"] for item in results_list)
    count = len(results_list)

    return {
        "avg_mae_ms": total_mae / count,
        "avg_rmse_ms": total_rmse / count,
    }
