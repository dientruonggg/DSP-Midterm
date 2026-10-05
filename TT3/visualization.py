"""
Module: TT3.visualization
Mô tả: Chứa các hàm vẽ đồ thị trực quan:
       1. Đồ thị phân bố xác suất Gauss của khoảng lặng và tiếng nói cùng đường ngưỡng Bayes.
       2. Đồ thị kết quả phân đoạn cho 4 file kiểm thử (Waveform, STE chuẩn hóa, biên chuẩn vs biên dự đoán).
Quy định: Hình vẽ rõ ràng, font chữ to dễ đọc, đầy đủ title, label trục và chú giải (legend).
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple


def plot_gaussian_distributions(
    silence_ste: np.ndarray,
    speech_ste: np.ndarray,
    mu_sil: float,
    sigma_sil: float,
    mu_sp: float,
    sigma_sp: float,
    threshold: float,
    save_path: str = None
) -> plt.Figure:
    """
    Vẽ đồ thị mật độ phân bố xác suất Gauss lý thuyết và lược đồ histogram của STE chuẩn hóa
    cho hai lớp Silence và Speech từ dữ liệu huấn luyện, cùng đường ngưỡng phân tách Bayes.

    Tham số đầu vào:
        silence_ste (np.ndarray): Mảng STE của các khung khoảng lặng.
        speech_ste (np.ndarray): Mảng STE của các khung tiếng nói.
        mu_sil (float): Kỳ vọng của khoảng lặng.
        sigma_sil (float): Độ lệch chuẩn của khoảng lặng.
        mu_sp (float): Kỳ vọng của tiếng nói.
        sigma_sp (float): Độ lệch chuẩn của tiếng nói.
        threshold (float): Ngưỡng năng lượng Bayes tìm được.
        save_path (str, optional): Đường dẫn lưu file ảnh (nếu cần).

    Giá trị trả lại:
        plt.Figure: Đối tượng đồ thị matplotlib đã khởi tạo.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    # Giới hạn trục x tập trung vào vùng phân tách cận 0 đến 0.05 để nhìn rõ giao điểm
    x_eval = np.linspace(0.0, 0.015, 2000)

    # Hàm mật độ xác suất phân bố chuẩn Gauss (PDF) tự tính toán bằng công thức
    def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Tính đường cong PDF lý thuyết cho khoảng lặng và tiếng nói
    pdf_sil = gaussian_pdf(x_eval, mu_sil, sigma_sil)
    pdf_sp = gaussian_pdf(x_eval, mu_sp, sigma_sp)

    # Vẽ đường cong mật độ phân bố xác suất của Silence và Speech
    ax.plot(x_eval, pdf_sil, color="crimson", lw=2.2, label=f"Silence Gauss (μ={mu_sil:.5f}, σ={sigma_sil:.5f})")
    ax.plot(x_eval, pdf_sp, color="royalblue", lw=2.2, label=f"Speech Gauss (μ={mu_sp:.4f}, σ={sigma_sp:.4f})")

    # Kẻ đường thẳng đứng biểu thị ngưỡng tối ưu Bayes xác định được từ dữ liệu
    ax.axvline(
        x=threshold,
        color="darkgreen",
        linestyle="--",
        lw=2.5,
        label=f"Ngưỡng tối ưu Bayes T_opt = {threshold:.5f}"
    )

    # Tô màu các vùng phân tách xác suất
    ax.fill_between(x_eval[x_eval <= threshold], pdf_sil[x_eval <= threshold], color="crimson", alpha=0.15)
    ax.fill_between(x_eval[x_eval >= threshold], pdf_sp[x_eval >= threshold], color="royalblue", alpha=0.15)

    # Thiết lập tiêu đề và nhãn các trục
    ax.set_title("TT3: Phân bố xác suất Gauss của STE chuẩn hóa và Ngưỡng tối ưu Bayes (Training)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Năng lượng ngắn hạn chuẩn hóa (STE_norm)", fontsize=11)
    ax.set_ylabel("Mật độ xác suất p(x)", fontsize=11)
    ax.set_xlim(0.0, 0.010)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    # Lưu ảnh ra đĩa nếu có yêu cầu
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_single_file_result(
    file_name: str,
    signal: np.ndarray,
    sample_rate: int,
    ste_norm: np.ndarray,
    frame_times: np.ndarray,
    gt_bounds: Tuple[float, float],
    pred_bounds: Tuple[float, float],
    threshold: float,
    mae_ms: float,
    rmse_ms: float,
    save_path: str = None
) -> plt.Figure:
    """
    Vẽ đồ thị kết quả phân đoạn cho 01 file tín hiệu âm thanh kiểm thử:
    Gồm 2 đồ thị con xếp chồng (Waveform và STE chuẩn hóa), các đường biên thời gian chuẩn (đỏ)
    và biên thời gian do thuật toán xác định (xanh), cùng các số liệu định lượng sai số.

    Tham số đầu vào:
        file_name (str): Tên file kiểm thử.
        signal (np.ndarray): Mảng tín hiệu âm thanh dạng sóng.
        sample_rate (int): Tần số lấy mẫu (Hz).
        ste_norm (np.ndarray): Vector STE chuẩn hóa theo khung.
        frame_times (np.ndarray): Mốc thời gian trung tâm của các khung.
        gt_bounds (Tuple[float, float]): (gt_start, gt_end) biên chuẩn Ground Truth.
        pred_bounds (Tuple[float, float]): (pred_start, pred_end) biên dự đoán của thuật toán.
        threshold (float): Ngưỡng năng lượng Bayes sử dụng.
        mae_ms (float): Sai số tuyệt đối trung bình (miligiây).
        rmse_ms (float): Sai số bình phương trung bình (miligiây).
        save_path (str, optional): Đường dẫn lưu file đồ thị.

    Giá trị trả lại:
        plt.Figure: Đối tượng figure của matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=120)

    # Trục thời gian tính bằng giây cho tín hiệu dạng sóng thô
    time_wave = np.arange(len(signal)) / float(sample_rate)
    gt_start, gt_end = gt_bounds
    pred_start, pred_end = pred_bounds

    # --- Đồ thị con 1: Tín hiệu dạng sóng (Waveform) ---
    ax1.plot(time_wave, signal, color="#4a5568", lw=0.7, alpha=0.85, label="Tín hiệu âm thanh")
    # Vẽ biên chuẩn groundtruth (đường thẳng đứng màu đỏ)
    ax1.axvline(gt_start, color="red", linestyle="-", lw=2.0, label=f"Biên chuẩn GT: [{gt_start:.2f}s, {gt_end:.2f}s]")
    ax1.axvline(gt_end, color="red", linestyle="-", lw=2.0)
    # Vẽ biên thuật toán tìm được (đường nét đứt màu xanh dương)
    ax1.axvline(pred_start, color="blue", linestyle="--", lw=2.0, label=f"Biên TT3 tìm được: [{pred_start:.2f}s, {pred_end:.2f}s]")
    ax1.axvline(pred_end, color="blue", linestyle="--", lw=2.0)

    ax1.set_title(
        f"Phân đoạn VAD file kiểm thử: {file_name} | MAE = {mae_ms:.1f} ms | RMSE = {rmse_ms:.1f} ms",
        fontsize=12,
        fontweight="bold"
    )
    ax1.set_ylabel("Biên độ", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # --- Đồ thị con 2: Năng lượng ngắn hạn STE chuẩn hóa ---
    ax2.plot(frame_times, ste_norm, color="#2b6cb0", lw=1.2, label="Đặc trưng STE chuẩn hóa")
    # Vẽ đường ngưỡng năng lượng Bayes dùng chung
    ax2.axhline(threshold, color="#dd6b20", linestyle=":", lw=2.0, label=f"Ngưỡng T_Bayes = {threshold:.5f}")
    # Kẻ các mốc biên chuẩn và biên thuật toán trên đồ thị năng lượng
    ax2.axvline(gt_start, color="red", linestyle="-", lw=1.8)
    ax2.axvline(gt_end, color="red", linestyle="-", lw=1.8)
    ax2.axvline(pred_start, color="blue", linestyle="--", lw=1.8)
    ax2.axvline(pred_end, color="blue", linestyle="--", lw=1.8)

    ax2.set_xlabel("Thời gian (giây)", fontsize=10)
    ax2.set_ylabel("STE chuẩn hóa", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    # Lưu hình ảnh nếu đường dẫn lưu trữ được cung cấp
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
