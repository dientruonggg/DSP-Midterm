"""
Main Execution Script & Pipeline Manager
Author: Pipeline Manager (Student 3)
Description:
  Điều phối toàn bộ quy trình kiểm thử và đánh giá cho đồ án VAD giữa kỳ:
  1. Duyệt qua 4 tệp âm thanh trong thư mục kiểm thử TinHieuKiemThu/
     - phone_F2.wav
     - phone_M2.wav
     - studio_F2.wav
     - studio_M2.wav
  2. Áp dụng thuật toán VAD (mặc định TT3 - Gaussian Bayes, hoặc TT1, TT2)
  3. Tính toán định lượng sai số MAE và RMSE (đơn vị ms) so sánh với Ground-Truth
  4. Hiển thị đồng thời 4 cửa sổ đồ thị tại 4 góc màn hình theo đúng quy chế chấm thi:
     - Góc trên - trái:  phone_F2
     - Góc trên - phải:  phone_M2
     - Góc dưới - trái:  studio_F2
     - Góc dưới - phải:  studio_M2
  5. In bảng đối sánh Benchmark trực quan ra màn hình dòng lệnh.
"""

import os
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from core.io_utils import read_wav, read_lab, get_speech_groundtruth
from core.features import extract_ste_features
from core.metrics import evaluate_file_performance, summarize_benchmark

from algorithms.tt1_hodgkinson import predict_vad_tt1, train_optimal_threshold_tt1
from algorithms.tt2_histogram import (
    predict_vad_tt2,
    compute_histogram_100bins,
    moving_average_smooth,
    find_histogram_peaks
)
from algorithms.tt3_gaussian import (
    predict_vad_tt3,
    extract_speech_silence_ste_frames,
    estimate_gaussian_parameters,
    solve_bayes_decision_threshold
)


def setup_screen_window(
    fig: Figure,
    position_corner: str,
    screen_width: int = 1920,
    screen_height: int = 1080
) -> None:
    """
    Position a Matplotlib figure window at a specific corner of the screen.

    Parameters:
        fig (Figure): Matplotlib Figure instance.
        position_corner (str): 'top_left', 'top_right', 'bottom_left', or 'bottom_right'.
        screen_width (int): Screen resolution width in pixels (default: 1920).
        screen_height (int): Screen resolution height in pixels (default: 1080).
    """
    # [DONE]: Set window geometry to place Figure at the designated screen quadrant
    # print("[CALL] setup_screen_window")

    # Tính kích thước cho mỗi cửa sổ (chia nửa màn hình ngang và dọc)
    win_w = screen_width // 2
    win_h = (screen_height - 80) // 2

    coords_map = {
        "top_left": (0, 0),
        "top_right": (win_w, 0),
        "bottom_left": (0, win_h + 30),
        "bottom_right": (win_w, win_h + 30),
    }

    x_pos, y_pos = coords_map.get(position_corner, (0, 0))

    try:
        manager = fig.canvas.manager
        # Hỗ trợ backend TkAgg
        if hasattr(manager, "window") and hasattr(manager.window, "wm_geometry"):
            manager.window.wm_geometry(f"{win_w}x{win_h}+{x_pos}+{y_pos}")
        # Hỗ trợ backend Qt
        elif hasattr(manager, "window") and hasattr(manager.window, "setGeometry"):
            manager.window.setGeometry(x_pos, y_pos, win_w, win_h)
    except Exception:
        # Bỏ qua nếu môi trường không có GUI hoặc backend không hỗ trợ đặt tọa độ
        pass


def plot_vad_result(
    ax: Axes,
    signal: np.ndarray,
    sample_rate: int,
    frame_times: np.ndarray,
    ste_norm: np.ndarray,
    gt_bounds: Tuple[float, float],
    pred_bounds: Tuple[float, float],
    title: str
) -> None:
    """
    Plot waveform, superimposed STE_norm, Ground-Truth boundaries (red),
    and algorithm-detected boundaries (blue/green) with full labels and legend.

    Parameters:
        ax (Axes): Matplotlib Axes object.
        signal (np.ndarray): 1D audio sample array.
        sample_rate (int): Sampling rate (Hz).
        frame_times (np.ndarray): Frame timestamps (s).
        ste_norm (np.ndarray): Normalized STE curve.
        gt_bounds (Tuple[float, float]): Ground-truth (t_start, t_end).
        pred_bounds (Tuple[float, float]): Detected (t_start, t_end).
        title (str): Subplot title.
    """
    # [DONE]: Render waveform x(n), normalized STE curve, red GT lines, blue predicted lines
    # print("[CALL] plot_vad_result")

    # Trục thời gian tính bằng giây cho toàn bộ dạng sóng x(n)
    time_axis = np.arange(len(signal)) / sample_rate

    # Vẽ dạng sóng tín hiệu gốc x(n)
    ax.plot(time_axis, signal, color="#808080", alpha=0.55, linewidth=0.8, label="Dạng sóng x(n)")

    # Vẽ đường đặc trưng năng lượng ngắn hạn STE chuẩn hóa đè lên
    ax.plot(frame_times, ste_norm, color="#ff7f0e", linewidth=1.5, label="STE chuẩn hóa (norm)")

    # Vẽ các đường kẻ dọc màu đỏ thể hiện biên chuẩn Ground-Truth
    gt_s, gt_e = gt_bounds
    ax.axvline(gt_s, color="red", linestyle="--", linewidth=1.8, label=f"GT Start: {gt_s:.2f}s")
    ax.axvline(gt_e, color="red", linestyle="-.", linewidth=1.8, label=f"GT End: {gt_e:.2f}s")

    # Vẽ các đường kẻ dọc màu xanh thể hiện biên do thuật toán xác định
    pred_s, pred_e = pred_bounds
    ax.axvline(pred_s, color="#1f77b4", linestyle=":", linewidth=2.2, label=f"Pred Start: {pred_s:.2f}s")
    ax.axvline(pred_e, color="#1f77b4", linestyle="-", linewidth=2.2, label=f"Pred End: {pred_e:.2f}s")

    # Cài đặt tiêu đề, nhãn trục và chú giải (legend)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Thời gian (giây)", fontsize=9)
    ax.set_ylabel("Biên độ / STE_norm", fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)


def run_pipeline(
    test_dir: str = "TinHieuKiemThu",
    algorithm_name: str = "tt3",
    show_plots: bool = True,
    save_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute VAD pipeline over 4 test files, log evaluation results,
    and display 4 figure windows positioned at the 4 screen corners.

    Parameters:
        test_dir (str): Directory containing test WAV and LAB files.
        algorithm_name (str): 'tt1', 'tt2', or 'tt3'.
        show_plots (bool): Whether to display GUI windows.
        save_dir (Optional[str]): Directory to save PNG figures (optional).

    Returns:
        List[Dict[str, Any]]: List of evaluation dictionaries for benchmark.
    """
    # [DONE]: Run batch evaluation on 4 test files, compute MAE/RMSE, setup 4 corner figure windows
    # print("[CALL] run_pipeline")

    test_files_layout = [
        {"name": "phone_F2",  "pos": "top_left",     "label": "Góc Trên - Trái: phone_F2"},
        {"name": "phone_M2",  "pos": "top_right",    "label": "Góc Trên - Phải: phone_M2"},
        {"name": "studio_F2", "pos": "bottom_left",  "label": "Góc Dưới - Trái: studio_F2"},
        {"name": "studio_M2", "pos": "bottom_right", "label": "Góc Dưới - Phải: studio_M2"},
    ]

    results: List[Dict[str, float]] = []
    figs = []

    print("\n" + "=" * 78)
    print(f"   BẮT ĐẦU ĐÁNH GIÁ VAD - THUẬT TOÁN ĐANG CHẠY: {algorithm_name.upper()}")
    print("=" * 78)

    for item in test_files_layout:
        file_base = item["name"]
        wav_path = os.path.join(test_dir, f"{file_base}.wav")
        lab_path = os.path.join(test_dir, f"{file_base}.lab")

        if not os.path.exists(wav_path) or not os.path.exists(lab_path):
            print(f"Cảnh báo: Thiếu file {wav_path} hoặc {lab_path}")
            continue

        # Đọc dữ liệu âm thanh và nhãn chuẩn Ground-Truth
        signal, fs = read_wav(wav_path)
        lab_segments = read_lab(lab_path)
        gt_bounds = get_speech_groundtruth(lab_segments)

        # Chạy thuật toán được chỉ định
        if algorithm_name == "tt1":
            pred_s, pred_e, ste_norm, frame_times = predict_vad_tt1(signal, fs, threshold=0.0025)
            algo_info = "TT1 (Hodgkinson, T=0.0025)"
        elif algorithm_name == "tt2":
            pred_s, pred_e, ste_norm, frame_times, t_adapt = predict_vad_tt2(signal, fs, weight_w=5.0)
            algo_info = f"TT2 (Histogram W=5, T={t_adapt:.4f})"
        else:
            pred_s, pred_e, ste_norm, frame_times = predict_vad_tt3(signal, fs, threshold=0.002864)
            algo_info = "TT3 (Gaussian Bayes, T=0.00286)"

        # Đánh giá sai số MAE & RMSE (ms)
        pred_bounds = (pred_s, pred_e)
        perf = evaluate_file_performance(file_base, pred_bounds, gt_bounds)
        results.append(perf)

        # Tạo figure riêng cho từng file kiểm thử theo yêu cầu
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_title = f"{item['label']} | {algo_info}\nGT=[{gt_bounds[0]:.2f}, {gt_bounds[1]:.2f}]s | Pred=[{pred_s:.2f}, {pred_e:.2f}]s | MAE={perf['mae_ms']:.1f}ms"
        plot_vad_result(ax, signal, fs, frame_times, ste_norm, gt_bounds, pred_bounds, plot_title)

        # Sắp xếp vị trí cửa sổ vào 4 góc màn hình
        setup_screen_window(fig, item["pos"])
        figs.append(fig)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{file_base}_{algorithm_name}.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # In bảng tổng kết Benchmark định lượng
    print("\nBẢNG KẾT QUẢ ĐỐI SÁNH ĐỊNH LƯỢNG:")
    print("-" * 78)
    print(f"{'Tệp Âm Thanh':<15} | {'Ground-Truth':<16} | {'Biên Tìm Được':<16} | {'MAE (ms)':<10} | {'RMSE (ms)':<10}")
    print("-" * 78)
    for r in results:
        gt_str = f"[{r['gt_start']:.2f}, {r['gt_end']:.2f}]s"
        pred_str = f"[{r['pred_start']:.2f}, {r['pred_end']:.2f}]s"
        print(f"{r['file_id']:<15} | {gt_str:<16} | {pred_str:<16} | {r['mae_ms']:<10.2f} | {r['rmse_ms']:<10.2f}")
    print("-" * 78)

    summary = summarize_benchmark(results)
    print(f"{'TRUNG BÌNH':<15} | {'-':<16} | {'-':<16} | {summary['avg_mae_ms']:<10.2f} | {summary['avg_rmse_ms']:<10.2f}")
    print("=" * 78)

    if show_plots and matplotlib.get_backend().lower() not in ("agg", ""):
        print("\n[INFO] Đang hiển thị đồng thời 4 Figure tại 4 góc màn hình. Đóng cửa sổ để kết thúc.")
        plt.show()

    return results


def main() -> None:
    """
    Command Line Interface entry point.
    """
    # [DONE]: Parse CLI options and initiate pipeline execution
    # print("[CALL] main")

    parser = argparse.ArgumentParser(description="DSP MidTerm - Voice Activity Detection (VAD) Pipeline Manager")
    parser.add_argument("--test-dir", type=str, default="TinHieuKiemThu", help="Thư mục chứa tín hiệu kiểm thử")
    parser.add_argument("--train-dir", type=str, default="TinHieuHuanLuyen", help="Thư mục chứa tín hiệu huấn luyện")
    parser.add_argument("--algo", type=str, choices=["tt1", "tt2", "tt3"], default="tt3", help="Chọn thuật toán chạy (tt1, tt2, tt3)")
    parser.add_argument("--train", action="store_true", help="Chạy chế độ huấn luyện tối ưu ngưỡng trên tập huấn luyện")
    parser.add_argument("--no-plot", action="store_true", help="Chạy không mở cửa sổ đồ thị (chế độ headless/test)")
    parser.add_argument("--save-dir", type=str, default=None, help="Thư mục xuất ảnh các Figure đồ thị")
    args = parser.parse_args()

    if args.train:
        print("\n" + "=" * 78)
        print(f"   CHẾ ĐỘ HUẤN LUYỆN TỐI ƯU NGƯỠNG TRÊN TẬP {args.train_dir}")
        print("=" * 78)
        if args.algo == "tt1":
            print("[HUẤN LUYỆN TT1] Đang quét lưới tìm ngưỡng T_opt cực tiểu hóa MAE...")
            t_opt = train_optimal_threshold_tt1(args.train_dir)
            print(f"-> Ngưỡng tối ưu toàn cục tìm được TT1: T_opt = {t_opt:.6f} (Chuẩn: 0.0025)")
        elif args.algo == "tt2":
            print("[THÔNG BÁO TT2] Thuật toán TT2 (Giannakopoulos) là ngưỡng tự thích nghi per-utterance.")
            print(f"Không cần huấn luyện ngưỡng tĩnh toàn cục; tự động tính M1, M2 và T_adapt cho từng file ({args.train_dir}/):")
            import glob
            wav_files = sorted(glob.glob(os.path.join(args.train_dir, "*.wav")))
            for wav_p in wav_files:
                fname = os.path.basename(wav_p)
                sig, fs = read_wav(wav_p)
                ste, _ = extract_ste_features(sig, fs)
                counts, centers = compute_histogram_100bins(ste)
                sm = moving_average_smooth(counts, 5)
                m1, m2 = find_histogram_peaks(sm, centers)
                t_val = (5.0 * m1 + m2) / 6.0
                print(f"  • {fname:<15}: M1 (Silence) = {m1:.4f}, M2 (Speech) = {m2:.4f} -> T_adapt = {t_val:.4f}")
        else:
            print("[HUẤN LUYỆN TT3] Đang thống kê phân bố Gauss và giải phương trình xác suất Bayes...")
            sil_ste, sp_ste = extract_speech_silence_ste_frames(args.train_dir)
            mu_sil, sigma_sil, mu_sp, sigma_sp = estimate_gaussian_parameters(sil_ste, sp_ste)
            t_bayes = solve_bayes_decision_threshold(mu_sil, sigma_sil, mu_sp, sigma_sp)
            print(f"-> Phân bố Silence: mu = {mu_sil:.6f}, sigma = {sigma_sil:.6f} ({len(sil_ste)} khung)")
            print(f"-> Phân bố Speech : mu = {mu_sp:.6f}, sigma = {sigma_sp:.6f} ({len(sp_ste)} khung)")
            print(f"-> Ngưỡng Bayes tối ưu lý thuyết TT3: T_Bayes = {t_bayes:.6f} (Chuẩn: 0.002864)")
        return

    run_pipeline(
        test_dir=args.test_dir,
        algorithm_name=args.algo,
        show_plots=not args.no_plot,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
