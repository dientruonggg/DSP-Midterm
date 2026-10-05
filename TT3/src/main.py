"""
Chương trình chính: main.py
Mục tiêu: Triển khai toàn bộ quy trình Thuật toán 3 (TT3) - Phân đoạn tiếng nói và khoảng lặng
          dựa trên mô hình phân bố Gauss của Năng lượng ngắn hạn (Short-Time Energy - STE).
Cách vận hành:
    1. Giai đoạn Huấn luyện (Training): Khảo sát 4 file âm thanh trong thư mục 'TinHieuHuanLuyen',
       thống kê các tham số phân bố Gauss (meanSil, stdSil, meanSp, stdSp),
       giải phương trình xác suất Bayes để xác định ngưỡng năng lượng tối ưu T_Bayes dùng chung.
    2. Giai đoạn Kiểm thử (Testing): Chạy thuật toán với ngưỡng T_Bayes trên 4 file 'TinHieuKiemThu',
       tính toán các chỉ số định lượng sai số MAE và RMSE (ms),
       xuất ra 4 figure trực quan thể hiện dạng sóng, đường STE và các mốc biên.
Quy định:
    - SV tự code toàn bộ các hàm xử lý tín hiệu, chỉ dùng hàm built-in của Numpy.
    - Mỗi hàm có brief comments mô tả chức năng, tham số đầu vào và giá trị trả lại.
    - Chú thích rõ ràng cho từng khối mã nguồn 5-10 dòng.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, Any, List

# Tìm kiếm thư mục gốc dự án linh hoạt
def find_project_root() -> Path:
    curr = Path(__file__).resolve()
    for candidate in [curr, *curr.parents]:
        if (candidate / "TinHieuHuanLuyen").exists() and (candidate / "TinHieuKiemThu").exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy thư mục gốc của dự án DSP_MidTerm.")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

try:
    from TT3.src.io_utils import read_wav, read_lab, get_speech_groundtruth
    from TT3.src.features import extract_ste_features
    from TT3.src.gaussian_vad import (
        survey_training_data,
        estimate_gaussian_parameters,
        solve_bayes_threshold,
        predict_vad
    )
    from TT3.src.metrics import evaluate_vad_file, summarize_evaluation
    from TT3.src.visualization import plot_gaussian_distributions, plot_single_file_result
except (ImportError, ModuleNotFoundError):
    try:
        from src.io_utils import read_wav, read_lab, get_speech_groundtruth
        from src.features import extract_ste_features
        from src.gaussian_vad import (
            survey_training_data,
            estimate_gaussian_parameters,
            solve_bayes_threshold,
            predict_vad
        )
        from src.metrics import evaluate_vad_file, summarize_evaluation
        from src.visualization import plot_gaussian_distributions, plot_single_file_result
    except (ImportError, ModuleNotFoundError):
        from io_utils import read_wav, read_lab, get_speech_groundtruth
        from features import extract_ste_features
        from gaussian_vad import (
            survey_training_data,
            estimate_gaussian_parameters,
            solve_bayes_threshold,
            predict_vad
        )
        from metrics import evaluate_vad_file, summarize_evaluation
        from visualization import plot_gaussian_distributions, plot_single_file_result


def run_training_phase(
    training_dir: str,
    output_dir: str,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> Dict[str, Any]:
    """
    Thực hiện khảo sát dữ liệu huấn luyện và xác định ngưỡng phân biệt tối ưu dùng chung.

    Tham số đầu vào:
        training_dir (str): Thư mục chứa dữ liệu huấn luyện (TinHieuHuanLuyen).
        output_dir (str): Thư mục lưu trữ biểu đồ và báo cáo đầu ra.
        frame_size_ms (float): Độ dài khung tính theo miligiây (25 ms).
        hop_size_ms (float): Bước nhảy khung tính theo miligiây (10 ms).

    Giá trị trả lại:
        Dict[str, Any]: Từ điển chứa các tham số ước lượng và ngưỡng tối ưu tìm được.
    """
    print("=" * 80)
    print(" GIAI ĐOẠN 1: KHẢO SÁT DỮ LIỆU HUẤN LUYỆN VÀ XÁC ĐỊNH BỘ THAM SỐ TỐI ƯU (TT3)")
    print(f" Cấu hình phân khung: Độ dài khung = {frame_size_ms:.1f} ms | Bước nhảy = {hop_size_ms:.1f} ms")
    print("=" * 80)

    # Khảo sát toàn bộ khung tín hiệu trên 4 file huấn luyện
    silence_ste, speech_ste, per_file_stats = survey_training_data(
        training_dir=training_dir,
        frame_size_ms=frame_size_ms,
        hop_size_ms=hop_size_ms
    )

    print(f"\n[+] Tổng số khung khoảng lặng khảo sát: {len(silence_ste):,}")
    print(f"[+] Tổng số khung tiếng nói khảo sát   : {len(speech_ste):,}")
    print("\nChi tiết thống kê trên từng file huấn luyện đơn lẻ:")
    print("-" * 80)
    print(f"{'Tên file':<16} | {'meanSil':<10} | {'stdSil':<10} | {'meanSp':<10} | {'stdSp':<10}")
    print("-" * 80)
    for fname, stats in per_file_stats.items():
        print(
            f"{fname:<16} | {stats['mean_sil']:<10.6f} | {stats['std_sil']:<10.6f} | "
            f"{stats['mean_sp']:<10.6f} | {stats['std_sp']:<10.6f}"
        )
    print("-" * 80)

    # Ước lượng các tham số kỳ vọng và độ lệch chuẩn gộp cho cả 4 file huấn luyện
    mu_sil, sigma_sil, mu_sp, sigma_sp = estimate_gaussian_parameters(silence_ste, speech_ste)

    print("\n[+] BỘ THAM SỐ THỐNG KÊ GỘP DÙNG CHUNG CHO 04 TÍN HIỆU HUẤN LUYỆN:")
    print(f"    - Khoảng lặng (Silence): meanSil = {mu_sil:.6f}, stdSil = {sigma_sil:.6f}")
    print(f"    - Tiếng nói (Speech)  : meanSp  = {mu_sp:.6f}, stdSp  = {sigma_sp:.6f}")

    # Giải phương trình xác suất Bayes để tìm ngưỡng tối ưu T_Bayes
    threshold_bayes = solve_bayes_threshold(mu_sil, sigma_sil, mu_sp, sigma_sp)
    print(f"\n[+] GIẢI PHƯƠNG TRÌNH PHÂN BỐ GAUSS XÁC SUẤT BAYES p(x|sil) = p(x|sp):")
    print(f"    => Ngưỡng phân biệt tối ưu tìm được: T_opt = {threshold_bayes:.6f}")

    # Vẽ đồ thị phân bố xác suất Gauss và lưu vào thư mục output
    dist_plot_path = os.path.join(output_dir, "gaussian_distributions_tt3.png")
    plot_gaussian_distributions(
        silence_ste=silence_ste,
        speech_ste=speech_ste,
        mu_sil=mu_sil,
        sigma_sil=sigma_sil,
        mu_sp=mu_sp,
        sigma_sp=sigma_sp,
        threshold=threshold_bayes,
        save_path=dist_plot_path
    )
    print(f"[+] Đã lưu biểu đồ phân bố xác suất Gauss tại: {dist_plot_path}")

    return {
        "mu_sil": mu_sil,
        "sigma_sil": sigma_sil,
        "mu_sp": mu_sp,
        "sigma_sp": sigma_sp,
        "threshold": threshold_bayes,
        "per_file_stats": per_file_stats
    }


def run_testing_phase(
    test_dir: str,
    threshold: float,
    output_dir: str,
    frame_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Thực hiện phân đoạn VAD trên 4 file kiểm thử với ngưỡng tối ưu tìm được,
    tính toán các chỉ số định lượng sai số MAE/RMSE và xuất 4 figure.

    Tham số đầu vào:
        test_dir (str): Thư mục chứa dữ liệu kiểm thử (TinHieuKiemThu).
        threshold (float): Ngưỡng năng lượng Bayes tìm được ở giai đoạn huấn luyện.
        output_dir (str): Thư mục lưu các figure kết quả.
        frame_size_ms (float): Độ dài khung (25 ms).
        hop_size_ms (float): Bước nhảy khung (10 ms).

    Giá trị trả lại:
        List[Dict[str, Any]]: Danh sách kết quả đánh giá định lượng cho từng file.
    """
    print("\n" + "=" * 80)
    print(f" GIAI ĐOẠN 2: THỰC NGHIỆM TRÊN TÍN HIỆU KIỂM THỬ VỚI NGƯỠNG T_opt = {threshold:.6f}")
    print(f" Cấu hình phân khung: Độ dài khung = {frame_size_ms:.1f} ms | Bước nhảy = {hop_size_ms:.1f} ms")
    print("=" * 80)

    # Lấy danh sách 4 file kiểm thử theo thứ tự tên file
    wav_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"Không tìm thấy file kiểm thử WAV trong: {test_dir}")

    eval_records: List[Dict[str, Any]] = []
    generated_figures: List[plt.Figure] = []

    # Duyệt tuần tự qua 4 file tín hiệu kiểm thử
    for wav_path in wav_files:
        file_name = os.path.basename(wav_path)
        lab_path = os.path.splitext(wav_path)[0] + ".lab"

        # Đọc dữ liệu âm thanh và nhãn ground truth chuẩn
        signal, sample_rate = read_wav(wav_path)
        lab_segments = read_lab(lab_path)
        gt_bounds = get_speech_groundtruth(lab_segments)

        # Chạy thuật toán phân đoạn VAD TT3 với ngưỡng Bayes dùng chung
        pred_start, pred_end, ste_norm, frame_times, _ = predict_vad(
            signal=signal,
            sample_rate=sample_rate,
            threshold=threshold,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms
        )
        pred_bounds = (pred_start, pred_end)

        # Tính toán sai số định lượng (MAE, RMSE tính theo miligiây)
        eval_item = evaluate_vad_file(file_name, pred_bounds, gt_bounds)
        eval_records.append(eval_item)

        # Xuất figure trực quan thể hiện input, năng lượng STE và biên phân đoạn
        fig_save_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_vad.png")
        fig = plot_single_file_result(
            file_name=file_name,
            signal=signal,
            sample_rate=sample_rate,
            ste_norm=ste_norm,
            frame_times=frame_times,
            gt_bounds=gt_bounds,
            pred_bounds=pred_bounds,
            threshold=threshold,
            mae_ms=eval_item["mae_ms"],
            rmse_ms=eval_item["rmse_ms"],
            save_path=fig_save_path
        )
        generated_figures.append(fig)

    # In bảng tổng hợp kết quả thực nghiệm chi tiết
    print("\nBẢNG KẾT QUẢ PHÂN ĐOẠN ĐỊNH LƯỢNG TRÊN TẬP KIỂM THỬ:")
    print("-" * 92)
    header = f"{'File tín hiệu':<15} | {'GT [s]':<14} | {'Dự đoán [s]':<14} | {'ΔStart (ms)':<11} | {'ΔEnd (ms)':<10} | {'MAE (ms)':<8} | {'RMSE (ms)':<8}"
    print(header)
    print("-" * 92)
    for r in eval_records:
        gt_str = f"[{r['gt_start']:.2f}, {r['gt_end']:.2f}]"
        pred_str = f"[{r['pred_start']:.2f}, {r['pred_end']:.2f}]"
        print(
            f"{r['file_name']:<15} | {gt_str:<14} | {pred_str:<14} | "
            f"{r['delta_start_ms']:<11.1f} | {r['delta_end_ms']:<10.1f} | "
            f"{r['mae_ms']:<8.1f} | {r['rmse_ms']:<8.1f}"
        )
    print("-" * 92)

    # Thống kê giá trị sai số trung bình toàn tập kiểm thử
    summary = summarize_evaluation(eval_records)
    print(f"[*] Sai số tuyệt đối trung bình toàn tập (Average MAE) : {summary['avg_mae_ms']:.2f} ms")
    print(f"[*] Sai số bình phương trung bình toàn tập (Average RMSE): {summary['avg_rmse_ms']:.2f} ms")
    print("-" * 92)

    return eval_records


def arrange_and_show_figures(is_gui_enabled: bool = True) -> None:
    """
    Sắp xếp các cửa sổ figure trên màn hình theo yêu cầu hướng dẫn nộp bài thi
    (đặt 4 figure trên 4 góc màn hình để quan sát trực quan), hoặc hiển thị nếu có giao diện GUI.

    Tham số đầu vào:
        is_gui_enabled (bool): Bật/tắt việc gọi plt.show().
    """
    # Nếu đang trong môi trường hỗ trợ hiển thị giao diện đồ họa GUI
    if is_gui_enabled and "DISPLAY" in os.environ:
        try:
            plt.show()
        except Exception as e:
            print(f"[!] Bỏ qua hiển thị tương tác: {e}")


def main() -> None:
    """
    Điểm khởi chạy chương trình (Main Entrypoint).
    """
    # Xác định các đường dẫn thư mục dự án
    train_dir = os.path.join(PROJECT_ROOT, "TinHieuHuanLuyen")
    test_dir = os.path.join(PROJECT_ROOT, "TinHieuKiemThu")
    output_dir = os.path.join(PROJECT_ROOT, "TT3", "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Chạy pha huấn luyện tìm tham số Gauss và ngưỡng Bayes
    train_res = run_training_phase(training_dir=train_dir, output_dir=output_dir)

    # 2. Chạy pha kiểm thử đánh giá trên 4 file kiểm thử
    run_testing_phase(
        test_dir=test_dir,
        threshold=train_res["threshold"],
        output_dir=output_dir
    )

    print(f"\n[V] Đã hoàn thành chương trình TT3. Toàn bộ hình ảnh kết quả được lưu tại: {output_dir}\n")


if __name__ == "__main__":
    main()
