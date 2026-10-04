#!/usr/bin/env python3
"""
Populate appendix slides (Phụ lục A -> Phụ lục E) on Canva presentation DAHWjThhNx0.
Pages 17-23 are already inserted and arranged:
- Page 17: Quadrants (Phụ lục A: Tham số hệ thống)
- Page 18: Image + 3 cards (Phụ lục B: Đối chiếu định lượng & Benchmark)
- Page 19: Image + 3 cards (Phụ lục C1: Mô hình hóa Gaussian)
- Page 20: Image + 3 cards (Phụ lục C2: Phân tích hiện tượng từng file)
- Page 21: Quadrants (Phụ lục D1: Vấn đáp 4 câu hỏi bẫy P1)
- Page 22: Quadrants (Phụ lục D2: Vấn đáp 4 câu hỏi bẫy P2)
- Page 23: Quadrants (Phụ lục E: Kiến thức nâng cao Variant 2)
"""

import sys
import os
import json
import time

sys.path.append("/home/bim/Projects/Hackathon-t9-2026/scripts")
from canva_mcp_client import CanvaMCPClient

TARGET_DESIGN_ID = "DAHWjThhNx0"

ASSETS = {
    "benchmark": "MAHXElwJ4AE",
    "gaussian": "MAHXEu4AEIg",
    "tt1_composite": "MAHXElHeuqk",
    "tt2_composite": "MAHXEtqHKAA",
    "tt3_composite": "MAHXErFTBO0",
}

def main():
    client = CanvaMCPClient()
    print("=" * 60)
    print("  🚀 THỰC THI ĐIỀN NỘI DUNG 7 SLIDE PHỤ LỤC CANVA")
    print(f"  Target Design ID: {TARGET_DESIGN_ID}")
    print("=" * 60)

    # 1. Open editing transaction
    print("\n[1/2] Opening editing transaction on target design...")
    tx = client.call_tool("start-editing-transaction", {
        "design_id": TARGET_DESIGN_ID,
        "user_intent": "Open transaction to populate appendix slides 17-23"
    })
    tx_id = tx["transaction"]["transaction_id"]
    pages = tx["pages"]
    all_texts = tx.get("richtexts", [])
    all_fills = tx.get("fills", [])
    print(f"[+] Transaction opened: ID={tx_id}")
    print(f"[+] Total pages in tx: {len(pages)}")

    # Helper function to get text element by approximate coordinate
    def find_text_el(page_idx, target_top, target_left, tol=50):
        candidates = []
        for t in all_texts:
            if t.get("page_index") != page_idx:
                continue
            pos = t.get("containerElement", {}).get("position", {})
            top = pos.get("top", 0)
            left = pos.get("left", 0)
            if abs(top - target_top) <= tol and abs(left - target_left) <= tol:
                candidates.append(t)
        if candidates:
            # Pick closest
            candidates.sort(key=lambda t: (
                abs(t.get("containerElement", {}).get("position", {}).get("top", 0) - target_top) +
                abs(t.get("containerElement", {}).get("position", {}).get("left", 0) - target_left)
            ))
            return candidates[0]["element_id"]
        raise RuntimeError(f"Text element not found on page {page_idx} near top={target_top}, left={target_left}")

    def find_fill_el(page_idx):
        for f in all_fills:
            if f.get("page_index") == page_idx:
                return f["element_id"]
        raise RuntimeError(f"Fill element not found on page {page_idx}")

    operations = []

    # -------------------------------------------------------------
    # SLIDE 17: PHỤ LỤC A (4 quadrants)
    # -------------------------------------------------------------
    print("  -> Preparing Slide 17 (Phụ lục A: Thông số cấu hình)...")
    operations.extend([
        {"type": "replace_text", "element_id": find_text_el(17, 48, 96), "text": "PHỤ LỤC A · THAM SỐ CẤU HÌNH"},
        {"type": "replace_text", "element_id": find_text_el(17, 84, 96), "text": "Bảng thông số kỹ thuật chuẩn hóa toàn bộ hệ thống VAD"},
        {"type": "replace_text", "element_id": find_text_el(17, 295, 140), "text": "01 · TẦN SỐ & PHÂN KHUNG"},
        {"type": "replace_text", "element_id": find_text_el(17, 365, 140), "text": "Fs = 16 kHz · Chuẩn thoại băng rộng\nFrame = 20 ms (320 samples) · Quasi-stationary\nHop = 10 ms (160 samples) · 50% Overlap"},
        {"type": "replace_text", "element_id": find_text_el(17, 295, 1040), "text": "02 · TIÊU CHUẨN TIẾNG NÓI"},
        {"type": "replace_text", "element_id": find_text_el(17, 365, 1040), "text": "Min Speech = 50 ms (5 hops)\nLoại bỏ triệt để nhiễu xung ngắn hạn\nNgăn nhận nhầm tiếng click, pop, gõ micro"},
        {"type": "replace_text", "element_id": find_text_el(17, 625, 140), "text": "03 · KHOẢNG LẶNG ĐỀ BÀI"},
        {"type": "replace_text", "element_id": find_text_el(17, 695, 140), "text": "Min Silence = 200 ms (20 hops)\nYêu cầu bắt buộc theo đặc tả đề bài\nGộp ngắt hơi sinh lý thành câu liên tục"},
        {"type": "replace_text", "element_id": find_text_el(17, 625, 1040), "text": "04 · ĐẶC TRƯNG & THAM CHIẾU"},
        {"type": "replace_text", "element_id": find_text_el(17, 695, 1040), "text": "Normalized STE Energy theo từng file\nGround-truth nhãn .lab chuẩn hóa\nThước đo khách quan: MAE & RMSE biên thời gian"},
        {"type": "replace_text", "element_id": find_text_el(17, 1022, 96), "text": "DSP 2026  ·  PHỤ LỤC"},
        {"type": "replace_text", "element_id": find_text_el(17, 1022, 1740), "text": "17"}
    ])

    # -------------------------------------------------------------
    # SLIDE 18: PHỤ LỤC B (Benchmark Image + 3 cards)
    # -------------------------------------------------------------
    print("  -> Preparing Slide 18 (Phụ lục B: Đối chiếu định lượng)...")
    operations.extend([
        {"type": "update_fill", "element_id": find_fill_el(18), "asset_id": ASSETS["benchmark"], "asset_type": "image", "alt_text": "Biểu đồ so sánh Benchmark"},
        {"type": "replace_text", "element_id": find_text_el(18, 48, 96), "text": "PHỤ LỤC B · ĐỐI CHIẾU ĐỊNH LƯỢNG"},
        {"type": "replace_text", "element_id": find_text_el(18, 84, 96), "text": "Đánh giá toàn diện 4 chỉ số: MAE, RMSE, Frame F1 & UV Recall"},
        {"type": "replace_text", "element_id": find_text_el(18, 300, 1320), "text": "TT1 (LOGISTIC MỞ RỘNG)"},
        {"type": "replace_text", "element_id": find_text_el(18, 345, 1320), "text": "MAE: 6.25 ms (Kỷ lục toàn diện)\nRMSE: 7.80 ms · F1: 0.996 · UV: 0.968"},
        {"type": "replace_text", "element_id": find_text_el(18, 530, 1320), "text": "TT2 (HISTOGRAM 2D)"},
        {"type": "replace_text", "element_id": find_text_el(18, 575, 1320), "text": "MAE: 20.00 ms · RMSE: 21.56 ms\nF1: 0.988 · UV: 94.7% (Cứu âm vô thanh)"},
        {"type": "replace_text", "element_id": find_text_el(18, 760, 1320), "text": "TT3 (GAUSSIAN 1D)"},
        {"type": "replace_text", "element_id": find_text_el(18, 805, 1320), "text": "MAE: 11.25 ms · RMSE: 14.87 ms\nF1: 0.994 · UV: 97.1% (Mô hình ổn định)"},
        {"type": "replace_text", "element_id": find_text_el(18, 1022, 96), "text": "DSP 2026  ·  PHỤ LỤC"},
        {"type": "replace_text", "element_id": find_text_el(18, 1022, 1740), "text": "18"}
    ])

    # -------------------------------------------------------------
    # SLIDE 19: PHỤ LỤC C1 (Gaussian Distribution Image + 3 cards)
    # -------------------------------------------------------------
    print("  -> Preparing Slide 19 (Phụ lục C1: Mô hình hóa Gaussian)...")
    operations.extend([
        {"type": "update_fill", "element_id": find_fill_el(19), "asset_id": ASSETS["gaussian"], "asset_type": "image", "alt_text": "Phân bố Gaussian STE"},
        {"type": "replace_text", "element_id": find_text_el(19, 48, 96), "text": "PHỤ LỤC C1 · MÔ HÌNH HÓA GAUSSIAN"},
        {"type": "replace_text", "element_id": find_text_el(19, 84, 96), "text": "Phân bố thống kê STE và điểm cắt ngưỡng Bayes giải tích (TT3)"},
        {"type": "replace_text", "element_id": find_text_el(19, 300, 1320), "text": "PHÂN BỐ KHOẢNG LẶNG"},
        {"type": "replace_text", "element_id": find_text_el(19, 345, 1320), "text": "μ_sil = 0.00033 · σ_sil = 0.00047\nNăng lượng tập trung cực hẹp sát 0"},
        {"type": "replace_text", "element_id": find_text_el(19, 530, 1320), "text": "PHÂN BỐ TIẾNG NÓI"},
        {"type": "replace_text", "element_id": find_text_el(19, 575, 1320), "text": "μ_sp = 0.19652 · σ_sp = 0.23327\nPhổ năng lượng trải rộng phương sai lớn"},
        {"type": "replace_text", "element_id": find_text_el(19, 760, 1320), "text": "NGƯỠNG GIAO THOA BAYES"},
        {"type": "replace_text", "element_id": find_text_el(19, 805, 1320), "text": "T_Bayes ≈ 0.00202 (Nghiệm PT bậc hai)\nGiao điểm xác suất: p(x|Sil) = p(x|Sp)"},
        {"type": "replace_text", "element_id": find_text_el(19, 1022, 96), "text": "DSP 2026  ·  PHỤ LỤC"},
        {"type": "replace_text", "element_id": find_text_el(19, 1022, 1740), "text": "19"}
    ])

    # -------------------------------------------------------------
    # SLIDE 20: PHỤ LỤC C2 (Composite Image + 3 cards)
    # -------------------------------------------------------------
    print("  -> Preparing Slide 20 (Phụ lục C2: Phân tích hiện tượng từng file)...")
    operations.extend([
        {"type": "update_fill", "element_id": find_fill_el(20), "asset_id": ASSETS["tt1_composite"], "asset_type": "image", "alt_text": "Composite 4 files TT1"},
        {"type": "replace_text", "element_id": find_text_el(20, 48, 96), "text": "PHỤ LỤC C2 · PHÂN TÍCH HIỆN TƯỢNG"},
        {"type": "replace_text", "element_id": find_text_el(20, 84, 96), "text": "Giải thích cơ chế vật lý dẫn đến độ lệch biên trên 4 file kiểm thử"},
        {"type": "replace_text", "element_id": find_text_el(20, 300, 1320), "text": "PHONE_M2 (MAE = 0.0 MS)"},
        {"type": "replace_text", "element_id": find_text_el(20, 345, 1320), "text": "Mốc .lab [0.53s, 2.52s] là bội số hop 10ms\nTrùng khít frame 53 & 252 (Sai số = 0)"},
        {"type": "replace_text", "element_id": find_text_el(20, 530, 1320), "text": "PHONE_F2 (MAE = 20-30 MS)"},
        {"type": "replace_text", "element_id": find_text_el(20, 575, 1320), "text": "Đuôi hơi thở cuối câu lẫn nhiễu mic phone\nTrễ 2-3 frame do giới hạn năng lượng STE"},
        {"type": "replace_text", "element_id": find_text_el(20, 760, 1320), "text": "STUDIO_F2 & M2 (MAE = 5-10 MS)"},
        {"type": "replace_text", "element_id": find_text_el(20, 805, 1320), "text": "SNR phòng thu cao, đóng mở dứt khoát\nĐộ chính xác cao (lệch tối đa ≤ 1 frame)"},
        {"type": "replace_text", "element_id": find_text_el(20, 1022, 96), "text": "DSP 2026  ·  PHỤ LỤC"},
        {"type": "replace_text", "element_id": find_text_el(20, 1022, 1740), "text": "20"}
    ])

    # -------------------------------------------------------------
    # SLIDE 21: PHỤ LỤC D1 (4 quadrants)
    # -------------------------------------------------------------
    print("  -> Preparing Slide 21 (Phụ lục D1: Vấn đáp 4 câu hỏi bẫy P1)...")
    operations.extend([
        {"type": "replace_text", "element_id": find_text_el(21, 48, 96), "text": "PHỤ LỤC D1 · VẤN ĐÁP: 4 CÂU HỎI BẪY (P1)"},
        {"type": "replace_text", "element_id": find_text_el(21, 84, 96), "text": "Chiến lược phản biện sắc bén các câu hỏi trọng tâm của Hội đồng"},
        {"type": "replace_text", "element_id": find_text_el(21, 295, 140), "text": "01 · TẠI SAO KHÔNG DÙNG F0?"},
        {"type": "replace_text", "element_id": find_text_el(21, 365, 140), "text": "F0 chỉ có ở âm hữu thanh (dây thanh quản rung).\nÂm vô thanh (/s/, /f/, /t/, /k/) hoàn toàn không có F0.\nDùng F0 sẽ cắt mất toàn bộ phụ âm ở đầu và cuối câu!"},
        {"type": "replace_text", "element_id": find_text_el(21, 295, 1040), "text": "02 · MAE = 0 MS CÓ HACK SỐ?"},
        {"type": "replace_text", "element_id": find_text_el(21, 365, 1040), "text": "Không! Mốc .lab [0.53s, 2.52s] chia hết cho 10ms.\nThuật toán phân lớp chuẩn frame 53 và frame 252.\nĐây là trùng hợp tự nhiên theo lưới phân giải thời gian."},
        {"type": "replace_text", "element_id": find_text_el(21, 625, 140), "text": "03 · VÌ SAO CHỌN W = 5 Ở TT2?"},
        {"type": "replace_text", "element_id": find_text_el(21, 695, 140), "text": "W = 1 kéo ngưỡng ra giữa, làm mất phụ âm yếu.\nW = 5 ưu tiên đỉnh nhiễu M1: T = (5*M1 + M2)/6.\nKéo ngưỡng sát khoảng lặng để bảo tồn âm vô thanh."},
        {"type": "replace_text", "element_id": find_text_el(21, 625, 1040), "text": "04 · TẠI SAO CẦN SMOOTHING?"},
        {"type": "replace_text", "element_id": find_text_el(21, 695, 1040), "text": "Histogram thô của STE có nhiều gai nhiễu ngẫu nhiên.\nLọc trung bình trượt 5 điểm đóng vai trò Low-Pass Filter.\nLoại bỏ cực trị giả, làm nổi bật 2 mode thực sự M1, M2."},
        {"type": "replace_text", "element_id": find_text_el(21, 1022, 96), "text": "DSP 2026  ·  PHỤ LỤC"},
        {"type": "replace_text", "element_id": find_text_el(21, 1022, 1740), "text": "21"}
    ])

    # -------------------------------------------------------------
    # SLIDE 22: PHỤ LỤC D2 (4 quadrants)
    # -------------------------------------------------------------
    print("  -> Preparing Slide 22 (Phụ lục D2: Vấn đáp 4 câu hỏi bẫy P2)...")
    operations.extend([
        {"type": "replace_text", "element_id": find_text_el(22, 48, 96), "text": "PHỤ LỤC D2 · VẤN ĐÁP: 4 CÂU HỎI BẪY (P2)"},
        {"type": "replace_text", "element_id": find_text_el(22, 84, 96), "text": "Chiến lược phản biện phương pháp luận & cơ sở xử lý tín hiệu"},
        {"type": "replace_text", "element_id": find_text_el(22, 295, 140), "text": "05 · QUY TẮC KHOẢNG LẶNG 200 MS"},
        {"type": "replace_text", "element_id": find_text_el(22, 365, 140), "text": "Ngắt nghỉ < 200 ms giữa các từ là sinh lý phát âm.\nĐề bài quy định khoảng lặng tách câu dài tối thiểu 200 ms.\nBộ xử lý hậu kỳ (Bridge) gộp lại tạo câu nói liên tục .lab."},
        {"type": "replace_text", "element_id": find_text_el(22, 295, 1040), "text": "06 · TT3 ĐA BIẾN KÉM HƠN 1 CHIỀU?"},
        {"type": "replace_text", "element_id": find_text_el(22, 365, 1040), "text": "Hiện tượng Overfitting do tập huấn luyện chỉ có 4 file.\nGaussian đa biến phải ước lượng ma trận hiệp phương sai lớn.\nSTE 1D ít tham số nên hoạt động ổn định và tổng quát hơn."},
        {"type": "replace_text", "element_id": find_text_el(22, 625, 140), "text": "07 · SAO KHÔNG TỐI ƯU TRÊN TEST?"},
        {"type": "replace_text", "element_id": find_text_el(22, 695, 140), "text": "Vi phạm nghiêm trọng nguyên tắc Data Leakage trong DSP & ML.\nTập Test phải giữ nguyên vẹn để phản ánh khách quan năng lực.\nNhóm chỉ train trên 4 file và kiểm thử đúng 1 lần duy nhất."},
        {"type": "replace_text", "element_id": find_text_el(22, 625, 1040), "text": "08 · KHÁC BIỆT PHONE VS STUDIO"},
        {"type": "replace_text", "element_id": find_text_el(22, 695, 1040), "text": "Studio: SNR cao, nền âm học sạch → MAE đạt 5 - 10 ms.\nPhone: SNR thấp, micro phi tuyến kèm đuôi hơi thở kéo dài.\nNăng lượng tiệm cận nhiễu nền kéo trễ biên → MAE 20 - 30 ms."},
        {"type": "replace_text", "element_id": find_text_el(22, 1022, 96), "text": "DSP 2026  ·  PHỤ LỤC"},
        {"type": "replace_text", "element_id": find_text_el(22, 1022, 1740), "text": "22"}
    ])

    # -------------------------------------------------------------
    # SLIDE 23: PHỤ LỤC E (4 quadrants)
    # -------------------------------------------------------------
    print("  -> Preparing Slide 23 (Phụ lục E: Kiến thức nâng cao Variant 2)...")
    operations.extend([
        {"type": "replace_text", "element_id": find_text_el(23, 48, 96), "text": "PHỤ LỤC E · KIẾN THỨC NÂNG CAO (BIẾN THỂ 2)"},
        {"type": "replace_text", "element_id": find_text_el(23, 84, 96), "text": "Cơ sở lý thuyết & kỹ thuật đột phá cho biến thể mở rộng"},
        {"type": "replace_text", "element_id": find_text_el(23, 295, 140), "text": "01 · TIỀN NHẤN PRE-EMPHASIS FIR"},
        {"type": "replace_text", "element_id": find_text_el(23, 365, 140), "text": "y[n] = x[n] - 0.97 x[n-1] (Lọc thông cao bậc 1).\nBù suy giảm phổ -6 dB/octave do bức xạ môi.\nNâng biên độ tần số cao (> 3 kHz) của âm vô thanh (/s/, /f/)."},
        {"type": "replace_text", "element_id": find_text_el(23, 295, 1040), "text": "02 · ZERO CROSSING RATE (ZCR)"},
        {"type": "replace_text", "element_id": find_text_el(23, 365, 1040), "text": "Voiced: Năng lượng tập trung ở tần số thấp → ZCR thấp.\nUnvoiced: Năng lượng cao tần dồi dào → ZCR tăng vọt.\nSilence: Kết hợp Energy Floor để tránh nhiễu nhảy ZCR."},
        {"type": "replace_text", "element_id": find_text_el(23, 625, 140), "text": "03 · HISTOGRAM 2D [STE - ZCR]"},
        {"type": "replace_text", "element_id": find_text_el(23, 695, 140), "text": "Lưới đặc trưng 2D 50 x 50 ô phối hợp năng lượng & tần suất.\nPhân lớp theo tỷ số Log-likelihood có làm trơn Laplacian.\nTăng vượt bậc Unvoiced Recall từ 89.1% lên 94.7%."},
        {"type": "replace_text", "element_id": find_text_el(23, 625, 1040), "text": "04 · LOGISTIC HỒI QUY ĐA BIẾN (TT1-2)"},
        {"type": "replace_text", "element_id": find_text_el(23, 695, 1040), "text": "Học tự động vector trọng số tối ưu kết hợp đa đặc trưng.\nXác định biên dứt khoát, giảm thiểu tối đa hiện tượng trễ biên.\nThiết lập kỷ lục MAE trung bình toàn bài: 6.25 ms!"},
        {"type": "replace_text", "element_id": find_text_el(23, 1022, 96), "text": "DSP 2026  ·  PHỤ LỤC"},
        {"type": "replace_text", "element_id": find_text_el(23, 1022, 1740), "text": "23"}
    ])

    print(f"\n[+] Total editing operations queued: {len(operations)}")
    print("  Applying batch editing operations...")
    client.call_tool("perform-editing-operations", {
        "transaction_id": tx_id,
        "page_index": 17,
        "pages": pages,
        "operations": operations,
        "user_intent": "Update text and images on appendix slides 17-23"
    })
    print("[+] Batch editing operations applied successfully!")

    print("\n[2/2] Committing editing transaction...")
    commit_res = client.call_tool("commit-editing-transaction", {
        "transaction_id": tx_id,
        "user_intent": "Commit appendix slides 17-23"
    })
    print(f"[+] Commit completed: {commit_res}")
    print("\n🎉 TẤT CẢ 7 SLIDE PHỤ LỤC ĐÃ ĐƯỢC THÊM VÀO BÀI THUYẾT TRÌNH THÀNH CÔNG!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
