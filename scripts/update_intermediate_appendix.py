#!/usr/bin/env python3
"""
Update Appendix slides (Slides 18-22, indices 17-21) on Canva design DAHWjThhNx0.
Specifically injects the intermediate results from training on TinHieuHuanLuyen:
- Slide 18: PHỤ LỤC 01 · TIỀN XỬ LÝ & BỘ THAM SỐ CHUẨN (Framing, 200ms silence bridge, 50ms speech, STE norm)
- Slide 19: PHỤ LỤC 02 · KẾT QUẢ TRUNG GIAN TT1 (Search interval [1e-4, 0.05], 40 epochs, convergence, T=0.00261)
- Slide 20: PHỤ LỤC 03 · KẾT QUẢ TRUNG GIAN TT2 (100 bins, 5-point MA, W=5 formula, 4 training files, noise adapt)
- Slide 21: PHỤ LỤC 04 · KẾT QUẢ TRUNG GIAN TT3 (Mean/Std Sil & Sp, Bayes quadratic equation, T_Bayes=0.00202)
- Slide 22: PHỤ LỤC 05 · ĐỐI THOẠI PHẢN BIỆN VẤN ĐÁP (Why no F0, MAE 0ms, SNR impact, 1D vs Multi-var)
"""

import sys
import os
import time

sys.path.append("/home/bim/Projects/Hackathon-t9-2026/scripts")
from canva_mcp_client import CanvaMCPClient

TARGET_DESIGN = "DAHWjThhNx0"

INTERMEDIATE_APPENDIX_DATA = [
    # -------------------------------------------------------------
    # Slide 18 (index 17) - Tham số & Tiền xử lý trung gian
    # -------------------------------------------------------------
    {
        "page_index": 17,
        "slide_num": "18",
        "category": "PHỤ LỤC 01 · TIỀN XỬ LÝ & BỘ THAM SỐ CHUẨN",
        "title": "Tham Số Phân Khung & Quy Tắc Khử Khoảng Lặng Ảo",
        "card1_title": "01 · PHÂN KHUNG TÍN HIỆU (FRAMING)",
        "card1_body": "Fs = 16 kHz · Chiều dài khung = 20 ms (320 mẫu)\nBước nhảy hop = 10 ms (160 mẫu) · Chồng lặp 50%",
        "card2_title": "02 · KHOẢNG LẶNG TỐI THIỂU (200 MS)",
        "card2_body": "Loại bỏ khoảng lặng ảo ngắn < 200 ms (20 khung)\nBắc cầu (bridge) các đoạn ngắt luồng hơi âm tắc",
        "card3_title": "03 · TIẾNG NÓI TỐI THIỂU (50 MS)",
        "card3_body": "Thời lượng phát âm tối thiểu = 50 ms (5 khung)\nKhử triệt để nhiễu xung (click, pop, tiếng va chạm)",
        "card4_title": "04 · CHUẨN HÓA NĂNG LƯỢNG STE",
        "card4_body": "Normalized STE = STE / max(STE) cho từng file\nĐưa miền giá trị về [0, 1] để áp dụng ngưỡng chung",
    },
    # -------------------------------------------------------------
    # Slide 19 (index 18) - TT1 Intermediate Binary Search
    # -------------------------------------------------------------
    {
        "page_index": 18,
        "slide_num": "19",
        "category": "PHỤ LỤC 02 · KẾT QUẢ TRUNG GIAN TT1",
        "title": "Quá Trình Hội Tụ Ngưỡng Nhị Phân Trên Tập Huấn Luyện",
        "card1_title": "01 · MIỀN TÌM KIẾM (SEARCH INTERVAL)",
        "card1_body": "Khoảng khảo sát ngưỡng STE: [10⁻⁴, 0.05]\nSố bước lặp nhị phân: 40 epochs (chia đôi liên tục)",
        "card2_title": "02 · HÀM MỤC TIÊU TỐI ƯU (OBJECTIVE)",
        "card2_body": "Tối thiểu hóa sai lệch thời lượng tiếng nói so với .lab\nΔDuration = |Total_Speech_Detected - Total_Speech_Lab|",
        "card3_title": "03 · NHẬT KÝ HỘI TỤ TRUNG GIAN",
        "card3_body": "Epoch 1: T = 0.02505 (thiếu âm, ΔD = -0.030 s)\nEpoch 6: T = 0.00244 · Epoch 40: T = 0.00261 (hội tụ)",
        "card4_title": "04 · NGƯỠNG TỐI ƯU CHỐT (T_OPT)",
        "card4_body": "Ngưỡng tối ưu dùng chung: T_opt = 0.00261\nĐạt MAE trung bình = 8.75 ms trên 4 file kiểm thử",
    },
    # -------------------------------------------------------------
    # Slide 20 (index 19) - TT2 Intermediate Histogram
    # -------------------------------------------------------------
    {
        "page_index": 19,
        "slide_num": "20",
        "category": "PHỤ LỤC 03 · KẾT QUẢ TRUNG GIAN TT2",
        "title": "Trích Xuất 2 Đỉnh Năng Lượng & Thích Nghi Mức Nhiễu",
        "card1_title": "01 · HISTOGRAM & LÀM TRƠN 5 BINS",
        "card1_body": "100 bins STE · Lọc trung bình trượt 5 bins (5-point MA)\nKhử đỉnh nhiễu cục bộ để trích xuất 2 đỉnh thực",
        "card2_title": "02 · CÔNG THỨC TRỌNG SỐ THUNG LŨNG (W = 5)",
        "card2_body": "T = (5×M₁ + M₂) / 6 (M₁: đỉnh lặng, M₂: đỉnh nói)\nThiên lệch sát đỉnh M₁ để bảo toàn âm vô thanh yếu",
        "card3_title": "03 · KẾT QUẢ 4 FILE HUẤN LUYỆN",
        "card3_body": "phone_F1: M₁=0.005, M₂=0.115 → T = 0.0233\nphone_M1: M₁=0.005, M₂=0.135 → T = 0.0267",
        "card4_title": "04 · THÍCH NGHI MÔI TRƯỜNG THU ÂM",
        "card4_body": "studio_F1: T = 0.0117 · studio_M1: T = 0.0150\nNhiễu điện thoại (SNR thấp) tự nâng ngưỡng T cao hơn",
    },
    # -------------------------------------------------------------
    # Slide 21 (index 20) - TT3 Intermediate Gaussian Bayes
    # -------------------------------------------------------------
    {
        "page_index": 20,
        "slide_num": "21",
        "category": "PHỤ LỤC 04 · KẾT QUẢ TRUNG GIAN TT3",
        "title": "Thống Kê Khung Chuẩn & Giải Nghiệm Giao Điểm Bayes",
        "card1_title": "01 · TRÍCH KHUNG THEO FILE .LAB",
        "card1_body": "Lọc toàn bộ khung tiếng nói (Sp) & khoảng lặng (Sil)\nTừ 4 file huấn luyện theo mốc nhãn chuẩn Ground Truth",
        "card2_title": "02 · THỐNG KÊ GAUSS (MEAN & STD)",
        "card2_body": "Khoảng lặng: MeanSil = 0.00033 · StdSil = 0.00047\nTiếng nói: MeanSp = 0.19652 · StdSp = 0.23327",
        "card3_title": "03 · GIẢI PHƯƠNG TRÌNH GIAO ĐIỂM BAYES",
        "card3_body": "Đẳng thức xác suất: P(x|Sil) = P(x|Sp) → Ax² + Bx + C = 0\nNghiệm bậc 2 tối ưu phân biệt: T_Bayes = 0.00202",
        "card4_title": "04 · PHÂN BỐ DỮ LIỆU TÁCH BIỆT RÕ",
        "card4_body": "Khoảng cách giữa 2 kỳ vọng lớn (|MeanSp - MeanSil| ≈ 0.2)\nPhân phối chuẩn mô hình hóa sắc nét ranh giới Sp/Sil",
    },
    # -------------------------------------------------------------
    # Slide 22 (index 21) - Q&A Traps & Counter-measures
    # -------------------------------------------------------------
    {
        "page_index": 21,
        "slide_num": "22",
        "category": "PHỤ LỤC 05 · ĐỐI THOẠI PHẢN BIỆN VẤN ĐÁP",
        "title": "Giải Trình Các Bẫy Câu Hỏi Lý Thuyết & Thực Nghiệm",
        "card1_title": "Q1 · VÌ SAO KHÔNG DÙNG F0 ĐỂ CẮT BIÊN?",
        "card1_body": "Phụ âm vô thanh (/s/, /f/, /t/) hoàn toàn không có pitch F0\nDùng F0 cắt biên sẽ làm mất phụ âm đầu và đuôi câu!",
        "card2_title": "Q2 · VÌ SAO MAE ĐẠT ĐƯỢC 0.0 MS Ở PHONE_M2?",
        "card2_body": "Mốc .lab [0.53 s, 2.52 s] trùng khớp lưới hop 10 ms (khung 53, 252)\nĐây là khớp khung hoàn hảo, không phải hiện tượng overfit",
        "card3_title": "Q3 · ẢNH HƯỞNG CỦA MỨC NHIỄU NỀN (SNR)",
        "card3_body": "Studio (SNR cao): Năng lượng lặng cực thấp, sai số 5-10 ms\nPhone (SNR thấp): Đuôi thở dễ chìm trong nhiễu nền",
        "card4_title": "Q4 · VÌ SAO CHỌN GAUSS 1 CHIỀU CHO TT3?",
        "card4_body": "Tập huấn luyện chỉ có 4 file (cỡ mẫu nhỏ N = 4)\nGauss 1D STE ổn định nhất; ma trận đa chiều dễ bị quá khớp",
    },
]

def main():
    client = CanvaMCPClient()
    print("=" * 65)
    print("  🚀 CẬP NHẬT PHỤ LỤC BÁO CÁO VAD VỚI DỮ LIỆU TRUNG GIAN HUẤN LUYỆN")
    print(f"  Target Design: {TARGET_DESIGN}")
    print("=" * 65)

    print("\n[Step 1] Opening editing transaction...")
    tx = client.call_tool("start-editing-transaction", {
        "design_id": TARGET_DESIGN,
        "user_intent": "Update appendix slides with intermediate training metrics"
    })
    tx_id = tx["transaction"]["transaction_id"]
    pages = tx["pages"]
    all_texts = tx.get("richtexts", [])

    print(f"[+] Total pages in transaction: {len(pages)}")

    def match_elements_for_page(p_idx):
        p_texts = [t for t in all_texts if t.get("page_index") == p_idx]
        mapping = {}
        for t in p_texts:
            pos = t.get("containerElement", {}).get("position", {})
            top = pos.get("top") or 0
            left = pos.get("left") or 0
            el_id = t.get("element_id")

            if top < 80 and left < 200:
                mapping["category"] = el_id
            elif 80 <= top < 200 and left < 200:
                mapping["title"] = el_id
            elif 270 <= top < 330 and 100 <= left < 500:
                mapping["c1_title"] = el_id
            elif 340 <= top < 500 and 100 <= left < 500:
                mapping["c1_body"] = el_id
            elif 270 <= top < 330 and left >= 900:
                mapping["c2_title"] = el_id
            elif 340 <= top < 500 and left >= 900:
                mapping["c2_body"] = el_id
            elif 600 <= top < 660 and 100 <= left < 500:
                mapping["c3_title"] = el_id
            elif 670 <= top < 800 and 100 <= left < 500:
                mapping["c3_body"] = el_id
            elif 600 <= top < 660 and left >= 900:
                mapping["c4_title"] = el_id
            elif 670 <= top < 800 and left >= 900:
                mapping["c4_body"] = el_id
            elif top > 950 and left < 200:
                mapping["footer_left"] = el_id
            elif top > 950 and left > 1500:
                mapping["footer_right"] = el_id
        return mapping

    operations = []

    for s in INTERMEDIATE_APPENDIX_DATA:
        p_idx = s["page_index"]
        slide_num = s["slide_num"]
        m = match_elements_for_page(p_idx)
        print(f"  -> Generating operations for Slide {slide_num} ({s['category']})... [matched {len(m)}/12 elements]")

        if len(m) < 12:
            print(f"  ⚠️ Warning: Missing elements on page {p_idx}: found {list(m.keys())}")

        operations.extend([
            {"type": "replace_text", "element_id": m["category"], "text": s["category"]},
            {"type": "replace_text", "element_id": m["title"], "text": s["title"]},
            {"type": "replace_text", "element_id": m["c1_title"], "text": s["card1_title"]},
            {"type": "replace_text", "element_id": m["c1_body"], "text": s["card1_body"]},
            {"type": "replace_text", "element_id": m["c2_title"], "text": s["card2_title"]},
            {"type": "replace_text", "element_id": m["c2_body"], "text": s["card2_body"]},
            {"type": "replace_text", "element_id": m["c3_title"], "text": s["card3_title"]},
            {"type": "replace_text", "element_id": m["c3_body"], "text": s["card3_body"]},
            {"type": "replace_text", "element_id": m["c4_title"], "text": s["card4_title"]},
            {"type": "replace_text", "element_id": m["c4_body"], "text": s["card4_body"]},
            {"type": "replace_text", "element_id": m["footer_left"], "text": "DSP 2026  ·  PHỤ LỤC THỰC NGHIỆM"},
            {"type": "replace_text", "element_id": m["footer_right"], "text": slide_num},
        ])

    print(f"\n[Step 2] Executing {len(operations)} text replacement operations...")
    client.call_tool("perform-editing-operations", {
        "transaction_id": tx_id,
        "page_index": 18,
        "pages": pages,
        "operations": operations,
        "user_intent": "Populate intermediate training metrics across 5 appendix slides"
    })
    print("[+] All operations applied successfully!")

    print("\n[Step 3] Committing editing transaction...")
    commit_res = client.call_tool("commit-editing-transaction", {
        "transaction_id": tx_id,
        "user_intent": "Commit updated intermediate appendix slides"
    })
    print(f"[+] Commit status: {commit_res}")

    print("\n" + "=" * 65)
    print("  🎉 HOÀN TẤT CẬP NHẬT 5 SLIDE PHỤ LỤC LÊN CANVA:")
    print("  1. Slide 18: Tiền xử lý & Khử khoảng lặng ảo (200ms bridge, 50ms min speech)")
    print("  2. Slide 19: Kết quả trung gian TT1 (Tìm kiếm nhị phân [1e-4, 0.05], 40 epochs)")
    print("  3. Slide 20: Kết quả trung gian TT2 (Histogram 100 bins, 5-point MA, W=5, 4 files)")
    print("  4. Slide 21: Kết quả trung gian TT3 (Mean/Std Sil & Sp, Giải pt Bayes, T=0.00202)")
    print("  5. Slide 22: Đối thoại phản biện vấn đáp (F0, MAE 0ms, SNR noise, 1D Gauss)")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(main())
