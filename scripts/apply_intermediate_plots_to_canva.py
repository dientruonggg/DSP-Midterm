#!/usr/bin/env python3
"""
Populate intermediate plots and explanation text onto Canva presentation DAHWjThhNx0:
- Slide 18 (index 17, 4-Quadrant): Preprocessing & Hyperparameters (Framing, 200ms rule, 50ms min speech)
- Slide 19 (index 18, Image+Cards): TT1 Intermediate Binary Search Plot (MAHXEztN3sg) + 3 Cards
- Slide 20 (index 19, Image+Cards): TT2 Intermediate Histogram Dual-Peak Plot (MAHXE3pfuAw) + 3 Cards
- Slide 21 (index 20, Image+Cards): TT3 Intermediate Gaussian Bayes Plot (MAHXE4ECrEE) + 3 Cards
- Slide 22 (index 21, 4-Quadrant): Q&A Defense Traps & Invariant Explanations (F0, MAE 0ms, SNR, 1D Gauss)
"""

import sys
import os
import json
import time

sys.path.append("/home/bim/Projects/Hackathon-t9-2026/scripts")
from canva_mcp_client import CanvaMCPClient

TARGET_DESIGN = "DAHWjThhNx0"

# Academic white-style asset IDs
ASSETS = {
    "tt1_binary_search": "MAHXEztN3sg",
    "tt2_histogram": "MAHXE3pfuAw",
    "tt3_gaussian_bayes": "MAHXE4ECrEE",
}

def main():
    client = CanvaMCPClient()
    print("=" * 65)
    print("  🚀 ĐẨY BIỂU ĐỒ TRUNG GIAN (CHUẨN STYLE ACADEMIC) LÊN CANVA")
    print(f"  Target Design: {TARGET_DESIGN}")
    print("=" * 65)

    print("\n[Step 1] Opening editing transaction...")
    tx = client.call_tool("start-editing-transaction", {
        "design_id": TARGET_DESIGN,
        "user_intent": "Update appendix slides with intermediate plots and text"
    })
    tx_id = tx["transaction"]["transaction_id"]
    pages = tx["pages"]
    all_texts = tx.get("richtexts", [])
    all_fills = tx.get("fills", [])

    print(f"[+] Total pages in transaction: {len(pages)}")

    def match_image_slide(p_idx):
        p_texts = [t for t in all_texts if t.get("page_index") == p_idx]
        p_fills = [f for f in all_fills if f.get("page_index") == p_idx]
        m = {}
        if p_fills:
            m["fill"] = p_fills[0]["element_id"]
        for t in p_texts:
            pos = t.get("containerElement", {}).get("position", {})
            top = pos.get("top") or 0
            left = pos.get("left") or 0
            el_id = t.get("element_id")
            if top < 60 and left < 200:
                m["category"] = el_id
            elif 60 <= top < 200 and left < 200:
                m["title"] = el_id
            elif 250 <= top < 330 and left > 1200:
                m["c1_title"] = el_id
            elif 330 <= top < 450 and left > 1200:
                m["c1_body"] = el_id
            elif 500 <= top < 560 and left > 1200:
                m["c2_title"] = el_id
            elif 560 <= top < 680 and left > 1200:
                m["c2_body"] = el_id
            elif 720 <= top < 790 and left > 1200:
                m["c3_title"] = el_id
            elif 790 <= top < 920 and left > 1200:
                m["c3_body"] = el_id
            elif top > 950 and left < 200:
                m["footer_left"] = el_id
            elif top > 950 and left > 1500:
                m["footer_right"] = el_id
        return m

    def match_quad_slide(p_idx):
        p_texts = [t for t in all_texts if t.get("page_index") == p_idx]
        m = {}
        for t in p_texts:
            pos = t.get("containerElement", {}).get("position", {})
            top = pos.get("top") or 0
            left = pos.get("left") or 0
            el_id = t.get("element_id")
            if top < 80 and left < 200:
                m["category"] = el_id
            elif 80 <= top < 200 and left < 200:
                m["title"] = el_id
            elif 270 <= top < 330 and 100 <= left < 500:
                m["c1_title"] = el_id
            elif 340 <= top < 500 and 100 <= left < 500:
                m["c1_body"] = el_id
            elif 270 <= top < 330 and left >= 900:
                m["c2_title"] = el_id
            elif 340 <= top < 500 and left >= 900:
                m["c2_body"] = el_id
            elif 600 <= top < 660 and 100 <= left < 500:
                m["c3_title"] = el_id
            elif 670 <= top < 800 and 100 <= left < 500:
                m["c3_body"] = el_id
            elif 600 <= top < 660 and left >= 900:
                m["c4_title"] = el_id
            elif 670 <= top < 800 and left >= 900:
                m["c4_body"] = el_id
            elif top > 950 and left < 200:
                m["footer_left"] = el_id
            elif top > 950 and left > 1500:
                m["footer_right"] = el_id
        return m

    operations = []

    # -------------------------------------------------------------
    # SLIDE 18 (index 17): 4-Quadrant Tham số & Tiền xử lý
    # -------------------------------------------------------------
    print("  -> Preparing Slide 18 (4-Quadrant Tham số & Tiền xử lý)...")
    m18 = match_quad_slide(17)
    operations.extend([
        {"type": "replace_text", "element_id": m18["category"], "text": "PHỤ LỤC 01 · TIỀN XỬ LÝ & BỘ THAM SỐ CHUẨN"},
        {"type": "replace_text", "element_id": m18["title"], "text": "Tham Số Phân Khung & Quy Tắc Khử Khoảng Lặng Ảo"},
        {"type": "replace_text", "element_id": m18["c1_title"], "text": "01 · PHÂN KHUNG TÍN HIỆU (FRAMING)"},
        {"type": "replace_text", "element_id": m18["c1_body"], "text": "Fs = 16 kHz · Chiều dài khung = 20 ms (320 mẫu)\nBước nhảy hop = 10 ms (160 mẫu) · Chồng lặp 50%"},
        {"type": "replace_text", "element_id": m18["c2_title"], "text": "02 · KHOẢNG LẶNG TỐI THIỂU (200 MS)"},
        {"type": "replace_text", "element_id": m18["c2_body"], "text": "Loại bỏ khoảng lặng ảo ngắn < 200 ms (20 khung)\nBắc cầu (bridge) các đoạn ngắt luồng hơi âm tắc"},
        {"type": "replace_text", "element_id": m18["c3_title"], "text": "03 · TIẾNG NÓI TỐI THIỂU (50 MS)"},
        {"type": "replace_text", "element_id": m18["c3_body"], "text": "Thời lượng phát âm tối thiểu = 50 ms (5 khung)\nKhử triệt để nhiễu xung (click, pop, tiếng va chạm)"},
        {"type": "replace_text", "element_id": m18["c4_title"], "text": "04 · CHUẨN HÓA NĂNG LƯỢNG STE"},
        {"type": "replace_text", "element_id": m18["c4_body"], "text": "Normalized STE = STE / max(STE) cho từng file\nĐưa miền giá trị về [0, 1] để áp dụng ngưỡng chung"},
        {"type": "replace_text", "element_id": m18["footer_left"], "text": "DSP 2026  ·  PHỤ LỤC THỰC NGHIỆM"},
        {"type": "replace_text", "element_id": m18["footer_right"], "text": "18"},
    ])

    # -------------------------------------------------------------
    # SLIDE 19 (index 18): TT1 Intermediate Binary Search Plot + 3 Cards
    # -------------------------------------------------------------
    print("  -> Preparing Slide 19 (TT1 Intermediate Plot + 3 Cards)...")
    m19 = match_image_slide(18)
    operations.append({
        "type": "update_fill",
        "element_id": m19["fill"],
        "asset_id": ASSETS["tt1_binary_search"],
        "asset_type": "image",
        "alt_text": "Biểu đồ hội tụ nhị phân TT1"
    })
    operations.extend([
        {"type": "replace_text", "element_id": m19["category"], "text": "PHỤ LỤC 02 · KẾT QUẢ HUẤN LUYỆN TT1"},
        {"type": "replace_text", "element_id": m19["title"], "text": "Minh Chứng Hội Tụ Tìm Kiếm Nhị Phân 40 Bước Lặp (CS425)"},
        {"type": "replace_text", "element_id": m19["c1_title"], "text": "01 · MIỀN TÌM KIẾM NHỊ PHÂN"},
        {"type": "replace_text", "element_id": m19["c1_body"], "text": "Khoảng khảo sát ngưỡng STE: [10⁻⁴, 0.05]\nSố bước lặp nhị phân: 40 epochs (chia đôi)"},
        {"type": "replace_text", "element_id": m19["c2_title"], "text": "02 · QUÁ TRÌNH HỘI TỤ TỐI ƯU"},
        {"type": "replace_text", "element_id": m19["c2_body"], "text": "Epoch 1: T = 0.02505 (thiếu âm, ΔD = -0.030 s)\nEpoch 6: T = 0.00244 · Epoch 40: T = 0.00261"},
        {"type": "replace_text", "element_id": m19["c3_title"], "text": "03 · NGƯỠNG DÙNG CHUNG"},
        {"type": "replace_text", "element_id": m19["c3_body"], "text": "Ngưỡng tối ưu chốt: T_opt = 0.00261\nMAE kiểm thử = 8.75 ms · F1-Score = 0.993"},
        {"type": "replace_text", "element_id": m19["footer_left"], "text": "DSP 2026  ·  PHỤ LỤC THỰC NGHIỆM"},
        {"type": "replace_text", "element_id": m19["footer_right"], "text": "19"},
    ])

    # -------------------------------------------------------------
    # SLIDE 20 (index 19): TT2 Intermediate Histogram Plot + 3 Cards
    # -------------------------------------------------------------
    print("  -> Preparing Slide 20 (TT2 Intermediate Plot + 3 Cards)...")
    m20 = match_image_slide(19)
    operations.append({
        "type": "update_fill",
        "element_id": m20["fill"],
        "asset_id": ASSETS["tt2_histogram"],
        "asset_type": "image",
        "alt_text": "Biểu đồ Histogram 2 đỉnh TT2"
    })
    operations.extend([
        {"type": "replace_text", "element_id": m20["category"], "text": "PHỤ LỤC 03 · KẾT QUẢ HUẤN LUYỆN TT2"},
        {"type": "replace_text", "element_id": m20["title"], "text": "Minh Chứng Histogram 2 Đỉnh Năng Lượng Trên 4 File Huấn Luyện"},
        {"type": "replace_text", "element_id": m20["c1_title"], "text": "01 · HISTOGRAM & LÀM TRƠN 5 BINS"},
        {"type": "replace_text", "element_id": m20["c1_body"], "text": "100 bins STE · Lọc trung bình trượt 5 bins\nKhử đỉnh nhiễu cục bộ tìm 2 mode thực sự"},
        {"type": "replace_text", "element_id": m20["c2_title"], "text": "02 · CÔNG THỨC TRỌNG SỐ W = 5"},
        {"type": "replace_text", "element_id": m20["c2_body"], "text": "T = (5×M₁ + M₂) / 6 (M₁: lặng, M₂: nói)\nThiên lệch sát đỉnh M₁ bảo toàn âm vô thanh"},
        {"type": "replace_text", "element_id": m20["c3_title"], "text": "03 · THÍCH ỨNG NHIỄU THEO FILE"},
        {"type": "replace_text", "element_id": m20["c3_body"], "text": "Phone (nhiễu): T = 0.0233 - 0.0267\nStudio (sạch): T = 0.0117 - 0.0150"},
        {"type": "replace_text", "element_id": m20["footer_left"], "text": "DSP 2026  ·  PHỤ LỤC THỰC NGHIỆM"},
        {"type": "replace_text", "element_id": m20["footer_right"], "text": "20"},
    ])

    # -------------------------------------------------------------
    # SLIDE 21 (index 20): TT3 Intermediate Gaussian Bayes Plot + 3 Cards
    # -------------------------------------------------------------
    print("  -> Preparing Slide 21 (TT3 Intermediate Plot + 3 Cards)...")
    m21 = match_image_slide(20)
    operations.append({
        "type": "update_fill",
        "element_id": m21["fill"],
        "asset_id": ASSETS["tt3_gaussian_bayes"],
        "asset_type": "image",
        "alt_text": "Biểu đồ phân phối Gauss TT3"
    })
    operations.extend([
        {"type": "replace_text", "element_id": m21["category"], "text": "PHỤ LỤC 04 · KẾT QUẢ HUẤN LUYỆN TT3"},
        {"type": "replace_text", "element_id": m21["title"], "text": "Minh Chứng Phân Phối Gauss Khung Âm & Giao Điểm Bayes"},
        {"type": "replace_text", "element_id": m21["c1_title"], "text": "01 · THỐNG KÊ GAUSS TỪ FILE .LAB"},
        {"type": "replace_text", "element_id": m21["c1_body"], "text": "Khoảng lặng: MeanSil = 0.00033 · StdSil = 0.00047\nTiếng nói: MeanSp = 0.19652 · StdSp = 0.23327"},
        {"type": "replace_text", "element_id": m21["c2_title"], "text": "02 · GIẢI PHƯƠNG TRÌNH BAYES"},
        {"type": "replace_text", "element_id": m21["c2_body"], "text": "Đẳng thức xác suất: p(x|Sil) = p(x|Sp)\nPhương trình bậc hai → T_Bayes = 0.00202"},
        {"type": "replace_text", "element_id": m21["c3_title"], "text": "03 · TÍNH ỔN ĐỊNH VÀ TỔNG QUÁT"},
        {"type": "replace_text", "element_id": m21["c3_body"], "text": "Gauss 1D STE ổn định nhất với tập train nhỏ\nĐạt MAE kiểm thử = 11.25 ms rất bền vững"},
        {"type": "replace_text", "element_id": m21["footer_left"], "text": "DSP 2026  ·  PHỤ LỤC THỰC NGHIỆM"},
        {"type": "replace_text", "element_id": m21["footer_right"], "text": "21"},
    ])

    # -------------------------------------------------------------
    # SLIDE 22 (index 21): 4-Quadrant Đối Thoại Phản Biện
    # -------------------------------------------------------------
    print("  -> Preparing Slide 22 (4-Quadrant Đối Thoại Phản Biện)...")
    m22 = match_quad_slide(21)
    operations.extend([
        {"type": "replace_text", "element_id": m22["category"], "text": "PHỤ LỤC 05 · ĐỐI THOẠI PHẢN BIỆN VẤN ĐÁP"},
        {"type": "replace_text", "element_id": m22["title"], "text": "Giải Trình Các Bẫy Câu Hỏi Lý Thuyết & Thực Nghiệm"},
        {"type": "replace_text", "element_id": m22["c1_title"], "text": "Q1 · VÌ SAO KHÔNG DÙNG F0 ĐỂ CẮT BIÊN?"},
        {"type": "replace_text", "element_id": m22["c1_body"], "text": "Phụ âm vô thanh (/s/, /f/, /t/) hoàn toàn không có pitch F0\nDùng F0 cắt biên sẽ làm mất phụ âm đầu và đuôi câu!"},
        {"type": "replace_text", "element_id": m22["c2_title"], "text": "Q2 · VÌ SAO MAE ĐẠT ĐƯỢC 0.0 MS Ở PHONE_M2?"},
        {"type": "replace_text", "element_id": m22["c2_body"], "text": "Mốc .lab [0.53 s, 2.52 s] trùng khớp lưới hop 10 ms (khung 53, 252)\nĐây là khớp khung hoàn hảo, không phải hiện tượng overfit"},
        {"type": "replace_text", "element_id": m22["c3_title"], "text": "Q3 · ẢNH HƯỞNG CỦA MỨC NHIỄU NỀN (SNR)"},
        {"type": "replace_text", "element_id": m22["c3_body"], "text": "Studio (SNR cao): Năng lượng lặng cực thấp, sai số 5-10 ms\nPhone (SNR thấp): Đuôi thở dễ chìm trong nhiễu nền"},
        {"type": "replace_text", "element_id": m22["c4_title"], "text": "Q4 · VÌ SAO CHỌN GAUSS 1 CHIỀU CHO TT3?"},
        {"type": "replace_text", "element_id": m22["c4_body"], "text": "Tập huấn luyện chỉ có 4 file (cỡ mẫu nhỏ N = 4)\nGauss 1D STE ổn định nhất; ma trận đa chiều dễ bị quá khớp"},
        {"type": "replace_text", "element_id": m22["footer_left"], "text": "DSP 2026  ·  PHỤ LỤC THỰC NGHIỆM"},
        {"type": "replace_text", "element_id": m22["footer_right"], "text": "22"},
    ])

    print(f"\n[Step 2] Executing {len(operations)} batch operations (Images + Texts)...")
    client.call_tool("perform-editing-operations", {
        "transaction_id": tx_id,
        "page_index": 18,
        "pages": pages,
        "operations": operations,
        "user_intent": "Update appendix slides with white-style intermediate plots and text"
    })
    print("[+] Batch editing operations applied successfully!")

    print("\n[Step 3] Committing editing transaction...")
    commit_res = client.call_tool("commit-editing-transaction", {
        "transaction_id": tx_id,
        "user_intent": "Commit updated appendix slides with academic white-style plots"
    })
    print(f"[+] Commit status: {commit_res}")

    print("\n" + "=" * 65)
    print("  🎉 HOÀN THÀNH 100%! TẤT CẢ BIỂU ĐỒ TRUNG GIAN ĐÃ ĐƯỢC ĐẨY LÊN CANVA:")
    print("  • Slide 18: Tham số chuẩn & Quy tắc khử khoảng lặng ảo 200 ms (4 Quadrants)")
    print("  • Slide 19: Biểu đồ hội tụ nhị phân TT1 (40 bước lặp, white style) + 3 thẻ")
    print("  • Slide 20: Biểu đồ 4 file Histogram TT2 (đỉnh M1, M2, W=5, white style) + 3 thẻ")
    print("  • Slide 21: Biểu đồ phân phối Gauss TT3 (Linear/Log, Bayes root, white style) + 3 thẻ")
    print("  • Slide 22: Đối thoại phản biện vấn đáp 4 câu hỏi bẫy kinh điển (4 Quadrants)")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(main())
