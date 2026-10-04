#!/usr/bin/env python3
"""
Rebuild Canva presentation DAHWjThhNx0 Appendix (Slides 18-22):
- Slides 18-21: Reusable template from Slide 15 (Image on left + 3 cards on right)
  - Slide 18: Preprocessing & 200 ms Silence Bridging Rule (Image: MAHXE0vJV6k)
  - Slide 19: TT1 Intermediate Binary Search Optimization (Image: MAHXE3wMYpk)
  - Slide 20: TT2 Intermediate Dual-Peak Histogram Modes (Image: MAHXE5sGpkQ)
  - Slide 21: TT3 Intermediate Gaussian Distributions & Bayes Root (Image: MAHXE9STqEA)
- Slide 22: Reusable template from Slide 16 (4-Quadrant layout)
  - Slide 22: Q&A Defense Traps & Invariant Explanations (F0, 0ms, SNR, 1D Gauss)
"""

import sys
import os
import json
import time

sys.path.append("/home/bim/Projects/Hackathon-t9-2026/scripts")
from canva_mcp_client import CanvaMCPClient

TARGET_DESIGN = "DAHWjThhNx0"

ASSETS = {
    "preprocessing_200ms": "MAHXE0vJV6k",
    "tt1_binary_search": "MAHXE3wMYpk",
    "tt2_histogram": "MAHXE5sGpkQ",
    "tt3_gaussian_bayes": "MAHXE9STqEA",
}

SLIDES_CONFIG = [
    # -------------------------------------------------------------
    # Slide 18 (index 17): Image + 3 Cards
    # -------------------------------------------------------------
    {
        "type": "image_and_cards",
        "page_index": 17,
        "slide_num": "18",
        "category": "PHỤ LỤC 01 · TIỀN XỬ LÝ & BẮC CẦU 200 MS",
        "title": "Minh Chứng Quy Tắc Khử Khoảng Lặng Ảo Ngắn (< 200 ms)",
        "asset_id": ASSETS["preprocessing_200ms"],
        "alt_text": "Quy tắc bắc cầu 200ms khử khoảng lặng ảo",
        "card1_title": "01 · PHÂN KHUNG TÍN HIỆU",
        "card1_body": "Fs = 16 kHz · Khung = 20 ms (320 mẫu)\nBước nhảy hop = 10 ms (160 mẫu) · Chồng lặp 50%",
        "card2_title": "02 · BẮC CẦU KHOẢNG LẶNG 200 MS",
        "card2_body": "Loại bỏ triệt để khoảng lặng ảo ngắn < 200 ms\nBảo toàn cụm phát âm liên tục chuẩn theo .lab",
        "card3_title": "03 · TIẾNG NÓI TỐI THIỂU 50 MS",
        "card3_body": "Thời lượng phát âm tối thiểu = 50 ms (5 khung)\nKhử nhiễu xung cơ học (click, pop, gõ micro)",
    },
    # -------------------------------------------------------------
    # Slide 19 (index 18): Image + 3 Cards
    # -------------------------------------------------------------
    {
        "type": "image_and_cards",
        "page_index": 18,
        "slide_num": "19",
        "category": "PHỤ LỤC 02 · KẾT QUẢ HUẤN LUYỆN TT1",
        "title": "Minh Chứng Hội Tụ Tìm Kiếm Nhị Phân 40 Bước Lặp (CS425)",
        "asset_id": ASSETS["tt1_binary_search"],
        "alt_text": "Đồ thị hội tụ nhị phân TT1",
        "card1_title": "01 · MIỀN TÌM KIẾM NHỊ PHÂN",
        "card1_body": "Khảo sát ngưỡng STE trong [10⁻⁴, 0.05]\nSố bước lặp: 40 epochs chia đôi liên tục",
        "card2_title": "02 · NHẬT KÝ HỘI TỤ TỐI ƯU",
        "card2_body": "Epoch 1: T = 0.02505 (thiếu âm, ΔD = -0.030 s)\nEpoch 6: T = 0.00244 · Epoch 40: T = 0.00261",
        "card3_title": "03 · NGƯỠNG DÙNG CHUNG",
        "card3_body": "Ngưỡng tối ưu chốt: T_opt = 0.00261\nMAE kiểm thử = 8.75 ms · F1-Score = 0.993",
    },
    # -------------------------------------------------------------
    # Slide 20 (index 19): Image + 3 Cards
    # -------------------------------------------------------------
    {
        "type": "image_and_cards",
        "page_index": 19,
        "slide_num": "20",
        "category": "PHỤ LỤC 03 · KẾT QUẢ HUẤN LUYỆN TT2",
        "title": "Minh Chứng Histogram 2 Đỉnh Năng Lượng Trên 4 File Huấn Luyện",
        "asset_id": ASSETS["tt2_histogram"],
        "alt_text": "Đồ thị Histogram 2 đỉnh TT2",
        "card1_title": "01 · HISTOGRAM & LÀM TRƠN 5 BINS",
        "card1_body": "100 bins STE · Lọc trung bình trượt 5 bins\nKhử đỉnh nhiễu cục bộ tìm 2 mode thực sự",
        "card2_title": "02 · CÔNG THỨC TRỌNG SỐ W = 5",
        "card2_body": "T = (5×M₁ + M₂) / 6 (M₁: lặng, M₂: nói)\nThiên lệch sát đỉnh M₁ bảo toàn âm vô thanh",
        "card3_title": "03 · THÍCH ỨNG NHIỄU THEO FILE",
        "card3_body": "Phone (nhiễu): T = 0.0233 - 0.0267\nStudio (sạch): T = 0.0117 - 0.0150",
    },
    # -------------------------------------------------------------
    # Slide 21 (index 20): Image + 3 Cards
    # -------------------------------------------------------------
    {
        "type": "image_and_cards",
        "page_index": 20,
        "slide_num": "21",
        "category": "PHỤ LỤC 04 · KẾT QUẢ HUẤN LUYỆN TT3",
        "title": "Minh Chứng Phân Phối Gauss Khung Âm & Giao Điểm Bayes",
        "asset_id": ASSETS["tt3_gaussian_bayes"],
        "alt_text": "Đồ thị phân phối Gauss và điểm cắt Bayes TT3",
        "card1_title": "01 · THỐNG KÊ GAUSS TỪ FILE .LAB",
        "card1_body": "Khoảng lặng: MeanSil = 0.00033 · StdSil = 0.00047\nTiếng nói: MeanSp = 0.19652 · StdSp = 0.23327",
        "card2_title": "02 · GIẢI PHƯƠNG TRÌNH BAYES",
        "card2_body": "Đẳng thức xác suất: p(x|Sil) = p(x|Sp)\nPhương trình bậc hai → T_Bayes = 0.00202",
        "card3_title": "03 · TÍNH ỔN ĐỊNH VÀ TỔNG QUÁT",
        "card3_body": "Gauss 1D STE ổn định nhất với tập train nhỏ\nĐạt MAE kiểm thử = 11.25 ms rất bền vững",
    },
    # -------------------------------------------------------------
    # Slide 22 (index 21): 4-Quadrant Cards
    # -------------------------------------------------------------
    {
        "type": "quadrants",
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
    print("  🚀 REBUILD TOÀN DIỆN PHỤ LỤC CANVA VỚI BIỂU ĐỒ MINH CHỨNG")
    print(f"  Target Design: {TARGET_DESIGN}")
    print("=" * 65)

    # Step 1: Check existing pages and reset extra pages
    print("\n[Step 1] Checking existing pages...")
    des = client.call_tool("get-design", {"design_id": TARGET_DESIGN})
    total_pages = des["design"]["page_count"]
    print(f"Current total pages: {total_pages}")

    if total_pages > 17:
        pages_to_delete = list(range(18, total_pages + 1))
        print(f"Deleting extra pages {pages_to_delete}...")
        client.call_tool("merge-designs", {
            "type": "modify_existing_design",
            "design_id": TARGET_DESIGN,
            "operations": [{"type": "delete_pages", "page_numbers": pages_to_delete}],
            "user_intent": "Reset to original 17 pages"
        })
        time.sleep(1)

    # Step 2: Append 4 Image+Cards slides (from Page 14) and 1 Quad slide (from Page 15)
    print("\n[Step 2] Appending template slides from original deck...")
    # Insert 4 Image+Cards slides (Canva 1-based page 14)
    for i in range(4):
        print(f"  -> Appending Image+Cards slide copy {i+1}/4 (from template Page 14)...")
        client.call_tool("merge-designs", {
            "type": "modify_existing_design",
            "design_id": TARGET_DESIGN,
            "operations": [{
                "type": "insert_pages",
                "source": {
                    "type": "design",
                    "design_id": TARGET_DESIGN,
                    "page_numbers": [14]
                }
            }],
            "user_intent": f"Append image+cards appendix slide {i+1}"
        })
        time.sleep(1)

    # Insert 1 Quad slide (Canva 1-based page 15)
    print("  -> Appending 4-Quadrant slide copy 1/1 (from template Page 15)...")
    client.call_tool("merge-designs", {
        "type": "modify_existing_design",
        "design_id": TARGET_DESIGN,
        "operations": [{
            "type": "insert_pages",
            "source": {
                "type": "design",
                "design_id": TARGET_DESIGN,
                "page_numbers": [15]
            }
        }],
        "user_intent": "Append 4-quadrant defense slide"
    })
    time.sleep(1)

    # Step 3: Open transaction to populate content and images
    print("\n[Step 3] Opening transaction to populate content and images...")
    tx = client.call_tool("start-editing-transaction", {
        "design_id": TARGET_DESIGN,
        "user_intent": "Populate intermediate plots and explanation cards on appendix slides"
    })
    tx_id = tx["transaction"]["transaction_id"]
    pages = tx["pages"]
    all_texts = tx.get("richtexts", [])
    all_fills = tx.get("fills", [])

    print(f"[+] Total pages in transaction: {len(pages)}")

    def match_image_slide_elements(p_idx):
        p_texts = [t for t in all_texts if t.get("page_index") == p_idx]
        p_fills = [f for f in all_fills if f.get("page_index") == p_idx]
        mapping = {}

        if p_fills:
            mapping["fill"] = p_fills[0]["element_id"]

        for t in p_texts:
            pos = t.get("containerElement", {}).get("position", {})
            top = pos.get("top") or 0
            left = pos.get("left") or 0
            el_id = t.get("element_id")

            if top < 60 and left < 200:
                mapping["category"] = el_id
            elif 60 <= top < 200 and left < 200:
                mapping["title"] = el_id
            elif 250 <= top < 330 and left > 1200:
                mapping["c1_title"] = el_id
            elif 330 <= top < 450 and left > 1200:
                mapping["c1_body"] = el_id
            elif 500 <= top < 560 and left > 1200:
                mapping["c2_title"] = el_id
            elif 560 <= top < 680 and left > 1200:
                mapping["c2_body"] = el_id
            elif 720 <= top < 790 and left > 1200:
                mapping["c3_title"] = el_id
            elif 790 <= top < 920 and left > 1200:
                mapping["c3_body"] = el_id
            elif top > 950 and left < 200:
                mapping["footer_left"] = el_id
            elif top > 950 and left > 1500:
                mapping["footer_right"] = el_id
        return mapping

    def match_quad_elements(p_idx):
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

    for s in SLIDES_CONFIG:
        p_idx = s["page_index"]
        slide_num = s["slide_num"]
        stype = s["type"]

        if stype == "image_and_cards":
            m = match_image_slide_elements(p_idx)
            print(f"  -> Slide {slide_num} (Image+Cards): matched {len(m)} elements")
            operations.append({
                "type": "update_fill",
                "element_id": m["fill"],
                "asset_id": s["asset_id"],
                "asset_type": "image",
                "alt_text": s["alt_text"]
            })
            operations.extend([
                {"type": "replace_text", "element_id": m["category"], "text": s["category"]},
                {"type": "replace_text", "element_id": m["title"], "text": s["title"]},
                {"type": "replace_text", "element_id": m["c1_title"], "text": s["card1_title"]},
                {"type": "replace_text", "element_id": m["c1_body"], "text": s["card1_body"]},
                {"type": "replace_text", "element_id": m["c2_title"], "text": s["card2_title"]},
                {"type": "replace_text", "element_id": m["c2_body"], "text": s["card2_body"]},
                {"type": "replace_text", "element_id": m["c3_title"], "text": s["card3_title"]},
                {"type": "replace_text", "element_id": m["c3_body"], "text": s["card3_body"]},
                {"type": "replace_text", "element_id": m["footer_left"], "text": "DSP 2026  ·  PHỤ LỤC THỰC NGHIỆM"},
                {"type": "replace_text", "element_id": m["footer_right"], "text": slide_num},
            ])
        elif stype == "quadrants":
            m = match_quad_elements(p_idx)
            print(f"  -> Slide {slide_num} (Quadrants): matched {len(m)} elements")
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

    print(f"\n[Step 4] Applying {len(operations)} batch operations (Images + Texts)...")
    client.call_tool("perform-editing-operations", {
        "transaction_id": tx_id,
        "page_index": 18,
        "pages": pages,
        "operations": operations,
        "user_intent": "Update appendix slides with intermediate plots and text"
    })
    print("[+] Batch editing operations applied successfully!")

    print("\n[Step 5] Committing editing transaction...")
    commit_res = client.call_tool("commit-editing-transaction", {
        "transaction_id": tx_id,
        "user_intent": "Commit updated appendix slides with intermediate plots"
    })
    print(f"[+] Commit status: {commit_res}")

    print("\n" + "=" * 65)
    print("  🎉 HOÀN THÀNH 100%! TẤT CẢ BIỂU ĐỒ TRUNG GIAN ĐÃ ĐƯỢC CHÈN VÀO CANVA:")
    print("  • Slide 18: Biểu đồ bắc cầu 200 ms (sạch khoảng lặng ảo) + 3 thẻ tham số")
    print("  • Slide 19: Biểu đồ hội tụ nhị phân TT1 (40 epochs) + 3 thẻ ghi nhận T_opt")
    print("  • Slide 20: Biểu đồ 4 file Histogram TT2 (đỉnh M1, M2, W=5) + 3 thẻ thích nghi")
    print("  • Slide 21: Biểu đồ phân phối Gauss TT3 (Linear/Log, nghiệm Bayes) + 3 thẻ tham số")
    print("  • Slide 22: Bảng 4 thẻ đối thoại phản biện và câu hỏi bẫy vấn đáp")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(main())
