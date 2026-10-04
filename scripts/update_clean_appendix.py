#!/usr/bin/env python3
"""
Populate clean, professional, non-overlapping Appendix slides (Pages 18-22) on Canva presentation DAHWjThhNx0.
- Exactly 5 clean 4-quadrant slides (copied from Slide 16)
- 0 fills / NO unwanted graphs
- Short, punchy 2-line bullet points per card -> ZERO OVERLAP
- Professional typography & bilingual layout matching the rest of the deck
"""

import sys
import os
import json
import time

sys.path.append("/home/bim/Projects/Hackathon-t9-2026/scripts")
from canva_mcp_client import CanvaMCPClient

TARGET_DESIGN_ID = "DAHWjThhNx0"

SLIDES_DATA = [
    # -------------------------------------------------------------
    # Page index 17 (Slide 18)
    # -------------------------------------------------------------
    {
        "page_index": 17,
        "slide_num": "18",
        "category": "APPENDIX 01 · SYSTEM CONFIGURATION",
        "title": "Standard Audio Processing & VAD Hyperparameters",
        "card1_title": "01 · TIME-DOMAIN FRAMING",
        "card1_body": "Fs = 16 kHz · Frame = 20 ms (320 samples)\nHop = 10 ms (160 samples) · 50% Overlap",
        "card2_title": "02 · MINIMUM SPEECH CONSTRAINT",
        "card2_body": "Min Speech = 50 ms (5 consecutive hops)\nRejects impulse noise, clicks, and mic pops",
        "card3_title": "03 · MINIMUM SILENCE BRIDGING",
        "card3_body": "Min Silence = 200 ms (20 hops · Rubric rule)\nBridges intra-sentence pauses into continuous utterance",
        "card4_title": "04 · QUANTITATIVE GROUND TRUTH",
        "card4_body": "Normalized STE per utterance via max(STE)\nEvaluates boundary MAE & RMSE against .lab files",
    },
    # -------------------------------------------------------------
    # Page index 18 (Slide 19)
    # -------------------------------------------------------------
    {
        "page_index": 18,
        "slide_num": "19",
        "category": "APPENDIX 02 · FULL BENCHMARK SUMMARY",
        "title": "Comprehensive 4-Metric Comparison: Baseline vs Enhanced",
        "card1_title": "01 · TT1: BINARY SEARCH STE",
        "card1_body": "Baseline: MAE = 8.75 ms · RMSE = 11.34 ms · F1 = 0.993\nEnhanced (Logistic): MAE = 6.25 ms (Best Record)",
        "card2_title": "02 · TT2: ADAPTIVE HISTOGRAM",
        "card2_body": "Baseline: MAE = 17.50 ms · RMSE = 18.81 ms · F1 = 0.977\nEnhanced (2D STE-ZCR): MAE = 20.00 ms · UV = 94.7%",
        "card3_title": "03 · TT3: GAUSSIAN BAYES MODEL",
        "card3_body": "Baseline (1D STE): MAE = 11.25 ms · Stable & Robust\nMulti-var: MAE = 15.00 ms · High Covariance Variance",
        "card4_title": "04 · ENVIRONMENT COMPARISON",
        "card4_body": "Clean Studio: MAE = 5 - 10 ms (High SNR, sharp boundary)\nPhone Audio: MAE = 20 - 30 ms (Breath tail near noise floor)",
    },
    # -------------------------------------------------------------
    # Page index 19 (Slide 20)
    # -------------------------------------------------------------
    {
        "page_index": 19,
        "slide_num": "20",
        "category": "APPENDIX 03 · Q&A DEFENSE: ALGORITHMIC TRAPS",
        "title": "Key Counter-Arguments for Algorithmic Design Decisions",
        "card1_title": "Q1 · WHY NOT USE F0 (PITCH) FOR VAD?",
        "card1_body": "Unvoiced consonants (/s/, /f/, /t/) lack pitch.\nUsing F0 deletes initial and trailing unvoiced phonemes.",
        "card2_title": "Q2 · IS MAE = 0.0 MS OVERFITTING?",
        "card2_body": "No. phone_M2 labels [0.53s, 2.52s] align with 10ms hops.\nMatching frames 53 & 252 naturally yields 0.0 ms error.",
        "card3_title": "Q3 · WHY CHOOSE WEIGHT W = 5 IN TT2?",
        "card3_body": "W = 1 places threshold mid-way, cutting unvoiced speech.\nW = 5 biases toward silence peak M1 to protect weak trails.",
        "card4_title": "Q4 · WHY SMOOTH HISTOGRAM BY 5 BINS?",
        "card4_body": "Raw STE histograms contain noisy local spikes.\n5-point MA filter acts as low-pass filter to find true modes.",
    },
    # -------------------------------------------------------------
    # Page index 20 (Slide 21)
    # -------------------------------------------------------------
    {
        "page_index": 20,
        "slide_num": "21",
        "category": "APPENDIX 04 · Q&A DEFENSE: METHODOLOGY TRAPS",
        "title": "Counter-Arguments for Overfitting, Data Leakage & Physics",
        "card1_title": "Q5 · 200 MS SILENCE BRIDGING RULE",
        "card1_body": "Pauses < 200 ms represent natural intra-utterance breath.\nBridging groups words into a single continuous speech unit.",
        "card2_title": "Q6 · WHY 1D GAUSSIAN BEATS MULTI-VARIATE?",
        "card2_body": "Training set has only 4 files (small sample size).\nFull covariance overfits; 1D STE provides robust generalization.",
        "card3_title": "Q7 · WHY NOT OPTIMIZE ON TEST DATA?",
        "card3_body": "Strictly forbidden data leakage in real-world DSP & ML.\nParameters learned strictly from Train; Test is run once blind.",
        "card4_title": "Q8 · WHY MULTI-FEATURE FOR TT1 VARIANT 2?",
        "card4_body": "Combines STE, Pre-emphasis energy, ZCR, and HF ratio.\nLogistic score captures phonemes with physical interpretability.",
    },
    # -------------------------------------------------------------
    # Page index 21 (Slide 22)
    # -------------------------------------------------------------
    {
        "page_index": 21,
        "slide_num": "22",
        "category": "APPENDIX 05 · ADVANCED DSP THEORY",
        "title": "Physical Foundations & Formulations (Extended Variant 2)",
        "card1_title": "01 · PRE-EMPHASIS FIR FILTER",
        "card1_body": "y[n] = x[n] - 0.97 x[n-1] (+6 dB/octave high-pass boost)\nCompensates lip radiation and boosts unvoiced energy > 3 kHz",
        "card2_title": "02 · ZERO-CROSSING RATE (ZCR)",
        "card2_body": "High for unvoiced fricatives; low for periodic vowels\nCoupled with energy floor to eliminate background noise triggers",
        "card3_title": "03 · JOINT 2D HISTOGRAM [STE, ZCR]",
        "card3_body": "50x50 state space mapping energy vs spectral activity\nBoosts Unvoiced Recall from 89.1% to 94.7%",
        "card4_title": "04 · GAUSSIAN BAYES DERIVATION",
        "card4_body": "Equiprobable intersection: p(x | Silence) = p(x | Speech)\nQuadratic root yields analytical Bayes threshold T_Bayes ≈ 0.0020",
    }
]

def main():
    client = CanvaMCPClient()
    print("=" * 60)
    print("  🚀 CẬP NHẬT 5 SLIDE PHỤ LỤC CANVA CHUẨN ĐẸP (KHÔNG OVERLAP, KHÔNG GRAPH RÁC)")
    print(f"  Target Design ID: {TARGET_DESIGN_ID}")
    print("=" * 60)

    tx = client.call_tool("start-editing-transaction", {
        "design_id": TARGET_DESIGN_ID,
        "user_intent": "Update 5 clean quad appendix slides without overlap"
    })
    tx_id = tx["transaction"]["transaction_id"]
    pages = tx["pages"]
    all_texts = tx.get("richtexts", [])

    print(f"[+] Total pages in tx: {len(pages)}")

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
            candidates.sort(key=lambda t: (
                abs(t.get("containerElement", {}).get("position", {}).get("top", 0) - target_top) +
                abs(t.get("containerElement", {}).get("position", {}).get("left", 0) - target_left)
            ))
            return candidates[0]["element_id"]
        raise RuntimeError(f"Text element not found on page {page_idx} near top={target_top}, left={target_left}")

    operations = []

    for s in SLIDES_DATA:
        p_idx = s["page_index"]
        slide_num = s["slide_num"]
        print(f"  -> Preparing Slide {slide_num} ({s['category']})...")

        operations.extend([
            {"type": "replace_text", "element_id": find_text_el(p_idx, 48, 96), "text": s["category"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 84, 96), "text": s["title"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 295, 140), "text": s["card1_title"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 365, 140), "text": s["card1_body"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 295, 1040), "text": s["card2_title"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 365, 1040), "text": s["card2_body"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 625, 140), "text": s["card3_title"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 695, 140), "text": s["card3_body"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 625, 1040), "text": s["card4_title"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 695, 1040), "text": s["card4_body"]},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 1022, 96), "text": "DSP 2026  ·  APPENDIX"},
            {"type": "replace_text", "element_id": find_text_el(p_idx, 1022, 1740), "text": slide_num},
        ])

    print(f"\n[+] Total operations to apply: {len(operations)}")
    print("  Applying batch editing operations...")
    client.call_tool("perform-editing-operations", {
        "transaction_id": tx_id,
        "page_index": 18,
        "pages": pages,
        "operations": operations,
        "user_intent": "Populate 5 clean quadrant appendix slides with zero overlap"
    })
    print("[+] Batch editing applied successfully!")

    print("\n[+] Committing transaction...")
    commit_res = client.call_tool("commit-editing-transaction", {
        "transaction_id": tx_id,
        "user_intent": "Commit clean appendix slides"
    })
    print(f"[+] Commit completed: {commit_res}")
    print("\n🎉 HOÀN TẤT CẢ 5 SLIDE PHỤ LỤC! BỐ CỤC HOÀN TOÀN SẠCH ĐẸP, KHÔNG OVERLAP, KHÔNG GRAPH RÁC!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
