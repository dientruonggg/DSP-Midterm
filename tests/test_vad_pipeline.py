"""
Unit and integration tests for DSP MidTerm VAD project.
"""

import os
import math
import numpy as np
import pytest

from core.io_utils import read_wav, read_lab, get_speech_groundtruth
from core.features import frame_signal, compute_ste, normalize_ste, extract_ste_features
from core.postprocess import apply_threshold, remove_short_silences_200ms, extract_speech_boundaries
from core.metrics import calculate_mae_rmse, evaluate_file_performance, summarize_benchmark

from algorithms.tt1_hodgkinson import predict_vad_tt1, hodgkinson_cost_function
from algorithms.tt2_histogram import (
    compute_histogram_100bins,
    moving_average_smooth,
    find_histogram_peaks,
    compute_adaptive_threshold_tt2,
    predict_vad_tt2
)
from algorithms.tt3_gaussian import (
    estimate_gaussian_parameters,
    solve_bayes_decision_threshold,
    predict_vad_tt3
)
from main import run_pipeline


def test_io_utils_wav_loading():
    """Verify loading real WAV files returns normalized float64 signal and correct sample rate."""
    wav_path = "TinHieuHuanLuyen/phone_F1.wav"
    assert os.path.exists(wav_path)

    sig, fs = read_wav(wav_path)
    assert fs == 16000
    assert isinstance(sig, np.ndarray)
    assert sig.dtype == np.float64
    assert np.all(sig >= -1.0) and np.all(sig <= 1.0)
    assert len(sig) > 0


def test_io_utils_lab_parsing():
    """Verify parsing Praat .lab file skips F0 metadata lines."""
    lab_path = "TinHieuHuanLuyen/phone_F1.lab"
    assert os.path.exists(lab_path)

    segs = read_lab(lab_path)
    assert len(segs) > 0
    # First segment is silence
    assert segs[0][2] == "sil"
    # Ensure no F0 labels are included
    for s, e, lbl in segs:
        assert lbl in ("sil", "v", "uv")
        assert s <= e

    t_s, t_e = get_speech_groundtruth(segs)
    assert abs(t_s - 0.53) < 1e-3
    assert abs(t_e - 2.75) < 1e-3


def test_features_framing_and_ste():
    """Verify framing geometry (20ms/10ms) and Short-Time Energy normalization."""
    fs = 16000
    # 1 second test sine wave
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)

    frames, frame_times = frame_signal(sig, fs, frame_size_ms=20.0, hop_size_ms=10.0)
    expected_frame_len = int(0.02 * fs)  # 320 samples
    assert frames.shape[1] == expected_frame_len
    assert len(frames) == len(frame_times)

    ste = compute_ste(frames)
    assert len(ste) == len(frames)
    assert np.all(ste >= 0)

    ste_norm = normalize_ste(ste)
    assert abs(np.max(ste_norm) - 1.0) < 1e-6
    assert np.all(ste_norm >= 0.0) and np.all(ste_norm <= 1.0)


def test_postprocess_short_silence_bridging():
    """Verify bridging silences < 20 frames (200ms) surrounded by speech."""
    # Create decisions: 1 1 1, then 10 zeros (100ms silence), then 1 1 1
    decisions = np.array([0, 0, 1, 1, 1] + [0] * 10 + [1, 1, 1, 0, 0], dtype=np.int32)
    smoothed = remove_short_silences_200ms(decisions, hop_size_ms=10.0, min_silence_ms=200.0)

    # The 10 zeros should now be bridged to 1s
    assert np.all(smoothed[2:18] == 1)
    # The leading and trailing zeros must remain 0
    assert smoothed[0] == 0 and smoothed[1] == 0
    assert smoothed[-1] == 0 and smoothed[-2] == 0


def test_postprocess_long_silence_kept():
    """Verify silences >= 200ms (>= 20 frames) are NOT bridged."""
    # 25 zeros = 250ms silence
    decisions = np.array([1, 1, 1] + [0] * 25 + [1, 1, 1], dtype=np.int32)
    smoothed = remove_short_silences_200ms(decisions, hop_size_ms=10.0, min_silence_ms=200.0)

    # 25 zeros must remain intact
    assert np.all(smoothed[3:28] == 0)


def test_metrics_mae_rmse_calculation():
    """Verify MAE and RMSE calculation formulas."""
    pred = (1.00, 4.00)
    gt = (1.02, 4.04)

    # diff_s = 20ms, diff_e = 40ms
    # MAE = (20 + 40) / 2 = 30ms
    # RMSE = sqrt((20^2 + 40^2) / 2) = sqrt((400 + 1600) / 2) = sqrt(1000) = 31.6227ms
    mae, rmse = calculate_mae_rmse(pred, gt)
    assert abs(mae - 30.0) < 1e-4
    assert abs(rmse - math.sqrt(1000.0)) < 1e-4


def test_algorithms_tt1():
    """Verify TT1 Hodgkinson runs properly on test file."""
    sig, fs = read_wav("TinHieuKiemThu/phone_M2.wav")
    t_s, t_e, ste_norm, frame_times = predict_vad_tt1(sig, fs, threshold=0.0025)

    assert 0.0 < t_s < t_e
    assert len(ste_norm) == len(frame_times)
    # For phone_M2, ground truth is [0.53, 2.52], prediction should be within 20ms
    assert abs(t_s - 0.53) <= 0.03
    assert abs(t_e - 2.52) <= 0.03


def test_algorithms_tt2_histogram():
    """Verify TT2 Histogram smoothing, peak finding, and adaptive threshold."""
    sig, fs = read_wav("TinHieuKiemThu/phone_M2.wav")
    ste_norm, frame_times = extract_ste_features(sig, fs)

    counts, bin_centers = compute_histogram_100bins(ste_norm, num_bins=100)
    assert len(counts) == 100
    assert len(bin_centers) == 100

    smoothed = moving_average_smooth(counts, window_size=5)
    assert len(smoothed) == 100

    m1, m2 = find_histogram_peaks(smoothed, bin_centers)
    assert 0.0 <= m1 < m2 <= 1.0

    t_adapt = compute_adaptive_threshold_tt2(ste_norm, weight_w=5.0)
    assert 0.0 < t_adapt < 1.0

    t_s, t_e, _, _, t_calc = predict_vad_tt2(sig, fs, weight_w=5.0)
    assert 0.0 < t_s < t_e
    assert abs(t_calc - t_adapt) < 1e-6


def test_algorithms_tt3_gaussian():
    """Verify TT3 Gaussian Bayes estimation and decision boundary."""
    # Synthetic distributions
    mu_sil, sigma_sil = 0.000359, 0.000715
    mu_sp, sigma_sp = 0.196747, 0.233472

    t_bayes = solve_bayes_decision_threshold(mu_sil, sigma_sil, mu_sp, sigma_sp)
    # Theoretical value should be close to 0.00286
    assert 0.001 < t_bayes < 0.005

    sig, fs = read_wav("TinHieuKiemThu/phone_M2.wav")
    t_s, t_e, ste_norm, frame_times = predict_vad_tt3(sig, fs, threshold=t_bayes)
    assert 0.0 < t_s < t_e
    assert abs(t_s - 0.53) <= 0.03
    assert abs(t_e - 2.52) <= 0.03


def test_full_pipeline_headless():
    """Verify main run_pipeline executes without error across all 3 algorithms."""
    for algo in ["tt1", "tt2", "tt3"]:
        results = run_pipeline(test_dir="TinHieuKiemThu", algorithm_name=algo, show_plots=False)
        assert len(results) == 4
        summary = summarize_benchmark(results)
        assert summary["avg_mae_ms"] > 0.0
        assert summary["avg_rmse_ms"] > 0.0


def test_algorithms_tt3_gaussian_equal_variance_edge_case():
    """Verify solve_bayes_decision_threshold handles equal variances (abs(coef_a) < 1e-12) without ZeroDivisionError."""
    # When sigma_sil == sigma_sp, coef_a is exactly 0.0
    mu_sil = 0.001
    mu_sp = 0.010
    sigma = 0.005

    # Should solve linear equation and yield exact midpoint (0.001 + 0.010) / 2 = 0.0055
    t_linear = solve_bayes_decision_threshold(mu_sil, sigma, mu_sp, sigma)
    assert abs(t_linear - 0.0055) < 1e-6


def test_call_trace_logging(capsys):
    """Verify functions emit [CALL] logging trace when invoked."""
    _ = compute_ste(np.zeros((1, 320), dtype=np.float64))
    captured = capsys.readouterr()
    assert "[CALL] compute_ste" in captured.out

