"""
Interface & Dry-Run Verification Suite for DSP MidTerm VAD Project.

Empirical verification script testing:
1. Criterion 1: Module syntax and imports.
2. Criterion 2: Dry-run invocation of all 20+ interface functions, verifying [CALL] log traces.
3. Criterion 3: Static AST scan confirming zero imports of prohibited toolboxes (scipy.signal, librosa).
4. Edge cases and stress testing.
"""

import ast
import io
import math
import os
import sys
sys.path.insert(0, os.path.abspath("."))

from contextlib import redirect_stdout
from typing import Callable, Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Module imports under test
import core
import core.io_utils
import core.features
import core.postprocess
import core.metrics
import algorithms
import algorithms.tt1_hodgkinson
import algorithms.tt2_histogram
import algorithms.tt3_gaussian
import main


def run_and_capture(func: Callable, *args, **kwargs) -> Tuple[Any, str]:
    """Execute a function and capture its stdout."""
    f = io.StringIO()
    with redirect_stdout(f):
        result = func(*args, **kwargs)
    return result, f.getvalue()


# ---------------------------------------------------------------------------
# Criterion 1: Module Imports and Syntax
# ---------------------------------------------------------------------------
MODULES_TO_TEST = [
    "core",
    "core.io_utils",
    "core.features",
    "core.postprocess",
    "core.metrics",
    "algorithms",
    "algorithms.tt1_hodgkinson",
    "algorithms.tt2_histogram",
    "algorithms.tt3_gaussian",
    "main",
]

@pytest.mark.parametrize("module_name", MODULES_TO_TEST)
def test_criterion1_module_importable(module_name: str):
    """Verify that all required modules exist and import cleanly without syntax errors."""
    assert module_name in sys.modules, f"Module {module_name} is not loaded in sys.modules."


# ---------------------------------------------------------------------------
# Criterion 3: Static Scan for Prohibited DSP Toolboxes
# ---------------------------------------------------------------------------
PROHIBITED_MODULES = ["scipy.signal", "librosa"]

def get_python_files() -> List[str]:
    py_files = []
    for root_dir in ["core", "algorithms"]:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))
    py_files.append("main.py")
    return sorted(py_files)


@pytest.mark.parametrize("py_file", get_python_files())
def test_criterion3_no_prohibited_imports(py_file: str):
    """AST static scan ensuring no imports of scipy.signal, librosa, or forbidden toolboxes."""
    with open(py_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=py_file)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prohibited in PROHIBITED_MODULES:
                    assert not alias.name.startswith(prohibited), (
                        f"Prohibited import '{alias.name}' detected in {py_file}:{node.lineno}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for prohibited in PROHIBITED_MODULES:
                    assert not node.module.startswith(prohibited), (
                        f"Prohibited from-import '{node.module}' detected in {py_file}:{node.lineno}"
                    )


# ---------------------------------------------------------------------------
# Criterion 2: Interface Functions Dry-Run & Clean Invocation Verification
# ---------------------------------------------------------------------------
# We test all 29 interface functions across core/, algorithms/, and main.py

def test_dryrun_core_io_read_wav():
    wav_path = "TinHieuHuanLuyen/phone_F1.wav"
    res, out = run_and_capture(core.io_utils.read_wav, wav_path)
    sig, fs = res
    assert isinstance(sig, np.ndarray)
    assert fs == 16000
    assert "[CALL] read_wav" not in out, f"Noisy '[CALL] read_wav' trace was not commented out!"


def test_dryrun_core_io_read_lab():
    lab_path = "TinHieuHuanLuyen/phone_F1.lab"
    res, out = run_and_capture(core.io_utils.read_lab, lab_path)
    assert isinstance(res, list)
    assert len(res) > 0
    assert "[CALL] read_lab" not in out, f"Noisy '[CALL] read_lab' trace was not commented out!"


def test_dryrun_core_io_get_speech_groundtruth():
    mock_segments = [(0.0, 0.5, "sil"), (0.5, 2.5, "v"), (2.5, 3.0, "sil")]
    res, out = run_and_capture(core.io_utils.get_speech_groundtruth, mock_segments)
    assert res == (0.5, 2.5)
    assert "[CALL] get_speech_groundtruth" not in out, f"Noisy '[CALL] get_speech_groundtruth' trace was not commented out!"


def test_dryrun_core_features_frame_signal():
    dummy_signal = np.sin(np.linspace(0, 10, 1600))
    res, out = run_and_capture(core.features.frame_signal, dummy_signal, 16000)
    frames, times = res
    assert frames.ndim == 2
    assert len(frames) == len(times)
    assert "[CALL] frame_signal" not in out, f"Noisy '[CALL] frame_signal' trace was not commented out!"


def test_dryrun_core_features_compute_ste():
    dummy_frames = np.ones((5, 320), dtype=np.float64) * 0.1
    res, out = run_and_capture(core.features.compute_ste, dummy_frames)
    assert isinstance(res, np.ndarray)
    assert len(res) == 5
    assert "[CALL] compute_ste" not in out, f"Noisy '[CALL] compute_ste' trace was not commented out!"


def test_dryrun_core_features_normalize_ste():
    dummy_ste = np.array([10.0, 50.0, 100.0], dtype=np.float64)
    res, out = run_and_capture(core.features.normalize_ste, dummy_ste)
    assert np.isclose(np.max(res), 1.0)
    assert "[CALL] normalize_ste" not in out, f"Noisy '[CALL] normalize_ste' trace was not commented out!"


def test_dryrun_core_features_extract_ste_features():
    dummy_signal = np.sin(np.linspace(0, 10, 1600))
    res, out = run_and_capture(core.features.extract_ste_features, dummy_signal, 16000)
    ste_norm, times = res
    assert len(ste_norm) == len(times)
    assert "[CALL] extract_ste_features" not in out, f"Noisy '[CALL] extract_ste_features' trace was not commented out!"


def test_dryrun_core_postprocess_apply_threshold():
    dummy_ste = np.array([0.01, 0.05, 0.1], dtype=np.float64)
    res, out = run_and_capture(core.postprocess.apply_threshold, dummy_ste, 0.04)
    assert np.array_equal(res, np.array([0, 1, 1]))
    assert "[CALL] apply_threshold" not in out, f"Noisy '[CALL] apply_threshold' trace was not commented out!"


def test_dryrun_core_postprocess_remove_short_silences_200ms():
    decisions = np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.int32)
    res, out = run_and_capture(core.postprocess.remove_short_silences_200ms, decisions, 10.0, 200.0)
    assert np.all(res == 1)
    assert "[CALL] remove_short_silences_200ms" not in out, f"Noisy '[CALL] remove_short_silences_200ms' trace was not commented out!"


def test_dryrun_core_postprocess_extract_speech_boundaries():
    decisions = np.array([0, 1, 1, 1, 0], dtype=np.int32)
    times = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    res, out = run_and_capture(core.postprocess.extract_speech_boundaries, decisions, times, 10.0, 20.0)
    assert res == (0.01, 0.05)
    assert "[CALL] extract_speech_boundaries" not in out, f"Noisy '[CALL] extract_speech_boundaries' trace was not commented out!"


def test_dryrun_core_metrics_calculate_mae_rmse():
    pred = (1.0, 2.0)
    gt = (1.02, 2.04)
    res, out = run_and_capture(core.metrics.calculate_mae_rmse, pred, gt)
    mae, rmse = res
    assert np.isclose(mae, 30.0)
    assert "[CALL] calculate_mae_rmse" not in out, f"Noisy '[CALL] calculate_mae_rmse' trace was not commented out!"


def test_dryrun_core_metrics_evaluate_file_performance():
    res, out = run_and_capture(core.metrics.evaluate_file_performance, "test_file", (1.0, 2.0), (1.02, 2.04))
    assert res["file_id"] == "test_file"
    assert "[CALL] evaluate_file_performance" not in out, f"Noisy '[CALL] evaluate_file_performance' trace was not commented out!"


def test_dryrun_core_metrics_summarize_benchmark():
    res_list = [{"mae_ms": 10.0, "rmse_ms": 15.0}, {"mae_ms": 20.0, "rmse_ms": 25.0}]
    res, out = run_and_capture(core.metrics.summarize_benchmark, res_list)
    assert res["avg_mae_ms"] == 15.0
    assert "[CALL] summarize_benchmark" not in out, f"Noisy '[CALL] summarize_benchmark' trace was not commented out!"


def test_dryrun_algo_tt1_hodgkinson_cost_function():
    mock_train = [{
        "ste_norm": np.array([0.0, 0.05, 0.0]),
        "frame_times": np.array([0.01, 0.02, 0.03]),
        "gt_boundaries": (0.01, 0.03)
    }]
    res, out = run_and_capture(algorithms.tt1_hodgkinson.hodgkinson_cost_function, 0.02, mock_train)
    assert isinstance(res, float)
    assert "[CALL] hodgkinson_cost_function" not in out, f"Noisy '[CALL] hodgkinson_cost_function' trace was not commented out!"


def test_dryrun_algo_tt1_train_optimal_threshold_tt1():
    res, out = run_and_capture(algorithms.tt1_hodgkinson.train_optimal_threshold_tt1, "TinHieuHuanLuyen", (0.002, 0.003), 5)
    assert isinstance(res, float)
    assert 0.001 <= res <= 0.005
    assert "[CALL] train_optimal_threshold_tt1" not in out, f"Noisy '[CALL] train_optimal_threshold_tt1' trace was not commented out!"


def test_dryrun_algo_tt1_predict_vad_tt1():
    sig = np.zeros(1600, dtype=np.float64)
    sig[400:1200] = 0.5
    res, out = run_and_capture(algorithms.tt1_hodgkinson.predict_vad_tt1, sig, 16000, 0.0025)
    t_s, t_e, ste, times = res
    assert t_s <= t_e
    assert "[CALL] predict_vad_tt1" not in out, f"Noisy '[CALL] predict_vad_tt1' trace was not commented out!"


def test_dryrun_algo_tt2_compute_histogram_100bins():
    ste = np.linspace(0, 1, 500)
    res, out = run_and_capture(algorithms.tt2_histogram.compute_histogram_100bins, ste, 100)
    counts, centers = res
    assert len(counts) == 100
    assert len(centers) == 100
    assert "[CALL] compute_histogram_100bins" not in out, f"Noisy '[CALL] compute_histogram_100bins' trace was not commented out!"


def test_dryrun_algo_tt2_moving_average_smooth():
    hist = np.ones(100, dtype=np.float64)
    res, out = run_and_capture(algorithms.tt2_histogram.moving_average_smooth, hist, 5)
    assert len(res) == 100
    assert "[CALL] moving_average_smooth" not in out, f"Noisy '[CALL] moving_average_smooth' trace was not commented out!"


def test_dryrun_algo_tt2_find_histogram_peaks():
    smoothed = np.zeros(100, dtype=np.float64)
    smoothed[10] = 5.0
    smoothed[60] = 10.0
    centers = np.linspace(0.005, 0.995, 100)
    res, out = run_and_capture(algorithms.tt2_histogram.find_histogram_peaks, smoothed, centers)
    m1, m2 = res
    assert m1 < m2
    assert "[CALL] find_histogram_peaks" not in out, f"Noisy '[CALL] find_histogram_peaks' trace was not commented out!"


def test_dryrun_algo_tt2_compute_adaptive_threshold_tt2():
    ste = np.array([0.001] * 200 + [0.3] * 200, dtype=np.float64)
    res, out = run_and_capture(algorithms.tt2_histogram.compute_adaptive_threshold_tt2, ste, 5.0)
    assert isinstance(res, float)
    assert 0.0 < res < 1.0
    assert "[CALL] compute_adaptive_threshold_tt2" not in out, f"Noisy '[CALL] compute_adaptive_threshold_tt2' trace was not commented out!"


def test_dryrun_algo_tt2_predict_vad_tt2():
    sig = np.zeros(1600, dtype=np.float64)
    sig[400:1200] = 0.5
    res, out = run_and_capture(algorithms.tt2_histogram.predict_vad_tt2, sig, 16000, 5.0)
    t_s, t_e, ste, times, t_adapt = res
    assert t_s <= t_e
    assert "[CALL] predict_vad_tt2" not in out, f"Noisy '[CALL] predict_vad_tt2' trace was not commented out!"


def test_dryrun_algo_tt3_extract_speech_silence_ste_frames():
    res, out = run_and_capture(algorithms.tt3_gaussian.extract_speech_silence_ste_frames, "TinHieuHuanLuyen")
    sil_ste, sp_ste = res
    assert len(sil_ste) > 0 and len(sp_ste) > 0
    assert "[CALL] extract_speech_silence_ste_frames" not in out, f"Noisy '[CALL] extract_speech_silence_ste_frames' trace was not commented out!"


def test_dryrun_algo_tt3_estimate_gaussian_parameters():
    sil = np.random.normal(0.001, 0.0005, 100)
    sp = np.random.normal(0.2, 0.05, 100)
    res, out = run_and_capture(algorithms.tt3_gaussian.estimate_gaussian_parameters, sil, sp)
    mu_sil, s_sil, mu_sp, s_sp = res
    assert mu_sil < mu_sp
    assert "[CALL] estimate_gaussian_parameters" not in out, f"Noisy '[CALL] estimate_gaussian_parameters' trace was not commented out!"


def test_dryrun_algo_tt3_solve_bayes_decision_threshold():
    res, out = run_and_capture(algorithms.tt3_gaussian.solve_bayes_decision_threshold, 0.000359, 0.000715, 0.196747, 0.233472)
    assert 0.001 < res < 0.01
    assert "[CALL] solve_bayes_decision_threshold" not in out, f"Noisy '[CALL] solve_bayes_decision_threshold' trace was not commented out!"


def test_dryrun_algo_tt3_predict_vad_tt3():
    sig = np.zeros(1600, dtype=np.float64)
    sig[400:1200] = 0.5
    res, out = run_and_capture(algorithms.tt3_gaussian.predict_vad_tt3, sig, 16000, 0.002864)
    t_s, t_e, ste, times = res
    assert t_s <= t_e
    assert "[CALL] predict_vad_tt3" not in out, f"Noisy '[CALL] predict_vad_tt3' trace was not commented out!"


def test_dryrun_main_setup_screen_window():
    fig = plt.figure()
    _, out = run_and_capture(main.setup_screen_window, fig, "top_left")
    plt.close(fig)
    assert "[CALL] setup_screen_window" not in out, f"Noisy '[CALL] setup_screen_window' trace was not commented out!"


def test_dryrun_main_plot_vad_result():
    fig, ax = plt.subplots()
    sig = np.zeros(320)
    times = np.linspace(0, 0.02, 10)
    ste = np.zeros(10)
    _, out = run_and_capture(
        main.plot_vad_result,
        ax, sig, 16000, times, ste, (0.005, 0.015), (0.005, 0.015), "Test Title"
    )
    plt.close(fig)
    assert "[CALL] plot_vad_result" not in out, f"Noisy '[CALL] plot_vad_result' trace was not commented out!"


def test_dryrun_main_run_pipeline():
    res, out = run_and_capture(main.run_pipeline, "TinHieuKiemThu", "tt3", False, None)
    assert len(res) == 4
    assert "[CALL] run_pipeline" not in out, f"Noisy '[CALL] run_pipeline' trace was not commented out!"


def test_dryrun_main_entrypoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--no-plot", "--algo", "tt3"])
    _, out = run_and_capture(main.main)
    assert "[CALL] main" not in out, f"Noisy '[CALL] main' trace was not commented out!"


# ---------------------------------------------------------------------------
# Standalone CLI Runner to generate complete diagnostic table
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running interface dry-run diagnostic...")
    # List of tuples: (Function Name, Callable, Args, Kwargs, Expected [CALL] string)
    all_interfaces = [
        ("read_wav", core.io_utils.read_wav, ("TinHieuHuanLuyen/phone_F1.wav",), {}, "[CALL] read_wav"),
        ("read_lab", core.io_utils.read_lab, ("TinHieuHuanLuyen/phone_F1.lab",), {}, "[CALL] read_lab"),
        ("get_speech_groundtruth", core.io_utils.get_speech_groundtruth, ([(0.0, 0.5, "sil"), (0.5, 2.5, "v")],), {}, "[CALL] get_speech_groundtruth"),
        ("frame_signal", core.features.frame_signal, (np.zeros(1600), 16000), {}, "[CALL] frame_signal"),
        ("compute_ste", core.features.compute_ste, (np.ones((2, 320)),), {}, "[CALL] compute_ste"),
        ("normalize_ste", core.features.normalize_ste, (np.array([1.0, 2.0]),), {}, "[CALL] normalize_ste"),
        ("extract_ste_features", core.features.extract_ste_features, (np.zeros(1600), 16000), {}, "[CALL] extract_ste_features"),
        ("apply_threshold", core.postprocess.apply_threshold, (np.array([0.1, 0.5]), 0.2), {}, "[CALL] apply_threshold"),
        ("remove_short_silences_200ms", core.postprocess.remove_short_silences_200ms, (np.array([1, 0, 1]),), {}, "[CALL] remove_short_silences_200ms"),
        ("extract_speech_boundaries", core.postprocess.extract_speech_boundaries, (np.array([0, 1, 0]), np.array([0.01, 0.02, 0.03])), {}, "[CALL] extract_speech_boundaries"),
        ("calculate_mae_rmse", core.metrics.calculate_mae_rmse, ((1.0, 2.0), (1.0, 2.0)), {}, "[CALL] calculate_mae_rmse"),
        ("evaluate_file_performance", core.metrics.evaluate_file_performance, ("test", (1.0, 2.0), (1.0, 2.0)), {}, "[CALL] evaluate_file_performance"),
        ("summarize_benchmark", core.metrics.summarize_benchmark, ([{"mae_ms": 1.0, "rmse_ms": 1.0}],), {}, "[CALL] summarize_benchmark"),
        ("hodgkinson_cost_function", algorithms.tt1_hodgkinson.hodgkinson_cost_function, (0.002, [{"ste_norm": np.array([0.1]), "frame_times": np.array([0.01]), "gt_boundaries": (0.01, 0.01)}]), {}, "[CALL] hodgkinson_cost_function"),
        ("train_optimal_threshold_tt1", algorithms.tt1_hodgkinson.train_optimal_threshold_tt1, ("TinHieuHuanLuyen", (0.002, 0.003), 2), {}, "[CALL] train_optimal_threshold_tt1"),
        ("predict_vad_tt1", algorithms.tt1_hodgkinson.predict_vad_tt1, (np.zeros(1600), 16000), {}, "[CALL] predict_vad_tt1"),
        ("compute_histogram_100bins", algorithms.tt2_histogram.compute_histogram_100bins, (np.linspace(0, 1, 100),), {}, "[CALL] compute_histogram_100bins"),
        ("moving_average_smooth", algorithms.tt2_histogram.moving_average_smooth, (np.ones(100), 5), {}, "[CALL] moving_average_smooth"),
        ("find_histogram_peaks", algorithms.tt2_histogram.find_histogram_peaks, (np.array([0, 1, 0, 2, 0]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])), {}, "[CALL] find_histogram_peaks"),
        ("compute_adaptive_threshold_tt2", algorithms.tt2_histogram.compute_adaptive_threshold_tt2, (np.linspace(0, 1, 100),), {}, "[CALL] compute_adaptive_threshold_tt2"),
        ("predict_vad_tt2", algorithms.tt2_histogram.predict_vad_tt2, (np.zeros(1600), 16000), {}, "[CALL] predict_vad_tt2"),
        ("extract_speech_silence_ste_frames", algorithms.tt3_gaussian.extract_speech_silence_ste_frames, ("TinHieuHuanLuyen",), {}, "[CALL] extract_speech_silence_ste_frames"),
        ("estimate_gaussian_parameters", algorithms.tt3_gaussian.estimate_gaussian_parameters, (np.array([0.001, 0.002]), np.array([0.1, 0.2])), {}, "[CALL] estimate_gaussian_parameters"),
        ("solve_bayes_decision_threshold", algorithms.tt3_gaussian.solve_bayes_decision_threshold, (0.000359, 0.000715, 0.196747, 0.233472), {}, "[CALL] solve_bayes_decision_threshold"),
        ("predict_vad_tt3", algorithms.tt3_gaussian.predict_vad_tt3, (np.zeros(1600), 16000), {}, "[CALL] predict_vad_tt3"),
        ("setup_screen_window", main.setup_screen_window, (plt.figure(), "top_left"), {}, "[CALL] setup_screen_window"),
        ("plot_vad_result", main.plot_vad_result, (plt.subplots()[1], np.zeros(320), 16000, np.zeros(1), np.zeros(1), (0.0, 0.01), (0.0, 0.01), "title"), {}, "[CALL] plot_vad_result"),
        ("run_pipeline", main.run_pipeline, ("TinHieuKiemThu", "tt3", False, None), {}, "[CALL] run_pipeline"),
        ("main", main.main, (), {}, "[CALL] main"),
    ]

    print(f"\nTotal interface functions evaluated: {len(all_interfaces)}")
    failures = []
    passes = []

    # Backup sys.argv for main()
    orig_argv = sys.argv
    sys.argv = ["main.py", "--no-plot", "--algo", "tt3"]

    for name, fn, args, kwargs, expected_trace in all_interfaces:
        try:
            _, out = run_and_capture(fn, *args, **kwargs)
            has_trace = expected_trace in out
            if not has_trace:
                passes.append(name)
                print(f"[PASS] {name:<35} -> cleanly executed without noisy trace")
            else:
                failures.append((name, expected_trace, out))
                print(f"[FAIL] {name:<35} -> noisy trace present: {expected_trace!r}")
        except Exception as e:
            failures.append((name, expected_trace, f"CRASH: {e}"))
            print(f"[CRASH] {name:<35} -> Exception: {e}")

    sys.argv = orig_argv

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(passes)} PASSED, {len(failures)} FAILED")
    print("=" * 80)
