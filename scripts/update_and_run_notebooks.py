"""
Script to update TT1 and TT2 notebooks with robust local path loading and IPython.display.Audio widgets,
then execute them to pre-compute and verify the embedded audio players.
"""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def update_and_run_tt1(notebook_path: Path, is_vietnamese: bool = True):
    print(f"[*] Processing {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Update Cell 0: robust paths and Audio import
    if is_vietnamese:
        nb.cells[0].source = '''# Cell 1: Thư viện và Cấu hình tham số hệ thống
from pathlib import Path
import wave
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Tìm kiếm thư mục gốc dự án linh hoạt và nạp dữ liệu cục bộ
def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "TinHieuHuanLuyen").exists() and (candidate / "TinHieuKiemThu").exists():
            return candidate
    return Path.cwd()

PROJECT_ROOT = find_project_root()
TRAIN_DIR = PROJECT_ROOT / "TinHieuHuanLuyen"
TEST_DIR = PROJECT_ROOT / "TinHieuKiemThu"
OUTPUT_DIR = PROJECT_ROOT / "src" / "TT1" / "output" / "mine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình tham số DSP theo yêu cầu giảng viên
FRAME_MS = 25        # Độ dài frame (ms)
HOP_MS = 10          # Bước trượt frame (ms)
MIN_SILENCE_MS = 200 # Khoảng lặng tối thiểu (ms) để lọc khoảng lặng ảo
'''
    else:
        nb.cells[0].source = '''# Cell 1: Libraries and System Parameter Configuration
from pathlib import Path
import wave
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Dataset directory paths
def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "TinHieuHuanLuyen").exists() and (candidate / "TinHieuKiemThu").exists():
            return candidate
    return Path.cwd()

PROJECT_ROOT = find_project_root()
TRAIN_DIR = PROJECT_ROOT / "TinHieuHuanLuyen"
TEST_DIR = PROJECT_ROOT / "TinHieuKiemThu"
OUTPUT_DIR = PROJECT_ROOT / "src" / "TT1" / "output" / "main"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DSP parameters according to project specifications
FRAME_MS = 25        # Frame length in milliseconds (ms)
HOP_MS = 10          # Frame hop size in milliseconds (ms)
MIN_SILENCE_MS = 200 # Minimum silence duration (ms) to filter spurious pauses
'''

    # 2. Update Cell 4: Plot test results and show Audio players
    cell4_src = nb.cells[4].source
    if is_vietnamese:
        old_plt = '    plt.tight_layout()\n    fig_path = OUTPUT_DIR / f"{Path(res[\'file\']).stem}.png"\n    plt.savefig(fig_path, dpi=150)\n    print(f"Đã xuất và lưu: {fig_path}")\n    plt.show()'
        new_plt = '''    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{Path(res['file']).stem}.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Đã xuất và lưu: {fig_path}")
    plt.show()

    # Nạp và hiển thị audio player cho file âm thanh cục bộ
    wav_file_path = TEST_DIR / res['file']
    print(f"🎵 Audio gốc ({res['file']}):")
    if wav_file_path.exists():
        display(Audio(filename=str(wav_file_path)))
    else:
        display(Audio(data=res['signal'], rate=res['fs']))

    # Hiển thị audio player cho phân đoạn tiếng nói nhận diện được
    st_sample = int(res['pred'][0] * res['fs'])
    en_sample = min(len(res['signal']), int(res['pred'][1] * res['fs']))
    if en_sample > st_sample:
        print(f"🔊 Đoạn tiếng nói nhận diện [{res['pred'][0]:.2f}s - {res['pred'][1]:.2f}s]:")
        display(Audio(data=res['signal'][st_sample:en_sample], rate=res['fs']))'''
    else:
        old_plt = '    plt.tight_layout()\n    fig_path = OUTPUT_DIR / f"{Path(res[\'file\']).stem}.png"\n    plt.savefig(fig_path, dpi=150)\n    print(f"Exported and saved figure: {fig_path}")\n    plt.show()'
        new_plt = '''    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"{Path(res['file']).stem}.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Exported and saved figure: {fig_path}")
    plt.show()

    # Load and display local audio player widget
    wav_file_path = TEST_DIR / res['file']
    print(f"🎵 Original Audio ({res['file']}):")
    if wav_file_path.exists():
        display(Audio(filename=str(wav_file_path)))
    else:
        display(Audio(data=res['signal'], rate=res['fs']))

    # Display audio player for predicted speech segment
    st_sample = int(res['pred'][0] * res['fs'])
    en_sample = min(len(res['signal']), int(res['pred'][1] * res['fs']))
    if en_sample > st_sample:
        print(f"🔊 Predicted Speech Segment [{res['pred'][0]:.2f}s - {res['pred'][1]:.2f}s]:")
        display(Audio(data=res['signal'][st_sample:en_sample], rate=res['fs']))'''

    if old_plt in cell4_src:
        nb.cells[4].source = cell4_src.replace(old_plt, new_plt)
    else:
        print(f"[!] Warning: old_plt pattern not found in Cell 4 of {notebook_path}")

    # 3. Execute notebook
    print(f"[*] Executing {notebook_path}...")
    client = NotebookClient(nb, timeout=60, kernel_name="python3")
    client.execute()

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[V] Successfully updated and executed {notebook_path}!")


def update_and_run_tt2(notebook_path: Path):
    print(f"[*] Processing {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Update Cell 0: robust paths and Audio import
    nb.cells[0].source = '''# Cell 1: Imports and configuration
from dataclasses import dataclass
from pathlib import Path
import wave

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from IPython.display import Audio, display

def find_project_root():
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "TinHieuHuanLuyen").exists() and (candidate / "TinHieuKiemThu").exists():
            return candidate
    return Path.cwd()

PROJECT_ROOT = find_project_root()
TRAIN_DIR = PROJECT_ROOT / "TinHieuHuanLuyen"
TEST_DIR = PROJECT_ROOT / "TinHieuKiemThu"
OUTPUT_DIR = PROJECT_ROOT / "src" / "TT2" / "output" / "mine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DSP parameters 
FRAME_MS = 25         # frame length (ms)
HOP_MS = 10           # frame hop (ms)
MIN_SILENCE_MS = 200  

# Boundary padding in frames (1 frame = HOP_MS = 10 ms).
# parameter: chosen by sweeping on TRAIN only (see Cell 7)
PAD_START = 1
PAD_END = 1'''

    # 2. Update Cell 5: Plot test results and show Audio players
    cell5_src = nb.cells[5].source
    old_end = '''    fig_path = OUTPUT_DIR / f"{Path(sample.name).stem}.png"\n    fig.savefig(fig_path, dpi=150)\n    print(f"Saved figure: {fig_path}")\n    plt.show()'''
    new_end = '''    fig_path = OUTPUT_DIR / f"{Path(sample.name).stem}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure: {fig_path}")
    plt.show()

    # Nạp và hiển thị audio player cho file kiểm thử cục bộ
    wav_file_path = TEST_DIR / sample.name
    print(f"🎵 Audio player cho file kiểm thử: {sample.name}")
    if wav_file_path.exists():
        display(Audio(filename=str(wav_file_path)))
    else:
        display(Audio(data=sample.signal, rate=sample.fs))

    # Audio player cho đoạn tiếng nói đã phân tách
    st_idx = int(result.start * sample.fs)
    en_idx = min(len(sample.signal), int(result.end * sample.fs))
    if en_idx > st_idx:
        print(f"🔊 Đoạn tiếng nói nhận diện [{result.start:.2f}s - {result.end:.2f}s]:")
        display(Audio(data=sample.signal[st_idx:en_idx], rate=sample.fs))'''

    if old_end in cell5_src:
        nb.cells[5].source = cell5_src.replace(old_end, new_end)
    else:
        print(f"[!] Warning: old_end pattern not found in Cell 5 of {notebook_path}")

    # 3. Execute notebook
    print(f"[*] Executing {notebook_path}...")
    client = NotebookClient(nb, timeout=60, kernel_name="python3")
    client.execute()

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[V] Successfully updated and executed {notebook_path}!")


if __name__ == "__main__":
    update_and_run_tt1(PROJECT_ROOT / "src" / "TT1" / "1-mine.ipynb", is_vietnamese=True)
    update_and_run_tt1(PROJECT_ROOT / "src" / "TT1" / "main.ipynb", is_vietnamese=False)
    update_and_run_tt2(PROJECT_ROOT / "src" / "TT2" / "1-mine.ipynb")
