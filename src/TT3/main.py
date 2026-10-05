"""
Chương trình chuyển tiếp: src/TT3/main.py
Cho phép khởi chạy chương trình TT3 trực tiếp từ thư mục src/TT3 hoặc từ thư mục gốc dự án.
"""

import sys
from pathlib import Path

CURR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURR_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURR_DIR))
sys.path.insert(0, str(CURR_DIR / "src"))

try:
    from src.TT3.src.main import main
except (ImportError, ModuleNotFoundError):
    try:
        from TT3.src.main import main
    except (ImportError, ModuleNotFoundError):
        from src.main import main

if __name__ == "__main__":
    main()
