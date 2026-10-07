"""pytest 全局配置。

作用只有一个：把项目根目录插到 ``sys.path`` 首位。

为什么需要它：
    本项目是"脚本式布局"（编号脚本在根目录、业务代码在 src/），
    不是可安装包。CI 里以 ``pytest tests/`` 方式运行（非
    ``python -m pytest``）时，当前工作目录不会进入 sys.path，
    测试里的 ``from src.data import ...`` 会直接 ImportError。
    本地用 ``python -m pytest`` 能跑通正是因为 -m 会把 CWD 加入
    sys.path —— 这个差异只在 Linux CI 上暴露。
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/conftest.py → tests/ → 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
