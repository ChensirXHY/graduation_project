"""统一日志模块。

训练脚本动辄跑几十分钟，``print`` 无法分级、无法落盘，
出问题后没有现场可查。本模块提供全项目统一的 logger。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> None:
    """初始化根 logger。进程内只需调用一次，重复调用会被忽略。

    Args:
        level: 日志级别，如 ``"DEBUG"``/``"INFO"``。
        log_to_file: 是否写入 ``logs/train.log``。
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    if log_to_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        # encoding="utf-8" 必需：Windows 下写入中文日志否则会抛 UnicodeEncodeError
        handler = logging.FileHandler(_LOG_DIR / "train.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
        root.addHandler(handler)

    # 压掉第三方库的冗余日志，让训练进度清晰可读
    for noisy in ("matplotlib", "PIL", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取统一格式的 logger。

    Args:
        name: 通常传 ``__name__``。

    Returns:
        配置好的 Logger。
    """
    setup_logging()
    return logging.getLogger(name)
