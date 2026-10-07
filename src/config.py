"""YAML 配置加载模块。

把超参从脚本里挪到 ``configs/base.yaml`` 的好处：
    1. 消融实验只改配置文件，不用动代码、不会忘记改回了哪里；
    2. 每次实验的参数随结果一起归档，保证可复现；
    3. 写论文时"实验设置"一节可以直接引用配置文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件。

    Args:
        path: 配置文件路径，相对路径以项目根目录为基准。

    Returns:
        配置字典。键名与 :file:`configs/base.yaml` 的注释一一对应。

    Raises:
        FileNotFoundError: 配置文件不存在。
        yaml.YAMLError: YAML 语法错误（错误信息包含行号）。

    Example:
        >>> cfg = load_config("configs/base.yaml")
        >>> cfg["model"]["d_model"]
        64
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    if not path.exists():
        raise FileNotFoundError(
            f"配置文件不存在：{path}\n" "  修复：从版本库恢复 configs/base.yaml，或用绝对路径指定。"
        )

    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误：{path} 顶层必须是键值映射")

    logger.debug("已加载配置 %s（%d 个顶层节）", path.name, len(config))
    return config
