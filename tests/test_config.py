"""单元测试：YAML 配置加载。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import load_config


def test_load_valid_config(tmp_path: Path) -> None:
    """合法 YAML 应能加载并返回字典。"""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(
        "model:\n  d_model: 64\n  nhead: 4\ntraining:\n  epochs: 100\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_file)

    assert cfg["model"]["d_model"] == 64
    assert cfg["training"]["epochs"] == 100


def test_missing_file_raises(tmp_path: Path) -> None:
    """不存在的配置文件应抛出含修复建议的 FileNotFoundError。"""
    missing = tmp_path / "not_exist.yaml"

    with pytest.raises(FileNotFoundError, match="配置文件不存在"):
        load_config(missing)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    """YAML 语法错误应抛出 yaml.YAMLError。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: [unclosed", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_config(bad)


def test_non_mapping_top_level_raises(tmp_path: Path) -> None:
    """顶层不是键值映射时应给出明确报错（例如 YAML 顶层是列表）。"""
    wrong = tmp_path / "list.yaml"
    wrong.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="顶层必须是键值映射"):
        load_config(wrong)


def test_real_config_loads() -> None:
    """仓库自带的 configs/base.yaml 必须能通过解析。

    这一条守护的是"配置与代码接口同步"：若有人改了 base.yaml 的
    结构但没改代码（或反过来），这个测试会第一时间报出来。
    """
    cfg = load_config("configs/base.yaml")

    assert (
        cfg["model"]["d_model"] % cfg["model"]["nhead"] == 0
    ), "d_model 必须能被 nhead 整除，否则 TransformerModel 构造会失败"
    assert cfg["data"]["input_len"] > 0
    assert cfg["data"]["pred_len"] > 0
    assert cfg["training"]["epochs"] > 0
