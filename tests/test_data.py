"""单元测试：滑动窗口、时间顺序切分、指标计算。

这些测试**不需要真实数据集**（大 CSV 与 npz 已移出版本控制），
用合成序列即可验证逻辑正确性，因此可以在 CI 里秒级跑完。

运行：
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import (
    chronological_split,
    compute_metrics,
    make_windows,
    naive_persistence_predict,
)


class TestMakeWindows:
    """滑动窗口构造的正确性。"""

    def test_output_shape(self) -> None:
        """形状应符合 (N, input_len, F) 与 (N, pred_len)。"""
        series = np.arange(200, dtype=np.float32).reshape(-1, 1)

        X, y = make_windows(series, input_len=96, pred_len=24)

        assert X.shape == (81, 96, 1)
        assert y.shape == (81, 24)

    def test_no_future_leakage(self) -> None:
        """核心不变量：输入只含过去，标签紧跟其后，二者不重叠。

        这是时序预测最危险的错误来源。一旦窗口与标签错位或重叠，
        模型会在训练时"看到答案"，测试指标虚高而实际毫无预测能力。
        """
        series = np.arange(120, dtype=np.float32).reshape(-1, 1)

        X, y = make_windows(series, input_len=96, pred_len=24)

        assert X[0, 0, 0] == 0, "输入窗口应从序列起点开始"
        assert X[0, -1, 0] == 95, "输入窗口最后一个点"
        assert y[0, 0] == 96, "标签首点应紧随输入末点，不留空档"
        assert y[0, -1] == 119, "标签末点"

    def test_windows_are_contiguous(self) -> None:
        """相邻样本的输入窗口应恰好相差 stride 步。"""
        series = np.arange(300, dtype=np.float32).reshape(-1, 1)

        X, _ = make_windows(series, input_len=50, pred_len=10, stride=5)

        for i in range(len(X) - 1):
            assert X[i + 1, 0, 0] - X[i, 0, 0] == 5

    def test_stride_reduces_samples(self) -> None:
        """增大 stride 应减少样本数。"""
        series = np.arange(500, dtype=np.float32).reshape(-1, 1)

        n_stride1 = len(make_windows(series, 96, 24, stride=1)[0])
        n_stride4 = len(make_windows(series, 96, 24, stride=4)[0])

        assert n_stride4 < n_stride1

    def test_1d_input_is_promoted(self) -> None:
        """一维输入应被自动升维为 (T, 1)。"""
        series = np.arange(150, dtype=np.float32)

        X, y = make_windows(series, input_len=96, pred_len=24)

        assert X.ndim == 3 and X.shape[2] == 1
        assert y.ndim == 2

    def test_multifeature_output_uses_first_column(self) -> None:
        """多特征输入时，标签应取第 0 列（本项目第 0 列即目标功率）。"""
        series = np.zeros((200, 3), dtype=np.float32)
        series[:, 0] = np.arange(200)  # 目标列
        series[:, 1] = 999.0  # 干扰列

        _, y = make_windows(series, input_len=96, pred_len=24)

        assert y[0, 0] == 96, "标签应来自第 0 列而非干扰列"

    def test_short_series_raises(self) -> None:
        """序列长度不足一个窗口时应显式报错，而非静默返回空数组。"""
        series = np.arange(50, dtype=np.float32).reshape(-1, 1)

        with pytest.raises(ValueError, match="不足以构造一个窗口"):
            make_windows(series, input_len=96, pred_len=24)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_params_raise(self, bad: int) -> None:
        """非正的 input_len/pred_len/stride 应被拒绝。"""
        series = np.arange(300, dtype=np.float32).reshape(-1, 1)

        with pytest.raises(ValueError, match="必须为正整数"):
            make_windows(series, input_len=bad, pred_len=24)


class TestChronologicalSplit:
    """按时间切分的正确性。"""

    def test_split_sizes_and_order(self) -> None:
        """切分应保持时间顺序且不重叠不遗漏。"""
        X = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1)
        y = np.arange(1000, dtype=np.float32).reshape(1000, 1)

        splits = chronological_split(X, y, train_ratio=0.7, val_ratio=0.15)

        assert len(splits["train"][0]) == 700
        assert len(splits["val"][0]) == 150
        assert len(splits["test"][0]) == 150

    def test_no_shuffle_keeps_temporal_order(self) -> None:
        """训练集必须完全早于验证集，验证集完全早于测试集。

        若此处失败，说明发生了数据泄漏，测试指标将不可信。
        """
        X = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1)
        y = np.arange(1000, dtype=np.float32).reshape(1000, 1)

        splits = chronological_split(X, y)

        assert splits["train"][0][-1, 0, 0] < splits["val"][0][0, 0, 0]
        assert splits["val"][0][-1, 0, 0] < splits["test"][0][0, 0, 0]

    def test_all_samples_used_once(self) -> None:
        """三个集合应恰好覆盖全部样本，无重复无遗漏。"""
        X = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1)
        y = np.arange(1000, dtype=np.float32).reshape(1000, 1)

        splits = chronological_split(X, y)
        collected = np.concatenate(
            [splits["train"][0], splits["val"][0], splits["test"][0]]
        ).ravel()

        np.testing.assert_array_equal(collected, np.arange(1000))

    def test_invalid_ratios_raise(self) -> None:
        """比例之和 >= 1 应报错。"""
        X = np.zeros((100, 1, 1), dtype=np.float32)
        y = np.zeros((100, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="必须小于 1"):
            chronological_split(X, y, train_ratio=0.9, val_ratio=0.2)


class TestComputeMetrics:
    """指标计算的正确性。"""

    def test_perfect_prediction(self) -> None:
        """完美预测时 MAE/RMSE 为 0，R2 为 1。"""
        y = np.array([10.0, 20.0, 30.0])

        m = compute_metrics(y, y)

        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0
        assert m["R2"] == pytest.approx(1.0)

    def test_mae_rmse_values(self) -> None:
        """已知误差下的解析值。"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.8, 3.3])

        m = compute_metrics(y_true, y_pred)

        assert m["MAE"] == pytest.approx(0.2)
        # 误差 [-0.1, 0.2, -0.3] → MSE = (0.01+0.04+0.09)/3 = 0.04667
        assert m["RMSE"] == pytest.approx(0.2160, abs=1e-3)

    def test_mape_ignores_near_zero_targets(self) -> None:
        """真实值接近 0 的样本不计入 MAPE（夜间零功率保护）。"""
        y_true = np.array([0.0, 100.0, 200.0, 300.0])
        y_pred = np.array([50.0, 110.0, 190.0, 330.0])
        # 有效样本只有 [100, 200, 300]：误差 [10, 10, 30] → MAPE = (0.1+0.05+0.1)/3

        m = compute_metrics(y_true, y_pred)

        assert m["MAPE"] == pytest.approx(8.333, abs=0.01)


class TestNaivePersistence:
    """持续预测基线的正确性。"""

    def test_repeats_last_value(self) -> None:
        """输出应把输入窗口最后一步的功率重复 pred_len 次。"""
        X = np.zeros((5, 96, 3), dtype=np.float32)
        X[:, -1, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]

        preds = naive_persistence_predict(X, pred_len=24)

        assert preds.shape == (5, 24)
        np.testing.assert_array_equal(preds[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(preds[0], np.full(24, 1.0))
