"""数据处理模块：窗口构造、时间顺序切分、指标计算。

这些函数是 02/03b/04/05 共用的基础逻辑。把它们收敛到本模块，
可以保证各脚本的切分口径完全一致，并由单元测试守护正确性。
"""

from __future__ import annotations

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


def make_windows(
    series: np.ndarray,
    input_len: int,
    pred_len: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """把时间序列切成 (输入窗口, 预测标签) 样本对。

    **这是时序预测最容易出错的地方**，务必理解切分逻辑::

        下标:  0  1  2 ... 95 | 96 97 ... 119 | 120 ...
               └── 输入 X (96) ┘└── 标签 y (24) ┘

        X[i] = series[i : i + input_len]
        y[i] = series[i + input_len : i + input_len + pred_len]

    两个必须守住的不变量：
        1. **标签首点紧跟输入末点**，不重叠也不留空档；
        2. **绝不使用未来信息** —— 归一化的 scaler 只能在训练集上 fit
           （见 :func:`chronological_split` 的说明）。

    Args:
        series: 形状为 ``(T, F)`` 或 ``(T,)`` 的时间序列，建议已归一化。
        input_len: 输入窗口长度（历史步数）。
        pred_len: 预测长度（未来步数）。
        stride: 滑窗步长，``>1`` 可减少样本量。

    Returns:
        ``(X, y)``：``X`` 形状 ``(N, input_len, F)``，``y`` 形状 ``(N, pred_len)``。
        ``F`` 为特征数；若输入是 1 维则 ``F=1``。标签默认取第 0 列。

    Raises:
        ValueError: 参数非正，或序列长度不足一个窗口。

    Example:
        >>> s = np.arange(200, dtype=float).reshape(-1, 1)
        >>> X, y = make_windows(s, input_len=96, pred_len=24)
        >>> X.shape, y.shape
        ((81, 96, 1), (81, 24))
    """
    if input_len <= 0 or pred_len <= 0 or stride <= 0:
        raise ValueError("input_len / pred_len / stride 必须为正整数")

    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    total = len(series)
    window = input_len + pred_len
    if total < window:
        raise ValueError(
            f"序列长度 {total} 不足以构造一个窗口（需要至少 {window}）。"
            "请检查数据是否加载完整，或调小 input_len/pred_len。"
        )

    n_samples = (total - window) // stride + 1
    X = np.empty((n_samples, input_len, series.shape[1]), dtype=np.float32)
    y = np.empty((n_samples, pred_len), dtype=np.float32)

    for i in range(n_samples):
        start = i * stride
        X[i] = series[start : start + input_len]
        # 标签取第 0 列（本项目特征列 0 即目标 TOTAL_AC_POWER）
        y[i] = series[start + input_len : start + window, 0]

    logger.info(
        "滑窗构造完成：%d 个样本（input_len=%d, pred_len=%d, stride=%d）",
        n_samples,
        input_len,
        pred_len,
        stride,
    )
    return X, y


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """按时间顺序切分数据集。

    **时序数据绝不能随机切分**：随机切分会让未来的样本混入训练集，
    造成"测试集指标虚高"的假象 —— 这是时序预测论文里最常见的硬伤。

    Args:
        X: 输入样本，形状 ``(N, input_len, F)``。
        y: 标签，形状 ``(N, pred_len)``。
        train_ratio: 训练集占比。
        val_ratio: 验证集占比（测试集为剩余部分）。

    Returns:
        形如 ``{"train": (X, y), "val": (X, y), "test": (X, y)}`` 的字典。

    Raises:
        ValueError: 比例之和 >= 1，或切分后某集合为空。

    Example:
        >>> X = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1)
        >>> y = np.arange(1000, dtype=np.float32).reshape(1000, 1)
        >>> splits = chronological_split(X, y)
        >>> [len(splits[k][0]) for k in ("train", "val", "test")]
        [700, 150, 150]
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio({train_ratio}) + val_ratio({val_ratio}) 必须小于 1")

    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        "test": (X[n_train + n_val :], y[n_train + n_val :]),
    }
    for name, (xs, _ys) in splits.items():
        if len(xs) == 0:
            raise ValueError(f"{name} 集为空，请调整切分比例或增加数据量")
        logger.info("%s 集：%d 个样本", name, len(xs))
    return splits


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算回归评价指标（输入为逆变换后的真实物理值）。

    Args:
        y_true: 真实值。
        y_pred: 预测值。

    Returns:
        ``MAE`` / ``RMSE`` / ``MAPE`` / ``R2`` 四项指标字典。
        MAPE 的百分比保护：真实值接近 0 时该样本不计入分母，
        否则夜间零功率会把 MAPE 顶到无穷大。

    Example:
        >>> m = compute_metrics(np.array([1.0, 2.0]), np.array([1.1, 1.8]))
        >>> round(m["MAE"], 3)
        0.15
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # MAPE：仅统计真实值 > 1% 最大值的样本，避免除以零与夜间的无穷百分比
    threshold = np.max(np.abs(y_true)) * 0.01
    mask = np.abs(y_true) > threshold
    mape = (
        float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100)
        if mask.any()
        else float("nan")
    )

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def naive_persistence_predict(X_test: np.ndarray, pred_len: int) -> np.ndarray:
    """持续预测基线：把输入窗口最后一步的功率值重复 pred_len 次。

    这是时序预测最低的门槛基线 —— 任何有意义的模型都必须优于它。
    若模型连持续预测都打不过，说明它什么都没学到。

    Args:
        X_test: 测试输入，形状 ``(N, input_len, F)``，特征 0 为功率。
        pred_len: 预测步数。

    Returns:
        形状 ``(N, pred_len)`` 的预测值（与 X 同量纲，尚未逆变换也无妨，
        因为持续预测是恒等映射）。
    """
    last_power = np.asarray(X_test)[:, -1, 0]
    return np.repeat(last_power.reshape(-1, 1), pred_len, axis=1)
