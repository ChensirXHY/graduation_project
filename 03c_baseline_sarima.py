"""SARIMA 基线模型（滚动 24 步预测）。

任务书明确要求「与 SARIMA、LSTM 等基线模型对比分析」，本脚本补齐该基线。

方法说明（与深度学习模型公平对齐的关键）：
    - 只对 train（< 2020-06-07）拟合 SARIMA 参数，与 DL 模型完全一致；
    - 验证段（06-07 ~ 06-12）只用于 append 推进卡尔曼状态，不参与参数估计
      （对应 DL 模型里"验证集只用于早停、不参与训练"的口径）；
    - 测试段（>= 06-12）做滚动 24 步预测：每个窗口用截至该窗口的历史，
      get_forecast(24) 一步预测 24 个未来点，再把该窗口第一个真实点 append 进来
      推进状态 —— 全程不使用未来信息。

季节周期 S=96：采样间隔 15 分钟，一天 = 96 个点，光伏功率有强日周期。
取 (1,0,1)×(0,1,0,96)：季节差分 D=1 消除日周期（本质是"季节随机游走"，
即预测 = 昨天同时刻的值 + ARMA(1,1) 对 24 小时差的修正），非季节部分用
ARMA(1,1) 刻画日内短期相关。

为什么不做季节 AR/MA（如 (1,1,1,96)）：季节 AR/MA 在滞后 96 处会把
状态维度撑到 96+，滚动预测的卡尔曼滤波慢两个数量级（实测 40 分钟以上），
而 (0,1,0,96) 状态维度仅约 3，秒级完成、且对日周期的刻画几乎等效。

输出：results/sarima_test_predictions.npy（真实 kW），
      results/sarima_prediction_sample.png。

运行：
    python 03c_baseline_sarima.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 允许以 python 03c_baseline_sarima.py 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import compute_metrics  # noqa: E402
from src.logger import get_logger  # noqa: E402

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

logger = get_logger(__name__)

CLEAN_CSV = Path("archive/plant_1_cleaned.csv")
NPZ = Path("archive/scaled_data.npz")
SCALER_Y = Path("archive/scaler_y.pkl")
INPUT_LEN = 96  # 历史 24 小时
PRED_LEN = 24  # 预测 6 小时
SEASONAL = 96  # 日周期（15 分钟采样）

# 与 02 脚本完全一致的切分点
TRAIN_END = pd.Timestamp("2020-06-07 00:00:00")
VAL_END = pd.Timestamp("2020-06-12 00:00:00")


def load_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载清洗后的功率序列，按 02 脚本口径切分 train/val/test。

    Returns:
        ``(train, val, test)`` 三个一维 numpy 数组（真实 kW）。
    """
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"清洗数据不存在：{CLEAN_CSV}\n"
            "  修复：先运行 01_explore_quality.py（需要 Kaggle 原始数据）。"
        )
    df = pd.read_csv(CLEAN_CSV)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df = df.sort_values("DATE_TIME").reset_index(drop=True)

    train = df[df["DATE_TIME"] < TRAIN_END]["TOTAL_AC_POWER"].values.astype(float)
    val = df[(df["DATE_TIME"] >= TRAIN_END) & (df["DATE_TIME"] < VAL_END)][
        "TOTAL_AC_POWER"
    ].values.astype(float)
    test = df[df["DATE_TIME"] >= VAL_END]["TOTAL_AC_POWER"].values.astype(float)
    logger.info("序列加载：train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def fit_sarima(train: np.ndarray):
    """拟合 SARIMA(1,0,1)×(1,1,1,96)。

    关闭 enforce_stationarity/invertibility：季节差分 D=1 后模型可能不严格平稳，
    且允许温和的数值近似 —— 这是序列基线拟合的常见做法，不影响预测。
    """
    logger.info("拟合 SARIMA(1,0,1)x(0,1,0,%d)，样本数 %d ...", SEASONAL, len(train))
    t0 = time.perf_counter()
    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(0, 1, 0, SEASONAL),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    logger.info(
        "拟合完成（%.1fs）。AIC=%.1f, BIC=%.1f",
        time.perf_counter() - t0,
        result.aic,
        result.bic,
    )
    return result


def rolling_forecast(result, val: np.ndarray, test: np.ndarray, n_windows: int) -> np.ndarray:
    """滚动 24 步预测测试段。

    Args:
        result: 在 train 上拟合好的结果对象。
        val: 验证段序列（只 append，不 refit）。
        test: 测试段序列（长度应 = n_windows + INPUT_LEN + PRED_LEN - 1）。
        n_windows: 测试窗口数。

    Returns:
        形状 ``(n_windows, PRED_LEN)`` 的真实预测值。
    """
    expected = n_windows + INPUT_LEN + PRED_LEN - 1
    if len(test) != expected:
        raise ValueError(
            f"测试段长度 {len(test)} 与窗口数 {n_windows} 不匹配（期望 {expected}）。"
            "请确认切分点与 02 脚本一致。"
        )

    # 用验证段推进状态，弥合 train 末尾与 test 开头之间 5 天的空档
    res = result
    if len(val):
        logger.info("append 验证段 %d 点推进状态（不 refit）", len(val))
        res = res.append(val, refit=False)

    preds = []
    t0 = time.perf_counter()
    for i in range(n_windows):
        fc = res.get_forecast(steps=PRED_LEN).predicted_mean
        preds.append(fc.values if hasattr(fc, "values") else np.asarray(fc))
        # 推进一个真实点（该窗口输出的第一步），为下一个窗口提供正确历史
        res = res.append(test[i + INPUT_LEN : i + INPUT_LEN + 1], refit=False)
        if (i + 1) % 100 == 0 or i == n_windows - 1:
            logger.info("  滚动预测 %d/%d", i + 1, n_windows)

    preds = np.asarray(preds, dtype=float)
    logger.info("滚动预测完成（%.1fs），形状 %s", time.perf_counter() - t0, preds.shape)
    return preds


def evaluate(preds: np.ndarray) -> None:
    """与 y_test 对齐后计算指标，保存预测文件与样本图。"""
    data = np.load(NPZ, allow_pickle=True)
    y_test = data["y_test"].astype(np.float32)
    with open(SCALER_Y, "rb") as fh:
        scaler_y = pickle.load(fh)
    y_real = scaler_y.inverse_transform(y_test)

    if preds.shape != y_real.shape:
        raise ValueError(f"预测形状 {preds.shape} 与 y_real {y_real.shape} 不一致")

    metrics = compute_metrics(y_real, preds)
    logger.info(
        "测试集 MAE: %.2f kW, RMSE: %.2f kW, MAPE: %.2f%%, R2: %.3f",
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MAPE"],
        metrics["R2"],
    )

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sarima_test_predictions.npy", preds)
    logger.info("预测已保存至 results/sarima_test_predictions.npy")

    # 样本对比图
    idx = 0
    future_hours = np.arange(0, PRED_LEN * 15, 15) / 60.0
    plt.figure(figsize=(8, 3.5))
    plt.plot(future_hours, y_real[idx], "o-", label="真实值")
    plt.plot(future_hours, preds[idx], "s--", label="SARIMA 预测")
    plt.xlabel("未来时间 (小时)")
    plt.ylabel("交流功率 (kW)")
    plt.title(f"SARIMA 预测样本 (测试集第 {idx + 1} 个窗口)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sarima_prediction_sample.png", dpi=150)
    plt.close()
    logger.info("样本图已保存至 results/sarima_prediction_sample.png")


def main() -> int:
    """主流程：加载 → 拟合 → 滚动预测 → 评估。"""
    train, val, test = load_series()
    n_windows = np.load(NPZ, allow_pickle=True)["X_test"].shape[0]

    result = fit_sarima(train)
    preds = rolling_forecast(result, val, test, n_windows)
    evaluate(preds)
    logger.info("SARIMA 基线完成。下一步运行 05_evaluate_compare.py 汇总对比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
