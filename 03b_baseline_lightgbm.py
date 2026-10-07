"""LightGBM 基线模型训练与评估（多步预测）。

简历中声称"对比基线 LightGBM"，本脚本让这个对比可复现：
    - 输入：把 (N, 96, 12) 的窗口展平为 (N, 1152) 的特征向量；
    - 模型：MultiOutputRegressor(LGBMRegressor)，24 个输出各一个回归树模型；
    - 输出：models/lightgbm_baseline.pkl（权重不入库）、
            results/lightgbm_test_predictions.npy（逐点预测，供 05 对比）。

注意：LightGBM 不擅长多步时序外推（它学的是"输入窗口统计量 → 未来均值"），
这正是它与 LSTM/Transformer 的对比价值所在 —— 基线就该比主模型弱，
否则说明主模型没有从序列结构中获益。

运行：
    python 03b_baseline_lightgbm.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# 允许以 python 03b_baseline_lightgbm.py 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config  # noqa: E402
from src.data import compute_metrics  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)
cfg = load_config("configs/base.yaml")


def load_windowed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载窗口数据并展平特征维度（LightGBM 只接受 2D 输入）。"""
    npz_path = Path(cfg["data"]["npz"])
    if not npz_path.exists():
        raise FileNotFoundError(
            f"数据集不存在：{npz_path}\n"
            "  修复：先运行 01_explore_quality.py 与 02_preprocess_windows.py。"
        )
    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X_train"].reshape(data["X_train"].shape[0], -1)
    X_val = data["X_val"].reshape(data["X_val"].shape[0], -1)
    X_test = data["X_test"].reshape(data["X_test"].shape[0], -1)
    logger.info("窗口展平：%s → %s（96×12=1152 维）", data["X_train"].shape, X_train.shape)
    return (
        X_train.astype(np.float32),
        data["y_train"].astype(np.float32),
        X_val.astype(np.float32),
        data["y_val"].astype(np.float32),
        X_test.astype(np.float32),
        data["y_test"].astype(np.float32),
    )


def train_lightgbm(X_train, y_train, X_val, y_val) -> MultiOutputRegressor:
    """训练 24 个单输出 LightGBM 模型（每个预测步一个）。"""
    b = cfg["baselines"]["lightgbm"]
    base = LGBMRegressor(
        n_estimators=int(b["n_estimators"]),
        learning_rate=float(b["learning_rate"]),
        num_leaves=int(b["num_leaves"]),
        n_jobs=-1,
        verbose=-1,  # 压掉 LightGBM 每棵树一轮的进度输出
        random_state=int(cfg["experiment"]["seed"]),
    )
    model = MultiOutputRegressor(base, n_jobs=1)  # 内层 n_jobs=-1 已并行

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    # 验证集指标用于和 Transformer 的早停口径对照（LightGBM 不早停）
    val_pred = model.predict(X_val)
    val_metrics = compute_metrics(y_val, val_pred)
    logger.info(
        "LightGBM 训练完成（%.1fs）。验证集 MAE: %.4f（归一化尺度）",
        elapsed,
        val_metrics["MAE"],
    )

    model_path = Path(cfg["output"]["model_dir"]) / "lightgbm_baseline.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("模型已保存至 %s", model_path)
    return model


def evaluate(model: MultiOutputRegressor, X_test, y_test) -> None:
    """测试集评估：逆变换后输出真实指标并保存逐点预测。"""
    t0 = time.perf_counter()
    preds_scaled = model.predict(X_test)

    with open(cfg["data"]["scaler_y"], "rb") as fh:
        scaler_y = pickle.load(fh)
    preds_real = scaler_y.inverse_transform(preds_scaled)
    trues_real = scaler_y.inverse_transform(y_test)

    metrics = compute_metrics(trues_real, preds_real)
    logger.info(
        "测试集 MAE: %.2f kW, RMSE: %.2f kW, MAPE: %.2f%%, R2: %.3f（推理 %.1fs）",
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MAPE"],
        metrics["R2"],
        time.perf_counter() - t0,
    )

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lightgbm_test_predictions.npy", preds_real)
    logger.info("逐点预测已保存至 results/lightgbm_test_predictions.npy")


def main() -> int:
    """主流程：加载 → 训练 → 评估。"""
    X_train, y_train, X_val, y_val, X_test, y_test = load_windowed_data()
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    evaluate(model, X_test, y_test)
    logger.info("LightGBM 基线完成。下一步运行 05_evaluate_compare.py 汇总对比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
