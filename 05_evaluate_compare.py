"""多模型对比与可视化：持续预测 / LightGBM / LSTM / Transformer（消融） / Informer。

输出：
    results/metrics.csv                    各模型指标表（"简历数字"的依据，入库）
    results/model_comparison.png           单窗口预测对比 + 误差柱状图

运行：
    python 05_evaluate_compare.py
"""

from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# 允许以 python 05_evaluate_compare.py 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config  # noqa: E402
from src.data import compute_metrics, naive_persistence_predict  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.models import InformerModel, LSTMModel, TransformerModel  # noqa: E402

# 图表中文渲染：Windows 用 SimHei，其余平台用 Noto/文泉驿兜底。
# 不设置时 matplotlib 默认字体缺 CJK 字形，中文标题会变成方块并刷警告。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

logger = get_logger(__name__)
cfg = load_config("configs/base.yaml")

RESULTS_DIR = Path(cfg["output"]["results_dir"])


def load_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载测试集与逆变换后的真实目标。

    Returns:
        ``(X_test, y_test_scaled, y_test_real, scaler_y)``。
    """
    npz_path = Path(cfg["data"]["npz"])
    if not npz_path.exists():
        raise FileNotFoundError(
            f"数据集不存在：{npz_path}\n"
            "  修复：先运行 01_explore_quality.py 与 02_preprocess_windows.py。"
        )
    data = np.load(npz_path, allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    with open(cfg["data"]["scaler_y"], "rb") as fh:
        scaler_y = pickle.load(fh)

    y_real = scaler_y.inverse_transform(y_test)
    return X_test, y_test, y_real, scaler_y


def predict_lstm(X_test: np.ndarray) -> np.ndarray:
    """加载 03 脚本训练的 LSTM 权重并做测试集推理（返回真实值）。"""
    weights = Path(cfg["baselines"]["lstm"]["weights"])
    if not weights.exists():
        logger.warning("LSTM 权重不存在：%s，跳过该模型", weights)
        raise FileNotFoundError(weights)

    b = cfg["baselines"]["lstm"]
    model = LSTMModel(
        input_size=X_test.shape[2],
        hidden_size=int(b["hidden_size"]),
        num_layers=int(b["num_layers"]),
        output_len=int(cfg["data"]["pred_len"]),
        dropout=float(b["dropout"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    dataset = torch.FloatTensor(X_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for x_batch in loader:
            preds.append(model(x_batch.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_transformer(X_test: np.ndarray) -> np.ndarray:
    """加载 04a 脚本训练的普通 Transformer 权重并推理（返回真实值）。"""
    weights = RESULTS_DIR.parent / "models" / "transformer.pth"
    if not weights.exists():
        raise FileNotFoundError(
            f"Transformer 权重不存在：{weights}\n" "  修复：先运行 04a_train_transformer.py。"
        )

    m = cfg["transformer_ablation"]
    model = TransformerModel(
        input_size=X_test.shape[2],
        d_model=int(m["d_model"]),
        nhead=int(m["nhead"]),
        num_layers=int(m["num_encoder_layers"]),
        dim_feedforward=int(m["dim_feedforward"]),
        dropout=float(m["dropout"]),
        output_len=int(cfg["data"]["pred_len"]),
        pool=str(m["pool"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    loader = torch.utils.data.DataLoader(torch.FloatTensor(X_test), batch_size=256, shuffle=False)
    use_residual = bool(cfg["training"].get("residual", False))
    with torch.no_grad():
        for x_batch in loader:
            out = model(x_batch.to(device)).cpu().numpy()
            if use_residual:
                # 模型输出残差增量（训练协议），加回基线才是绝对功率
                out = out + x_batch[:, -1, 0:1].numpy()
            preds.append(out)
    return np.concatenate(preds, axis=0)


def predict_informer(X_test: np.ndarray) -> np.ndarray:
    """加载 04 脚本训练的 Informer 权重并推理（返回真实值）。"""
    weights = RESULTS_DIR.parent / "models" / "informer.pth"
    if not weights.exists():
        raise FileNotFoundError(
            f"Informer 权重不存在：{weights}\n" "  修复：先运行 04_train_informer.py。"
        )

    m = cfg["model"]
    model = InformerModel(
        input_size=X_test.shape[2],
        d_model=int(m["d_model"]),
        nhead=int(m["nhead"]),
        e_layers=int(m["e_layers"]),
        d_layers=int(m["d_layers"]),
        d_ff=int(m["d_ff"]),
        dropout=float(m["dropout"]),
        factor=int(m["factor"]),
        label_len=int(m["label_len"]),
        pred_len=int(cfg["data"]["pred_len"]),
        distil=bool(m["distil"]),
        activation=str(m["activation"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    loader = torch.utils.data.DataLoader(torch.FloatTensor(X_test), batch_size=256, shuffle=False)
    use_residual = bool(cfg["training"].get("residual", False))
    with torch.no_grad():
        for x_batch in loader:
            out = model(x_batch.to(device)).cpu().numpy()
            # Informer 输出完整解码序列（含开始标记重构段），取最后 pred_len 步
            out = out[:, -int(cfg["data"]["pred_len"]) :]
            if use_residual:
                # 模型输出残差增量（训练协议），加回基线才是绝对功率
                out = out + x_batch[:, -1, 0:1].numpy()
            preds.append(out)
    return np.concatenate(preds, axis=0)


def predict_lightgbm() -> np.ndarray:
    """加载 03b 脚本保存的逐点预测（返回真实值）。"""
    preds_file = RESULTS_DIR / "lightgbm_test_predictions.npy"
    if not preds_file.exists():
        raise FileNotFoundError(
            f"LightGBM 预测不存在：{preds_file}\n" "  修复：先运行 03b_baseline_lightgbm.py。"
        )
    return np.load(preds_file)


def evaluate_all(
    X_test: np.ndarray, y_real: np.ndarray, scaler_y
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """对每个可用模型计算指标。

    Returns:
        ``(指标字典, 逐点预测字典)``。指标键为模型中文名；
        预测值均为逆变换后的真实物理值（kW）。
    """
    preds_real: dict[str, np.ndarray] = {
        "持续预测（naive）": scaler_y.inverse_transform(
            naive_persistence_predict(X_test, int(cfg["data"]["pred_len"]))
        ),
    }

    for name, loader in [
        ("LightGBM", predict_lightgbm),
        ("LSTM", lambda: scaler_y.inverse_transform(predict_lstm(X_test))),
        (
            "Transformer（普通注意力，消融）",
            lambda: scaler_y.inverse_transform(predict_transformer(X_test)),
        ),
        ("Informer（本文）", lambda: scaler_y.inverse_transform(predict_informer(X_test))),
    ]:
        try:
            preds_real[name] = loader()
        except FileNotFoundError as exc:
            logger.warning("跳过 %s：%s", name, exc)

    metrics = {}
    for name, preds in preds_real.items():
        metrics[name] = compute_metrics(y_real, preds)
        logger.info(
            "%-16s MAE %7.2f kW  RMSE %7.2f kW  MAPE %5.2f%%  R2 %.3f",
            name,
            metrics[name]["MAE"],
            metrics[name]["RMSE"],
            metrics[name]["MAPE"],
            metrics[name]["R2"],
        )
    return metrics, preds_real


def save_metrics_csv(metrics: dict[str, dict[str, float]]) -> None:
    """把指标写入 results/metrics.csv（该文件入库，是简历数字的依据）。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "metrics.csv"
    rows = sorted(metrics.items(), key=lambda kv: kv[1]["MAE"])

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["模型", "MAE(kW)", "RMSE(kW)", "MAPE(%)", "R2"])
        for name, m in rows:
            writer.writerow(
                [
                    name,
                    f"{m['MAE']:.2f}",
                    f"{m['RMSE']:.2f}",
                    f"{m['MAPE']:.2f}",
                    f"{m['R2']:.3f}",
                ]
            )
    logger.info("指标表已写入 %s", path)


def plot_comparison(y_real: np.ndarray, preds_real: dict[str, np.ndarray]) -> None:
    """绘制单窗口对比图与指标柱状图。"""
    sample_idx = 0
    future_hours = np.arange(0, y_real.shape[1] * 15, 15) / 60.0

    # 图 1：单窗口预测对比
    plt.figure(figsize=(8, 3.8))
    plt.plot(future_hours, y_real[sample_idx], "o-", label="真实值", linewidth=2)
    styles = ["s--", "^--", "d--"]
    for (name, preds), style in zip(preds_real.items(), styles, strict=False):
        plt.plot(future_hours, preds[sample_idx], style, label=name, alpha=0.85)
    plt.xlabel("未来时间 (小时)")
    plt.ylabel("交流功率 (kW)")
    plt.title("各模型预测对比（测试集第 1 个窗口）")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # 图 2：MAE/RMSE 柱状图
    names = list(preds_real.keys())
    maes = [float(np.mean(np.abs(p - y_real))) for p in preds_real.values()]
    rmses = [float(np.sqrt(np.mean((p - y_real) ** 2))) for p in preds_real.values()]

    x_pos = np.arange(len(names))
    width = 0.38
    plt.figure(figsize=(max(6, len(names) * 1.6), 4))
    plt.bar(x_pos - width / 2, maes, width, label="MAE (kW)")
    plt.bar(x_pos + width / 2, rmses, width, label="RMSE (kW)")
    plt.xticks(x_pos, names, rotation=12)
    plt.ylabel("误差 (kW)")
    plt.title("各模型测试集误差对比")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_error_bar.png", dpi=150)
    plt.close()
    logger.info("对比图已保存至 %s", RESULTS_DIR)


def main() -> int:
    """主流程：加载测试集 → 各模型推理 → 指标与图表。"""
    t0 = time.perf_counter()
    X_test, _, y_real, scaler_y = load_test_data()
    metrics, preds_real = evaluate_all(X_test, y_real, scaler_y)

    if not metrics:
        logger.error("没有任何模型可用。请先运行 03b 与 04 脚本。")
        return 1

    save_metrics_csv(metrics)
    plot_comparison(y_real, preds_real)

    # 打印可直接粘贴进 README 的表格
    print()
    print("| 模型 | MAE (kW) | RMSE (kW) | MAPE (%) | R² |")
    print("|---|---|---|---|---|")
    for name, m in sorted(metrics.items(), key=lambda kv: kv[1]["MAE"]):
        print(
            f"| {name} | {m['MAE']:.2f} | {m['RMSE']:.2f} " f"| {m['MAPE']:.2f} | {m['R2']:.3f} |"
        )

    logger.info("对比完成，总耗时 %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
