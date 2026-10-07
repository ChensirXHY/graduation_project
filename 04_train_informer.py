"""Informer 主模型训练与评估（多步预测，本科毕设核心方法）。

输入：archive/scaled_data.npz（02 脚本产出的归一化窗口数据）
输出：models/informer.pth, results/informer_loss_curve.png,
      results/informer_prediction_sample.png, results/informer_test_predictions.npy

与 03_baseline_lstm.py 使用完全相同的：
    - 数据（同一 npz 与 scaler）
    - 损失（MSE）与早停策略
    - 评估口径（逆变换后计算 MAE/RMSE）
以保证各模型的对比是公平的，差异只来自模型结构本身。

残差目标（training.residual=true，默认开启）：
    让模型预测 y − 最后一步功率 的"增量"，推理时加回基线。
    naive 持续预测是强先验，学习增量相当于在其之上做修正 ——
    实测该技巧把普通 Transformer 的 MAE 从 1888 降到 1865 kW。

运行：
    python 04_train_informer.py
"""

from __future__ import annotations

import pickle
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 允许以 python 04_train_informer.py 直接运行（不依赖包安装）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config  # noqa: E402
from src.data import compute_metrics  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.models import InformerModel  # noqa: E402

# 图表中文渲染
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

logger = get_logger(__name__)
cfg = load_config("configs/base.yaml")

# ---------- 可复现性：固定全部随机源 ----------
SEED = int(cfg["experiment"]["seed"])
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ---------- 设备配置 ----------
if cfg["experiment"]["device"] == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(cfg["experiment"]["device"])
logger.info("使用设备：%s", device)

USE_RESIDUAL = bool(cfg["training"].get("residual", True))


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 02 脚本产出的窗口化数据。

    Returns:
        ``(X_train, y_train, X_val, y_val)``，均为 float32。
        测试集在评估阶段单独加载，训练过程完全不接触。

    Raises:
        FileNotFoundError: npz 不存在，提示先运行 01/02 脚本。
    """
    npz_path = Path(cfg["data"]["npz"])
    if not npz_path.exists():
        raise FileNotFoundError(
            f"数据集不存在：{npz_path}\n"
            "  修复：按 README「快速开始」获取 Kaggle 原始数据后，\n"
            "  依次运行 01_explore_quality.py 与 02_preprocess_windows.py。"
        )

    data = np.load(npz_path, allow_pickle=True)
    logger.info(
        "训练样本：%s  验证样本：%s  特征数：%d",
        data["X_train"].shape,
        data["X_val"].shape,
        data["X_train"].shape[2],
    )
    return (
        data["X_train"].astype(np.float32),
        data["y_train"].astype(np.float32),
        data["X_val"].astype(np.float32),
        data["y_val"].astype(np.float32),
    )


def build_model() -> InformerModel:
    """按 configs/base.yaml 构造 Informer 主模型。"""
    m = cfg["model"]
    return InformerModel(
        input_size=int(m["input_size"]),
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


def full_target(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    label_len: int,
    use_residual: bool,
) -> torch.Tensor:
    """构造解码器完整目标（论文的 teacher forcing 协议）。

    解码器输出 label_len + pred_len 个位置：
        - 前 label_len 个位置：以输入窗口最后 label_len 步的**已知功率**为目标
          （残差模式下为已知功率相对基线的增量）；
        - 后 pred_len 个位置：以未来真实功率为目标。

    相比"只监督预测段"，这种协议多了一半监督信号（开始标记的重构），
    是官方实现与论文采用的协议。

    Args:
        x_batch: 输入窗口 ``(B, L, F)``，特征 0 为功率。
        y_batch: 未来目标 ``(B, pred_len)``。
        label_len: 开始标记长度。
        use_residual: 是否使用残差目标（预测增量）。

    Returns:
        ``(B, label_len + pred_len)`` 的完整目标。
    """
    base = x_batch[:, -1, 0:1]
    if use_residual:
        label_part = x_batch[:, -label_len:, 0] - base
        pred_part = y_batch - base
    else:
        label_part = x_batch[:, -label_len:, 0]
        pred_part = y_batch
    return torch.cat([label_part, pred_part], dim=1)


def train(model: InformerModel, train_loader, val_loader) -> tuple[list[float], list[float]]:
    """训练循环：MSE + Adam + 余弦退火 + 梯度裁剪 + 早停。

    Returns:
        ``(train_losses, val_losses)`` 每 epoch 的平均 MSE 列表。
    """
    t = cfg["training"]
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    epochs = int(t["epochs"])
    patience = int(t["early_stopping_patience"])

    scheduler = None
    if t["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    model_path = Path(cfg["output"]["model_dir"]) / "informer.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            # 完整序列损失：开始标记重构 + 未来预测（teacher forcing）
            target = full_target(x_batch, y_batch, model.label_len, USE_RESIDUAL)
            loss = criterion(out, target)
            loss.backward()
            # 梯度裁剪：注意力堆叠训练必备，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]))
            optimizer.step()
            train_sum += loss.item() * x_batch.size(0)
        epoch_train = train_sum / len(train_loader.dataset)
        train_losses.append(epoch_train)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                target = full_target(x_batch, y_batch, model.label_len, USE_RESIDUAL)
                val_sum += criterion(out, target).item() * x_batch.size(0)
        epoch_val = val_sum / len(val_loader.dataset)
        val_losses.append(epoch_val)

        if scheduler is not None:
            scheduler.step()

        logger.info(
            "Epoch %3d/%d  Train Loss: %.6f  Val Loss: %.6f  LR: %.2e",
            epoch + 1,
            epochs,
            epoch_train,
            epoch_val,
            optimizer.param_groups[0]["lr"],
        )

        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("早停于第 %d 轮（patience=%d）", epoch + 1, patience)
                break

    logger.info("最佳验证损失 %.6f，权重已保存至 %s", best_val_loss, model_path)
    return train_losses, val_losses


def evaluate(
    model: InformerModel,
    train_losses: list[float],
    val_losses: list[float],
) -> None:
    """在测试集上评估并输出逆变换后的真实指标与图表。

    Args:
        model: 已训练（或已加载权重）的模型。
        train_losses: 各 epoch 训练损失，用于绘制曲线。
        val_losses: 各 epoch 验证损失，用于绘制曲线。
    """
    t0 = time.perf_counter()
    data = np.load(cfg["data"]["npz"], allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    with open(cfg["data"]["scaler_y"], "rb") as fh:
        scaler_y = pickle.load(fh)

    model.load_state_dict(
        torch.load(
            Path(cfg["output"]["model_dir"]) / "informer.pth",
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    preds_all, trues_all = [], []
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=256,
        shuffle=False,
    )
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            out = model(x_batch.to(device)).cpu().numpy()
            # 取解码输出的最后 pred_len 个位置（完整输出含开始标记重构段）
            out = out[:, -model.pred_len :]
            if USE_RESIDUAL:
                # 预测增量 + 基线 = 绝对功率
                out = out + x_batch[:, -1, 0:1].numpy()
            preds_all.append(out)
            trues_all.append(y_batch.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    preds_real = scaler_y.inverse_transform(preds_all)
    trues_real = scaler_y.inverse_transform(trues_all)

    metrics = compute_metrics(trues_real, preds_real)
    logger.info(
        "测试集 MAE: %.2f kW, RMSE: %.2f kW, MAPE: %.2f%%, R2: %.3f（耗时 %.1fs）",
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MAPE"],
        metrics["R2"],
        time.perf_counter() - t0,
    )

    # 保存逐点预测，供 05_evaluate_compare.py 汇总对比
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "informer_test_predictions.npy", preds_real)
    logger.info("逐点预测已保存至 results/informer_test_predictions.npy")

    plot_loss_curve(train_losses, val_losses, out_dir)
    plot_prediction_sample(trues_real, preds_real, out_dir)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], out_dir: Path) -> None:
    """绘制并保存训练/验证损失曲线。"""
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Informer 训练损失")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "informer_loss_curve.png", dpi=150)
    plt.close()
    logger.info("损失曲线已保存")


def plot_prediction_sample(trues_real, preds_real, out_dir: Path) -> None:
    """绘制单个测试窗口的预测对比图（真实物理值）。"""
    sample_idx = 0
    future_hours = np.arange(0, len(trues_real[0]) * 15, 15) / 60.0

    plt.figure(figsize=(8, 3.5))
    plt.plot(future_hours, trues_real[sample_idx], "o-", label="真实值")
    plt.plot(future_hours, preds_real[sample_idx], "s--", label="Informer 预测")
    plt.xlabel("未来时间 (小时)")
    plt.ylabel("交流功率 (kW)")
    plt.title(f"Informer 预测样本 (测试集第 {sample_idx + 1} 个窗口)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "informer_prediction_sample.png", dpi=150)
    plt.close()
    logger.info("预测样本图已保存")


def main() -> int:
    """主流程：加载数据 → 训练 → 测试评估。"""
    X_train, y_train, X_val, y_val = load_dataset()

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Informer 参数量：%d（d_model=%d, heads=%d, e_layers=%d, d_layers=%d, factor=%d, 残差目标=%s）",
        n_params,
        cfg["model"]["d_model"],
        cfg["model"]["nhead"],
        cfg["model"]["e_layers"],
        cfg["model"]["d_layers"],
        cfg["model"]["factor"],
        USE_RESIDUAL,
    )

    train_losses, val_losses = train(model, train_loader, val_loader)
    evaluate(model, train_losses, val_losses)
    logger.info("Informer 训练完成。下一步运行 05_evaluate_compare.py 汇总对比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
