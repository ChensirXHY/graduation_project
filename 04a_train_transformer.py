"""普通注意力 Transformer 训练与评估（消融对照脚本）。

用途：作为 Informer 的消融基线 —— 与 Informer 共享同一数据与训练协议，
唯一差异是注意力机制（全量注意力 vs ProbSparse 稀疏注意力），
用于证明稀疏机制在效果不降的前提下带来的效率优势。

输入：archive/scaled_data.npz（02 脚本产出的归一化窗口数据）
输出：models/transformer.pth, results/transformer_loss_curve.png,
      results/transformer_prediction_sample.png,
      results/transformer_test_predictions.npy

残差目标（training.residual=true，默认开启）：
    预测 y − 最后一步功率 的增量，推理时加回基线。
    实测该技巧把 MAE 从 1888 降到 1865 kW。

运行：
    python 04a_train_transformer.py
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

# 允许以 python 04_train_transformer.py 直接运行（不依赖包安装）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config  # noqa: E402
from src.data import compute_metrics  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.models import TransformerModel  # noqa: E402

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


def set_seed_worker(worker_id: int) -> None:
    """给 DataLoader 的每个 worker 也设置种子，保证 batch 顺序可复现。"""
    np.random.seed(SEED + worker_id)


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


def build_model(input_size: int, output_len: int) -> TransformerModel:
    """按 configs/base.yaml 的 transformer_ablation 节构造消融对照模型。"""
    m = cfg["transformer_ablation"]
    return TransformerModel(
        input_size=input_size,
        d_model=int(m["d_model"]),
        nhead=int(m["nhead"]),
        num_layers=int(m["num_encoder_layers"]),
        dim_feedforward=int(m["dim_feedforward"]),
        dropout=float(m["dropout"]),
        output_len=output_len,
        pool=str(m["pool"]),
    )


def train(model: TransformerModel, train_loader, val_loader) -> tuple[list[float], list[float]]:
    """训练循环：MSE + Adam + 余弦退火 + 梯度裁剪 + 早停。

    Returns:
        ``(train_losses, val_losses)`` 每 epoch 的平均 MSE 列表。
    """
    t = cfg["training"]
    use_residual = bool(t.get("residual", False))
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
    model_path = Path(cfg["output"]["model_dir"]) / "transformer.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            # 残差目标：预测增量，损失在增量空间计算
            target = y_batch - x_batch[:, -1, 0:1] if use_residual else y_batch
            loss = criterion(out, target)
            loss.backward()
            # 梯度裁剪：Transformer 训练必备，防止注意力层梯度爆炸
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
                target = y_batch - x_batch[:, -1, 0:1] if use_residual else y_batch
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
    model: TransformerModel,
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

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model.load_state_dict(
        torch.load(
            Path(cfg["output"]["model_dir"]) / "transformer.pth",
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    preds_all, trues_all = [], []
    use_residual = bool(cfg["training"].get("residual", False))
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch.to(device)).cpu().numpy()
            if use_residual:
                # 预测增量 + 基线 = 绝对功率
                preds = preds + x_batch[:, -1, 0:1].numpy()
            preds_all.append(preds)
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
    np.save(out_dir / "transformer_test_predictions.npy", preds_real)
    logger.info("逐点预测已保存至 results/transformer_test_predictions.npy")

    plot_loss_curve(train_losses, val_losses, out_dir)
    plot_prediction_sample(trues_real, preds_real, out_dir)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], out_dir: Path) -> None:
    """绘制并保存训练/验证损失曲线。"""
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Transformer 训练损失")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "transformer_loss_curve.png", dpi=150)
    plt.close()
    logger.info("损失曲线已保存")


def plot_prediction_sample(trues_real, preds_real, out_dir: Path) -> None:
    """绘制单个测试窗口的预测对比图（真实物理值）。"""
    sample_idx = 0
    future_hours = np.arange(0, len(trues_real[0]) * 15, 15) / 60.0

    plt.figure(figsize=(8, 3.5))
    plt.plot(future_hours, trues_real[sample_idx], "o-", label="真实值")
    plt.plot(future_hours, preds_real[sample_idx], "s--", label="Transformer 预测")
    plt.xlabel("未来时间 (小时)")
    plt.ylabel("交流功率 (kW)")
    plt.title(f"Transformer 预测样本 (测试集第 {sample_idx + 1} 个窗口)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "transformer_prediction_sample.png", dpi=150)
    plt.close()
    logger.info("预测样本图已保存")


def main() -> int:
    """主流程：加载数据 → 训练 → 测试评估。"""
    X_train, y_train, X_val, y_val = load_dataset()

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    # 时序数据不 shuffle：保持时间顺序（其实窗口样本独立，但保持与 03 一致）
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        worker_init_fn=set_seed_worker,
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = build_model(X_train.shape[2], y_train.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    m = cfg["transformer_ablation"]
    logger.info(
        "Transformer（消融）参数量：%d（d_model=%d, layers=%d, heads=%d, pool=%s）",
        n_params,
        m["d_model"],
        m["num_encoder_layers"],
        m["nhead"],
        m["pool"],
    )

    train_losses, val_losses = train(model, train_loader, val_loader)
    evaluate(model, train_losses, val_losses)
    logger.info("Transformer 训练完成。下一步运行 05_evaluate_compare.py 汇总对比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
