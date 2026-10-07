"""
LSTM 基线模型训练与评估（多步预测）
输入：archive/scaled_data.npz
输出：models/lstm_baseline.pth, results/lstm_loss_curve.png, results/lstm_prediction_sample.png
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 加载标准化数据
data = np.load("archive/scaled_data.npz", allow_pickle=True)
X_train = data["X_train"]  # (N, 96, 12)
y_train = data["y_train"]  # (N, 24)
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]
feature_names = data["feature_names"]

print(f"训练样本：{X_train.shape[0]}  输入形状：{X_train.shape[1:]}  输出长度：{y_train.shape[1]}")
print(f"验证样本：{X_val.shape[0]}")
print(f"测试样本：{X_test.shape[0]}")

# 构建 DataLoader（时序数据不 shuffle）
BATCH_SIZE = 64
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# LSTM 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_len=24, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_len)  # 最后时刻隐状态映射到未来多步

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        out = self.fc(last_hidden)  # (batch, pred_len)
        return out


INPUT_SIZE = X_train.shape[2]
PRED_LEN = y_train.shape[1]
model = LSTMModel(input_size=INPUT_SIZE, output_len=PRED_LEN).to(device)
print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

# 训练设置
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 100
PATIENCE = 10
best_val_loss = float("inf")
patience_counter = 0
train_losses = []
val_losses = []

# 创建输出目录
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 训练循环（含早停）
for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * X_batch.size(0)
    epoch_train_loss = train_loss_sum / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss_sum += loss.item() * X_batch.size(0)
    epoch_val_loss = val_loss_sum / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(
        f"Epoch {epoch+1:3d}/{EPOCHS}  Train Loss: {epoch_train_loss:.6f}  Val Loss: {epoch_val_loss:.6f}"
    )

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "models/lstm_baseline.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"早停于第 {epoch+1} 轮")
            break

# 测试评估
model.load_state_dict(torch.load("models/lstm_baseline.pth", map_location=device))
model.eval()
with open("archive/scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

preds_all, trues_all = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        trues = y_batch.cpu().numpy()
        preds_all.append(preds)
        trues_all.append(trues)

preds_all = np.concatenate(preds_all, axis=0)
trues_all = np.concatenate(trues_all, axis=0)
preds_real = scaler_y.inverse_transform(preds_all)
trues_real = scaler_y.inverse_transform(trues_all)

mae = np.mean(np.abs(preds_real - trues_real))
rmse = np.sqrt(np.mean((preds_real - trues_real) ** 2))
print(f"测试集 MAE: {mae:.2f} kW, RMSE: {rmse:.2f} kW")

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
if len(val_losses) > PATIENCE:
    early_stop_epoch = len(val_losses) - PATIENCE
    plt.axvline(x=early_stop_epoch, linestyle="--", color="red", alpha=0.5, label="早停点")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM 训练损失")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lstm_loss_curve.png", dpi=150)
plt.close()

# 绘制单个样本预测对比
sample_idx = 0
future_hours = np.arange(0, PRED_LEN * 15, 15) / 60.0  # 转为小时
plt.figure(figsize=(8, 3.5))
plt.plot(future_hours, trues_real[sample_idx], "o-", label="真实值")
plt.plot(future_hours, preds_real[sample_idx], "s--", label="预测值")
plt.xlabel("未来时间 (小时)")
plt.ylabel("交流功率 (kW)")
plt.title(f"LSTM 预测样本 (测试集第 {sample_idx+1} 个窗口)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lstm_prediction_sample.png", dpi=150)
plt.close()

print("LSTM基线训练完成，模型与图表已保存。")
