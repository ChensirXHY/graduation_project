import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings

warnings.filterwarnings("ignore")

# ==================== 0. 数据加载 ====================
df = pd.read_csv("archive/plant_1_cleaned.csv")
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
df.sort_values("DATE_TIME", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"加载数据：{df.shape[0]} 行, 时间范围：{df['DATE_TIME'].min()} 至 {df['DATE_TIME'].max()}")

# ==================== 1. 时间特征构建 ====================
# 提取时间分量
df["hour"] = df["DATE_TIME"].dt.hour
df["minute"] = df["DATE_TIME"].dt.minute
# df['dayofyear'] = df['DATE_TIME'].dt.dayofyear  # 如果数据跨年，可加

# 循环时间编码（避免23点与0点断层）
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

# ==================== 2. 衍生气象/功率特征 ====================
# 昼夜标记（辐照度 > 0.005 kW/m² 视为白天，避免传感器微小噪声）
df["is_daytime"] = (df["IRRADIATION"] > 0.005).astype(int)

# 温度差（组件温度 - 环境温度，反映面板发热程度）
df["temp_diff"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]

# 滞后特征（滚动统计，注意不能跨训练/验证/测试边界，这里先对所有数据做，后面切分后再丢弃边界样本）
# 这里用 shift 产生过去 1 小时的均值，窗口大小为 4（15min * 4 = 1h）
df["power_roll_mean_1h"] = df["TOTAL_AC_POWER"].rolling(window=4, min_periods=1).mean()
df["power_roll_std_1h"] = df["TOTAL_AC_POWER"].rolling(window=4, min_periods=1).std().fillna(0)

# ==================== 3. 选取最终特征列 ====================
feature_cols = [
    "TOTAL_AC_POWER",  # 目标变量（历史值作为特征，未来值作为目标）
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "is_daytime",
    "temp_diff",
    "power_roll_mean_1h",
    "power_roll_std_1h",
]

target_col = "TOTAL_AC_POWER"  # 预测目标

# 查看当前数据是否有 NaN（滚动均值开头会有 NaN，但已前向填充）
print("\n特征缺失值检查：")
print(df[feature_cols].isnull().sum())

# ==================== 4. 时间顺序切分数据集（防止泄露） ====================
# 按日期分割，使用第一天和最后一天边界
train_end_date = pd.Timestamp("2020-06-07 00:00:00")
val_end_date = pd.Timestamp("2020-06-12 00:00:00")

train_mask = df["DATE_TIME"] < train_end_date
val_mask = (df["DATE_TIME"] >= train_end_date) & (df["DATE_TIME"] < val_end_date)
test_mask = df["DATE_TIME"] >= val_end_date

df_train = df[train_mask].copy()
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()

print(
    f"\n训练集：{len(df_train)} 条  ({df_train['DATE_TIME'].min()} 至 {df_train['DATE_TIME'].max()})"
)
print(f"验证集：{len(df_val)} 条  ({df_val['DATE_TIME'].min()} 至 {df_val['DATE_TIME'].max()})")
print(f"测试集：{len(df_test)} 条  ({df_test['DATE_TIME'].min()} 至 {df_test['DATE_TIME'].max()})")

# ==================== 5. 保存中间结果（供下一阶段使用） ====================
# 保存切分后的原始特征数据（未归一化），以便归一化时只用训练集统计量
df_train.to_csv("archive/train_features.csv", index=False)
df_val.to_csv("archive/val_features.csv", index=False)
df_test.to_csv("archive/test_features.csv", index=False)

# 保存特征列名称（后续脚本参考）
with open("archive/feature_columns.txt", "w") as f:
    f.write("\n".join(feature_cols) + "\n")

print("\n已生成 train_features.csv, val_features.csv, test_features.csv")

# ==================== 二   归一化  ====================
#       1.加载上一轮保存的数据
df_train = pd.read_csv("archive/train_features.csv")
df_val = pd.read_csv("archive/val_features.csv")
df_test = pd.read_csv("archive/test_features.csv")
# 读取特征列清单
with open("archive/feature_columns.txt", "r") as f:
    feature_cols = f.read().strip().split("\n")
target_col = "TOTAL_AC_POWER"
print("特征列：", feature_cols)
#         2. 提取特征矩阵
X_train_raw = df_train[feature_cols].values  # shape: (样本数, 特征维)
X_val_raw = df_val[feature_cols].values
X_test_raw = df_test[feature_cols].values
# 目标值存为单独数组（用于归一化和窗口构建）
y_train_raw = df_train[target_col].values
y_val_raw = df_val[target_col].values
y_test_raw = df_test[target_col].values
#         3. 归一化（仅用训练集拟合）
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
#  在训练集上拟合
scaler_X.fit(X_train_raw)
scaler_y.fit(y_train_raw.reshape(-1, 1))
#  变换所有数据集
X_train_scaled = scaler_X.transform(X_train_raw)
X_val_scaled = scaler_X.transform(X_val_raw)
X_test_scaled = scaler_X.transform(X_test_raw)
y_train_scaled = scaler_y.transform(y_train_raw.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
# 保存归一化器（预测后要逆变换回原始功率）
with open("archive/scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
with open("archive/scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)
print("归一化完成，scaler 已保存。")


#              4. 滑动窗口构建函数
def create_sliding_windows(X, y, seq_len=96, pred_len=24):
    """
    将时间序列数据转换为监督学习样本。

    参数:
        X: 特征矩阵 (时间步数, 特征维)
        y: 目标序列 (时间步数,)
        seq_len: 用作输入的历史窗口长度
        pred_len: 需要预测的未来步数

    返回:
        X_windows: (样本数, seq_len, 特征维)
        y_windows: (样本数, pred_len)
    """
    X_windows, y_windows = [], []
    total_len = len(X)

    for i in range(total_len - seq_len - pred_len + 1):
        X_windows.append(X[i : i + seq_len])
        y_windows.append(y[i + seq_len : i + seq_len + pred_len])

    return np.array(X_windows), np.array(y_windows)


#              5. 生成窗口样本
SEQ_LEN = 96  # 过去24小时（15分钟 * 96 = 24h）
PRED_LEN = 24  # 未来6小时
print(f"\n历史窗口：{SEQ_LEN} 步  |  预测窗口：{PRED_LEN} 步")
X_train, y_train = create_sliding_windows(X_train_scaled, y_train_scaled, SEQ_LEN, PRED_LEN)
X_val, y_val = create_sliding_windows(X_val_scaled, y_val_scaled, SEQ_LEN, PRED_LEN)
X_test, y_test = create_sliding_windows(X_test_scaled, y_test_scaled, SEQ_LEN, PRED_LEN)
print(f"训练样本：{X_train.shape}  -> 输入: {X_train.shape} 目标: {y_train.shape}")
print(f"验证样本：{X_val.shape}")
print(f"测试样本：{X_test.shape}")
#                6. 保存最终数据集
np.savez(
    "archive/scaled_data.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    feature_names=feature_cols,
)
print("\n数据集已保存至 scaled_data.npz")
