#数据探索和生成质量报告
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
# 定义图片文件夹名称，和py脚本同目录
img_dir = "archive"
# ---------- 文件路径 ----------
GEN_PATH = 'archive\Plant_1_Generation_Data.csv'
WEATHER_PATH = 'archive\Plant_1_Weather_Sensor_Data.csv'

# ---------- 读取发电数据 ----------
print("正在读取发电数据...")
df_gen = pd.read_csv(GEN_PATH)
print(f"发电数据形状: {df_gen.shape}")

# ---------- 读取气象数据 ----------
print("正在读取气象数据...")
df_weather = pd.read_csv(WEATHER_PATH)
print(f"气象数据形状: {df_weather.shape}")


# ---------- 发电数据时间处理 ----------
# 格式: DD-MM-YYYY HH:MM
df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], format='%d-%m-%Y %H:%M')
print(f"发电数据时间范围: {df_gen['DATE_TIME'].min()} 至 {df_gen['DATE_TIME'].max()}")

# ---------- 气象数据时间处理 ----------
# 格式: YYYY/MM/DD HH:MM (注意是斜杠)
df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'], format='%Y/%m/%d %H:%M')
print(f"气象数据时间范围: {df_weather['DATE_TIME'].min()} 至 {df_weather['DATE_TIME'].max()}")

# ---------- 按时间聚合总功率 ----------
# 将所有逆变器的 AC_POWER 按时间求和
df_total = df_gen.groupby('DATE_TIME')['AC_POWER'].sum().reset_index()
df_total.rename(columns={'AC_POWER': 'TOTAL_AC_POWER'}, inplace=True)

# ---------- 左连接气象数据 ----------
df_merged = pd.merge(df_total, df_weather, on='DATE_TIME', how='left')
print(f"合并后数据形状: {df_merged.shape}")

# ---------- 检查缺失值 ----------
print("\n===== 缺失值统计 =====")
print(df_merged.isnull().sum())

# ---------- 简单前向填充（如果气象数据有少量缺失） ----------
df_merged['IRRADIATION'] = df_merged['IRRADIATION'].ffill()
df_merged['AMBIENT_TEMPERATURE'] = df_merged['AMBIENT_TEMPERATURE'].ffill()
df_merged['MODULE_TEMPERATURE'] = df_merged['MODULE_TEMPERATURE'].ffill()


# 绘制日周期曲线图
# ---------- 设置中文字体（避免图表中文乱码） ----------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 图1：前5天逐日功率曲线 ----------
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
# 按天分组（避免跨天）
df_merged['DATE_ONLY'] = df_merged['DATE_TIME'].dt.date
unique_days = df_merged['DATE_ONLY'].unique()[:5]  # 只取前5天

for i, day in enumerate(unique_days):
    day_data = df_merged[df_merged['DATE_ONLY'] == day]
    axes[i].plot(day_data['DATE_TIME'], day_data['TOTAL_AC_POWER'],
                 color='blue', linewidth=0.8)
    axes[i].set_ylabel('AC Power (kW)')
    axes[i].set_title(f'日期: {day}')
    axes[i].grid(True, alpha=0.3)

plt.xlabel('时间')
plt.suptitle('前5天逐日交流功率曲线（每15分钟一点）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(img_dir,'daily_power_curve.png'), dpi=150)  # 保存图表用于论文
plt.show()

# ---------- 图2：辐照度 vs 功率散点图 ----------
plt.figure(figsize=(6, 5))
# 只选取白天（辐照度>0.05 避免夜间零值堆积）
daytime = df_merged[df_merged['IRRADIATION'] > 0.05]
plt.scatter(daytime['IRRADIATION'], daytime['TOTAL_AC_POWER'],
            alpha=0.5, s=5, c='orange')
plt.xlabel('辐照度 (kW/m^2)')
plt.ylabel('交流功率 (kW)')
plt.title('辐照度与发电功率关系（白天数据）')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(img_dir,'irr_vs_power.png'), dpi=150)
plt.show()


# 数据质量速查表
print("===== 数据质量速查 =====")
print(f"总时间步数: {len(df_merged)}")
print(f"时间频率: {df_merged['DATE_TIME'].diff().mode()[0]}")  # 常见时间间隔
print(f"总功率范围: {df_merged['TOTAL_AC_POWER'].min():.2f} ~ {df_merged['TOTAL_AC_POWER'].max():.2f} kW")
print(f"辐照度范围: {df_merged['IRRADIATION'].min():.2f} ~ {df_merged['IRRADIATION'].max():.2f} W/m²")
print(f"夜间零值占比: {(df_merged['TOTAL_AC_POWER']==0).mean():.1%}")

# 检查是否有功率为负值（理论上不应该）
if (df_merged['TOTAL_AC_POWER'] < 0).any():
    print("⚠️ 警告：发现负功率值，需要进一步清洗！")
else:
    print("✅ 未发现负功率值。")

# 检查时间戳是否严格单调递增
if df_merged['DATE_TIME'].is_monotonic_increasing:
    print("✅ 时间戳单调递增，滑动窗口构建安全。")
else:
    print("⚠️ 时间戳无序！必须按 DATE_TIME 排序。")
    df_merged.sort_values('DATE_TIME', inplace=True)

# 保存清洗后的数据（方便下一步直接用）
df_merged.to_csv('archive/plant_1_cleaned.csv', index=False)
print("清洗后数据已保存至 archive/plant_1_cleaned.csv")