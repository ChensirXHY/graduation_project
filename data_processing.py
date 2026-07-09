#数据探索和生成质量报告
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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