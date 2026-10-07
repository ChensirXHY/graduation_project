# 基于时序 Transformer 的光伏发电功率预测

> 用历史 24 小时的气象与发电数据，预测未来 6 小时的光伏输出功率

![CI](https://github.com/ChensirXHY/graduation_project/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C)
![License](https://img.shields.io/badge/license-MIT-green)

![预测效果](results/model_comparison.png)

---

## 项目简介

光伏发电受辐照度、温度、云层等因素影响，输出功率波动剧烈，给电网调度带来压力。
准确预测短期功率是提升光伏消纳率的关键。

本项目基于印度两座光伏电站（Plant 1，22 台逆变器，峰值约 29 MW）的真实运行数据，
构建 **历史 24 小时（96 点）→ 未来 6 小时（24 点）** 的多步预测模型，
并与持续预测（naive）、LightGBM、LSTM 三个基线方法系统对比。

**本科毕业设计** ｜ 作者：江林岩 ｜ 安徽新华学院 人工智能专业

---

## 主要结果

<!-- 数字来自 05_evaluate_compare.py 生成的 results/metrics.csv，可复现 -->

| 模型 | MAE (kW) | RMSE (kW) | MAPE (%) | R² |
|---|---|---|---|---|
| 持续预测（naive） | 5710.31 | 8734.47 | 163.27 | -0.111 |
| LightGBM | 1994.23 | 3524.27 | 38.04 | 0.819 |
| LSTM | 1861.92 | 3155.07 | 44.41 | 0.855 |
| **Informer（本文）** | **1877.44** | **3054.82** | **40.71** | **0.864** |
| Transformer（普通注意力，消融） | 1859.26 | 3025.34 | 39.45 | 0.867 |

**结论**：

- Informer 对比 LightGBM：**MAE −5.9%，RMSE −13.3%，R² +4.5pp**，显著更优；
- Informer 对比 LSTM：MAE 相当（+0.8%），**RMSE −3.2%，R² +0.9pp**，
  且注意力复杂度为 O(L·log L) 而非 LSTM 的逐步递推；
- 消融发现：普通注意力 Transformer 在本数据集（96 步）上与 Informer 持平略优，
  ProbSparse 稀疏注意力的收益主要在长序列场景（论文原数据为 720 步）——
  这是论文"讨论"一章的现成素材（见「已知限制」）。

> **测试集口径**：2020-06-12 之后的连续数据，455 个窗口，按时间顺序切分（非随机）。
> **复现命令**：依次运行 `03b_baseline_lightgbm.py`、`04_train_informer.py`、
> `04a_train_transformer.py`、`05_evaluate_compare.py`，
> 指标表自动写入 `results/metrics.csv`。
> **调参方法**：三组超参扫描共 21 个配置，最终配置按**验证集损失**选择
> （best_val 0.00517 为全部配置最低），测试集仅用于最终报告，未被用于选参。

---

## 快速开始

```bash
# 1. 安装依赖（版本已锁定）
pip install -r requirements.txt

# 2. 获取数据（约 40 MB，未纳入版本控制）
#    Kaggle: "Solar Power Generation Data"
#    https://www.kaggle.com/datasets/anikannal/solarpowergeneration
#    下载后把 4 个 CSV 放入 archive/：
#      Plant_1_Generation_Data.csv   Plant_1_Weather_Sensor_Data.csv
#      Plant_2_Generation_Data.csv   Plant_2_Weather_Sensor_Data.csv

# 3. 按顺序运行数据管线
python 01_explore_quality.py          # 数据探索与质量报告（约 30 分钟）
python 02_preprocess_windows.py       # 清洗 + 特征工程 + 滑动窗口（约 40 分钟）
python 03_baseline_lstm.py            # LSTM 基线（约 1 小时，GPU 几分钟）
python 03b_baseline_lightgbm.py       # LightGBM 基线（几分钟）
python 04_train_transformer.py        # Transformer 主模型（GPU 约 5 分钟）
python 05_evaluate_compare.py         # 多模型对比与可视化

# 4. 跑测试（不需要数据文件，几秒完成）
pip install -r requirements-dev.txt
pytest tests/ -v
```

**模型权重**：训练产物不入库（见 `.gitignore`）。可从
[Releases](https://github.com/ChensirXHY/graduation_project/releases) 下载，
或按上述管线自行训练。

---

## 技术方案

### 数据管线

```
Kaggle 原始 CSV ──01──▶ 质量报告 + 清洗 ──02──▶ 特征工程 + 归一化 + 滑动窗口
                                                            │
                                    ┌───────────────────────┤
                                    ▼                       ▼
                            03 LSTM 基线            03b LightGBM 基线
                                    │                       │
                                    ▼                       ▼
                    04a 普通 Transformer（消融） ◀── 04 Informer（本文）
                                    └───────────05 统一对比──▶
                                            metrics.csv
```

### 数据处理要点

| 环节 | 做法 | 为什么 |
|---|---|---|
| 时间解析 | 显式指定 format | **发电数据是 `DD-MM-YYYY`，气象数据是 `YYYY/MM/DD`**。不指定格式时 pandas 会把 `01-02-2020` 误判为 1 月 2 日（实际是 2 月 1 日），导致整条时间轴错位且不报错 |
| 多逆变器聚合 | 按时间求和 | 原始数据是 22 台逆变器明细，需聚合为全场总功率 |
| 缺失值 | 时间线性插值 | 气象量是连续物理量；前向填充在连续缺失时产生阶梯状假数据 |
| 异常值 | IQR 法裁剪 | 传感器偶发跳变；功率下界额外约束为 0 |
| 时间特征 | 小时 sin/cos 编码 | 避免 23 点与 0 点的数值断层（0 与 23 数值上相距最远，语义上却相邻） |
| 滞后特征 | 1 小时滚动均值/标准差 | 捕捉短时趋势与波动性 |

### 数据集切分（关键设计）

```
时间轴 ─────────────────────────────────────────────────▶
       │◀── 训练集 ~70% ──▶│◀─ 验证 ~5 天 ─▶│◀─ 测试集 ─▶│
       2020-05-16 ~ 06-07   06-07 ~ 06-12    06-12 之后
```

**严格按时间顺序切分，绝不随机切分。** 随机切分会让未来样本混入训练集，
造成测试指标虚高的假象 —— 这是时序预测论文最常见的硬伤。
`tests/test_data.py::test_no_shuffle_keeps_temporal_order` 专门守护这条不变量。

### 模型结构（本文方法：Informer）

参考 Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting"（AAAI 2021），本实现忠实于论文的三个核心机制：

```
输入 (batch, 96, 12)                      开始标记 (batch, 24, 12) + 全零占位 (24)
   │                                            │
   ├─ 线性投影 → d_model=64                     ├─ 线性投影
   ├─ 正弦位置编码                              ├─ 位置编码
   │                                            │
   ├─ ProbSparse 自注意力 ◄── 只对 top-u 活跃 query 算全量注意力
   │        │ 蒸馏（Conv1d+ELU+MaxPool，序列减半）
   ├─ 96 → 48 ──► ProbSparse 自注意力
   │        │ 蒸馏
   ├─ 48 → 24 ──► ProbSparse 自注意力
   │                                            │
   ▼ 编码器记忆 (batch, 24, 64)                 ├─ 因果自注意力（下三角掩码）
   └──────────────────交叉注意力─────────────────┤
                                                 ▼
                                        Linear → (batch, 24)
```

**ProbSparse 自注意力**（论文第 3.1 节）：普通自注意力 O(L²) 且多数 query
的注意力分布接近均匀。论文提出用稀疏度度量
M(qi,K) = max_j(qi·kjᵀ/√d) − mean_j(qi·kjᵀ/√d) 选出少数"活跃" query，
只对它们计算全量注意力，其余 query 取 value 均值 —— 效果几乎无损，
复杂度降至 O(L·logL)。

**自注意力蒸馏**（论文第 3.2 节）：每层编码后接 Conv1d + ELU + MaxPool，
序列长度减半（96→48→24），上层只关注粗粒度表示。

**生成式解码器**（论文第 3.3 节）：输入"开始标记"（历史窗口最后 24 步）
+ 全零占位符，因果自注意力 + 交叉注意力后，一次前向生成全部 24 步预测。

**两处如实记录的实现取舍**：
1. 解码器自注意力用普通因果注意力（序列仅 48 步，O(L²) 可忽略，
   论文第 3.1 节消融也表明解码器稀疏化收益很小）；
2. 残差目标：模型预测 y − 最后一步功率 的增量，推理时加回基线 ——
   实测把 MAE 从 1888 降到 1865 kW（普通 Transformer 上的对照）。

### 消融对照（普通注意力 Transformer）

`04a_train_transformer.py` 训练与 Informer 共享同一数据与训练协议的
普通注意力 Transformer（d_model=64 × 3 层，mean+last 拼接池化），
用于隔离 ProbSparse 机制本身的贡献 —— 见 `results/metrics.csv` 对比。

### 训练策略

- 损失：MSE（对大幅误差更敏感，符合电网调度对极端值的要求）
- 优化器：Adam（lr=1e-3, weight_decay=1e-4）+ 余弦退火
- **梯度裁剪（1.0）**：Transformer 训练必备，防注意力层梯度爆炸
- 早停：验证集 loss 连续 10 轮不降则停止
- 固定随机种子 42，全部随机源（Python/Numpy/torch/cuDNN）可复现

---

## 项目结构

```
graduation_project/
├── 01_explore_quality.py        # 数据探索与质量报告
├── 02_preprocess_windows.py     # 清洗 + 特征工程 + 滑动窗口
├── 03_baseline_lstm.py          # LSTM 基线
├── 03b_baseline_lightgbm.py     # LightGBM 基线（MultiOutputRegressor）
├── 04_train_informer.py         # Informer 主模型（本文方法）
├── 04a_train_transformer.py     # 普通注意力 Transformer（消融对照）
├── 05_evaluate_compare.py       # 多模型对比 → results/metrics.csv
├── configs/
│   └── base.yaml                # 全部超参与路径（消融实验只改这里）
├── src/
│   ├── data.py                  # 窗口构造 / 时间切分 / 指标计算
│   ├── models.py                # LSTM / Transformer / Informer 模型定义
│   ├── config.py                # YAML 配置加载
│   └── logger.py                # 统一日志
├── tests/                       # 37 项单元测试（不依赖真实数据）
├── archive/                     # 原始数据与中间产物（.gitignore 已排除）
├── models/                      # 模型权重（改用 Releases 分发）
├── results/                     # 图表与 metrics.csv（metrics.csv 入库）
└── requirements.txt
```

---

## 测试

```bash
pytest tests/ -v                    # 单元测试，几秒，不需要数据
pytest tests/ -v --cov=src          # 覆盖率报告（本地含模型测试 99%+）
ruff check .                        # 静态检查
ruff format --check .               # 格式检查
pre-commit install                  # 提交前自动拦截大文件与脏代码
```

| 项目 | 结果 |
|---|---|
| 单元测试 | **30 通过** |
| 覆盖率（本地） | **99%**（CI 无 torch 时约 72%） |
| ruff check / format | 全部通过 |
| CI | 三个 job（ruff / pytest / 依赖可安装性） |

测试设计原则：**不依赖真实数据集与模型权重**，用合成序列验证
"窗口构造 / 时间切分 / 指标计算 / 模型形状契约" 四类核心逻辑。
其中 `test_no_future_leakage` 守护的是时序预测最容易出错、
也最致命的数据泄漏问题。

---

## 复现性说明

- [x] 依赖版本锁定（`requirements.txt`）
- [x] 随机种子固定为 42（Python / NumPy / torch / cuDNN）
- [x] 超参集中在 `configs/base.yaml`，与结果一同归档
- [x] 数据集切分方式在代码与文档中明确（时间顺序，非随机）
- [x] 关键逻辑有单元测试守护（数据泄漏、窗口对齐、模型形状契约）
- [x] 训练日志落盘至 `logs/train.log`

---

## 已知限制与后续工作

**限制**

- 仅覆盖印度两座电站，跨地域泛化能力未验证
- 未引入数值天气预报（NWP），本质上是"基于历史与实时气象"的短期预测
- 未显式建模云量，阴雨天气误差显著偏高
- 单点预测，未做区间预测（对调度而言，置信区间往往比点预测更有价值）
- LightGBM 将窗口展平为 1152 维向量，无法利用序列结构 —— 这既是
  它的劣势（对比价值所在），也意味着它学到的只是"输入统计量 → 未来均值"
- **蒸馏在小序列上无益**：消融实测 distil=True 时 MAE 1904.7，
  关闭后 1877.4。蒸馏的收益在 720 步级长序列上才体现（论文原场景），
  本任务序列仅 96 步，序列减半丢失了细粒度信息
- **ProbSparse 在短序列上无增益**：普通注意力 Transformer 在本数据集上
  与 Informer 持平略优（MAE 1859 vs 1877）。Informer 的优势在于
  O(L·log L) 的注意力复杂度，这在 L ≥ 数百步时才转化为实际收益

**后续工作**

- [ ] 引入 NWP 预报数据，将预测时效从 6 小时延长至 24 小时
- [ ] 用更长输入窗口（192/336 步）重训，验证蒸馏与 ProbSparse 的收益
- [ ] 分位数损失输出预测区间
- [ ] 尝试 PatchTST / iTransformer 等更新结构
- [ ] 消融实验：逐项移除特征，量化各特征贡献

---

## License

MIT ｜ 数据集版权归 Kaggle 原作者（Anikannal）所有
