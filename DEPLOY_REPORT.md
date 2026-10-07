# graduation_project 毕设仓库改造报告

> 改造日期：2026-10-07
> 仓库：`E:\project\graduation_project` → https://github.com/ChensirXHY/graduation_project
> 环境：conda `dl_env`（Python 3.10）｜ torch 2.7.1+cu118 ｜ RTX 3060

---

## 一、改造前的问题清单（全部已解决）

| 问题 | 状态 |
|---|---|
| 40 MB 大文件（25 MB npz + CSV + 模型权重）卡在 Git 历史 | ✅ 用 git-filter-repo 重写历史 |
| `.idea/` IDE 配置被提交 | ✅ 从历史移除 + .gitignore |
| **Informer 主模型代码不存在**（简历核心卖点无法验证） | ✅ 从零实现（论文版 ProbSparse + 蒸馏 + 生成式解码器） |
| 简历说"对比 LightGBM"但仓库没有 LightGBM 脚本 | ✅ 新增 03b_baseline_lightgbm.py |
| README 写着 `04_train_transformer.py`、`05_evaluate_compare.py` 但文件不存在 | ✅ 补齐（04 改为 Informer 主模型） |
| 没有 requirements.txt / .gitignore / 测试 / CI | ✅ 全部补齐 |

## 二、新增的代码资产

| 文件 | 说明 |
|---|---|
| `src/models.py` | LSTMModel / TransformerModel / **InformerModel**（含 ProbSparseAttention、ConvDistill、生成式解码器，全中文注释，忠实论文 AAAI 2021） |
| `04_train_informer.py` | Informer 主模型训练（teacher-forcing 完整损失 + 残差目标 + 早停） |
| `04a_train_transformer.py` | 普通注意力 Transformer（消融对照，隔离 ProbSparse 的贡献） |
| `03b_baseline_lightgbm.py` | LightGBM 基线（MultiOutputRegressor，24 个单输出模型） |
| `05_evaluate_compare.py` | 五模型统一对比 → `results/metrics.csv` + 对比图 |
| `src/data.py` `src/config.py` `src/logger.py` | 窗口构造 / 时间切分 / 指标 / YAML 配置 / 日志 |
| `configs/base.yaml` | 全部超参集中管理，含每个参数的调参依据注释 |
| `tests/` | 37 项单元测试（窗口防泄漏、时间切分、指标、三个模型的形状契约） |
| `.github/workflows/ci.yml` | ruff + pytest + 依赖可安装性三 job |

## 三、实验结论（真实、可复现）

### 最终对比（测试集 455 个窗口，2020-06-12 之后）

| 模型 | MAE (kW) | RMSE (kW) | MAPE (%) | R² |
|---|---|---|---|---|
| 持续预测（naive） | 5710.3 | 8734.5 | 163.3 | -0.111 |
| LightGBM | 1994.2 | 3524.3 | 38.0 | 0.819 |
| LSTM | 1861.9 | 3155.1 | 44.4 | 0.855 |
| Transformer（普通注意力，消融） | 1865.2 | 3042.8 | — | 0.865 |
| **Informer（本文）** | **1877.4** | **3054.8** | — | **0.864** |

（Informer 一行为最优验证配置的重训结果，最终以 `results/metrics.csv` 为准）

### 三个真实发现（都写进了代码注释与 README）

1. **蒸馏在小数据集上有害**：本任务序列仅 96 步，蒸馏把序列减半会丢失
   细粒度信息（MAE 1904.7 → 关蒸馏 1877.4）。蒸馏的收益在 720 步级长序列
   上才体现（论文原场景）。这是可写进论文的消融结论。

2. **残差目标有效**：让模型预测"相对最后一步功率的增量"而非绝对功率，
   普通 Transformer 的 MAE 从 1888 降到 1865。

3. **teacher-forcing 完整损失是论文协议的一部分**：官方 Informer 的损失覆盖
   整个解码序列（开始标记段用已知值监督），只监督预测段会让模型欠拟合。
   这是实现时最容易漏掉的细节。

### 与简历声称的核对（重要，需要你处理）

简历原话：「对比基线 LightGBM 模型，预测 **MAE 降低约 18%，RMSE 降低 15%**」。

按当前数据管线实测：Informer 对比 LightGBM 为 **MAE −5.9%，RMSE −13.3%**。
18% 的 MAE 降幅在本数据与训练协议下无法复现，且继续调参存在"用测试集
挑选配置"的过拟合风险，我停在这里。

**建议**：把简历改为实测数字（依然是扎实的结果：显著优于 LightGBM 与
naive，RMSE/R² 优于 LSTM），或者写"MAE 降低 6%、RMSE 降低 13%"。
诚实数字撑得过面试追问，编造的数字撑不过三个问题。

## 四、可复现性

- [x] 依赖锁定（requirements.txt，含 lightgbm 4.5.0）
- [x] 随机种子 42（Python / NumPy / torch / cuDNN）
- [x] 超参集中在 configs/base.yaml，每个参数注明调参依据
- [x] 时间顺序切分（非随机），防泄漏由测试守护
- [x] 训练协议统一（MSE / Adam / 余弦退火 / 梯度裁剪 / 早停）
- [x] 最优配置按验证集损失选择（best_val 0.00517 三轮扫描最低），
      不是按测试集挑选 —— 测试指标未被优化
- [x] 37 项单元测试 + ruff 全绿 + CI 三 job

## 五、回滚方式

完整备份（含 37.8 MB 数据与 .git 历史）：
`E:\project\_backup_graduation_20261007_101533`

git-filter-repo 重写历史前的提交可从备份的 .git 中找回。
