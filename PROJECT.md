# SmallMatPrep

## 📋 项目信息

| 属性 | 内容 |
|------|------|
| 名称 | SmallMatPrep |
| 描述 | 小样本材料数据诊断、领域感知插补与评估工具包 |
| 状态 | 🟢 开发中 |
| 优先级 | 高 |
| 环境 | Python >= 3.9, pandas, numpy, scikit-learn |
| 仓库 | D:\09-项目\material_ml_paper\v3_python |
| 最后更新 | 2026-06-03 |

---

## ✅ 已完成

- [x] 项目基础结构搭建（data / inspect / impute / modeling / evaluate / report）
- [x] 最小可运行 demo（examples/run_minimal_demo.py）
- [x] 配置文件模板与数据 schema
- [x] Git 初始化
- [x] 安装 Kimi WebBridge + MCP Servers（Playwright / GitHub）
- [x] MatImpute-lite（KNN 初始化 + ExtraTrees 迭代插补）
- [x] CEP 理论先验特征工程（add_cep_feature + simple_electrolyte_cep）
- [x] MatImpute-lite + CEP 对比 demo（examples/run_matimpute_cep_demo.py）
- [x] 真实电解质 Excel 数据初步实验（bosailisi.xlsx + supplement.xlsx）
- [x] Bootstrap bias-variance 分解（evaluate/decompose.py + test_decompose.py）
- [x] 数据量敏感的模型推荐逻辑（modeling/recommend.py）
- [x] 小样本诊断功能（inspect/diagnosis.py：sample_diagnosis + missing_pattern_report）
- [x] 补充单元测试（test_core_modules / test_matimpute_cep / test_decompose / test_diagnosis / test_ai_module 共 5 个文件）

---

### 核心模块详细状态

| 模块 | 路径 | 职责 | 状态 | 测试 |
|------|------|------|------|------|
| **data** | `data/loaders.py` | CSV / JSON 加载 | ✅ 已完成 | `test_core_modules` |
| **inspect** | `inspect/missingness.py` | 缺失值统计报告 | ✅ 已完成 | `test_core_modules` |
| **inspect** | `inspect/diagnosis.py` | 小样本欠拟合诊断、缺失模式报告 | 🟡 基本完成（函数已实现，诊断建议可打磨） | `test_diagnosis.py` |
| **impute** | `impute/baseline.py` | 中位数填充 / KNN 插补 | ✅ 已完成 | `test_core_modules` |
| **impute** | `impute/material.py` | 分组中位数插补 | ✅ 已完成 | `test_core_modules` |
| **impute** | `impute/constraints.py` | **材料物理约束（成分守恒/范围/相似度/不确定性）** | ✅ 已完成 | `run_constrained_impute_demo.py` |
| **impute** | `impute/iterative.py` | **MatImpute-lite**（KNN 初始化 + ExtraTrees 迭代精修） | ✅ 已完成 | `test_matimpute_cep.py` |
| **features** | `features/cep.py` | **CEP 物理先验特征工程** | ✅ 已完成 | `test_matimpute_cep.py` |
| **modeling** | `modeling/recommend.py` | 数据量感知的模型推荐 | ✅ 已完成 | `test_core_modules` |
| **evaluate** | `evaluate/metrics.py` | MAE / RMSE / R² | ✅ 已完成 | `test_core_modules` |
| **evaluate** | `evaluate/decompose.py` | **Bootstrap bias-variance 分解** | ✅ 已完成 | `test_decompose.py` |
| **report** | `report/summary.py` | 结果汇总文字报告 | ✅ 已完成 | `test_core_modules` |
| **ai** | `ai/`（client / config_generator / diagnose_reporter） | AI 辅助配置生成和诊断报告 | 🟡 已完成（非论文主线，可选增强） | `test_ai_module.py` |
| **config_schema** | `config_schema.py` | 配置校验 | ✅ 已完成 | `test_core_modules` |

---

## 🔴 进行中

- [ ] 将真实电解质实验结果整理成论文表格和方法说明
- [ ] 补充端到端集成测试
- [ ] 在真实电解质数据上验证约束模块效果（预期能防止非物理值）

---

## ⬜ 待办

- [ ] 将真实电解质实验结果整理成论文表格和方法说明

---

## 🗂️ 当前项目结构速览

### 核心包：`smallmatprep/`

| 路径 | 作用 | 重要性 |
|------|------|--------|
| `smallmatprep/__init__.py` | 工具包总入口，导出 `load_csv`、`impute_median`、`impute_knn_extratrees`、`add_cep_feature` 等常用函数 | 核心 |
| `smallmatprep/data/loaders.py` | 加载 CSV 和 JSON 配置 | 基础 |
| `smallmatprep/inspect/missingness.py` | 统计缺失值数量和比例 | 基础 |
| `smallmatprep/impute/baseline.py` | 中位数填充、KNN 填充 | baseline 核心 |
| `smallmatprep/impute/material.py` | 分组中位数填充，适合按材料体系或类别补值 | 辅助 |
| `smallmatprep/impute/iterative.py` | MatImpute-lite：KNN 初始化 + ExtraTrees 迭代精修 | 新增核心 |
| `smallmatprep/features/cep.py` | CEP/proxy 特征工程：加入理论或廉价实验参考值 | 新增核心 |
| `smallmatprep/modeling/recommend.py` | 根据样本数和特征数推荐模型 | 辅助 |
| `smallmatprep/evaluate/metrics.py` | 计算 MAE、RMSE、R2 | 评估核心 |
| `smallmatprep/report/summary.py` | 汇总缺失报告、模型结果和推荐信息 | 辅助 |
| `smallmatprep/ai/` | AI 辅助配置生成和诊断报告 | 可选增强，不是当前论文主线 |

### 实验脚本：`examples/`

| 路径 | 作用 | 当前用途 |
|------|------|----------|
| `examples/run_minimal_demo.py` | 最小流程演示：加载数据、缺失报告、中位数填充、建模评估 | 入门 demo |
| `examples/run_matimpute_cep_demo.py` | synthetic 电解质小样本实验，对比 baseline、MatImpute-lite、MatImpute-lite + CEP | 方法有效性演示 |
| `examples/run_real_electrolyte_demo.py` | 读取导师真实 Excel 数据，清洗 `bosailisi.xlsx` 和 `supplement.xlsx`，测试真实配方数据策略 | 论文实验主线 |
| `examples/run_ai_demo.py` | AI 配置生成演示 | 工具增强，可选 |
| `examples/sample_electrolyte.csv` | 最小样例数据 | 入门 |
| `examples/config_template.json` | 配置模板 | 入门 |

### 测试与文档

| 路径 | 作用 | 重要性 |
|------|------|--------|
| `tests/test_core_modules.py` | 测试加载、缺失诊断、基础插补、模型推荐和评估 | 重要 |
| `tests/test_matimpute_cep.py` | 测试 MatImpute-lite 和 CEP 新功能 | 重要 |
| `tests/test_ai_module.py` | 测试 AI 模块 | 可选 |
| `README.md` | 给外部用户看的安装、用法和模块介绍 | 重要 |
| `PROJECT.md` | 项目状态、实验结果、决策日志和下一步计划 | 当前上下文核心 |
| `pyproject.toml` / `requirements.txt` | 包配置与依赖声明 | 工程基础 |
| `paper.pdf` / `paper.txt` / `paper_extracted.txt` | Zhang & Ling 2018 文献材料和提取文本 | 论文背景参考 |

### 当前主线理解

这个项目现在有两条实验线：

1. synthetic demo：证明“真正缺失”的数值特征可以用 MatImpute-lite + CEP 明显提升预测。
2. 真实 Excel 数据：发现 `A-X` 配方空白更像“组分未添加”，不是“未知缺失”，因此真实配方数据默认应使用 `rule zero`，而不是盲目插补。

当前最重要的研究结论：

- 配方列 `A-X`：空白应优先按 0 处理。
- 目标列 `P15/P17`：不填，缺失样本在对应目标实验中丢弃。
- MatImpute-lite：适合温度、密度、黏度等真正未测的连续特征，不适合直接补“未添加组分”。
- CEP/proxy：`P15 -> P17` 有小幅提升，但必须确认 `P15` 是否在预测 `P17` 前可获得、且是否成本更低。

---

## 📖 文献阅读记录

### Zhang & Ling 2018 — 小样本材料 ML 的经典策略

| 属性 | 内容 |
|------|------|
| 标题 | A strategy to apply machine learning to small datasets in materials science |
| 作者 | Ying Zhang, Chen Ling (Toyota Research Institute of North America) |
| 发表 | npj Computational Materials, 2018, 4:25 |
| DOI | 10.1038/s41524-018-0081-z |
| 被引 | 917+ |
| 阅读日期 | 2026-06-03 |
| 阅读状态 | ✅ 已读全文（8 页） |

**核心问题**：
材料数据集通常只有几十到几百条样本，导致 ML 模型预测不准。前人只观察到"数据越多越准"，但没有系统研究"数据少究竟如何影响模型"。

**关键发现**：
- 数据量对精度的影响不是直接的，而是通过**模型自由度（DoF）**介导
- 小数据训练时，精度提高的代价是更高模型复杂度 → 导致**欠拟合（underfitting）**
- 误差分解：bias² 是 variance 的 4 倍以上，说明模型太简单，抓不住规律

**解决方案：CEP（Crude Estimation of Property）**
- 在特征空间中引入对目标性质的**粗略估计**作为额外描述符
- CEP 可以来自：不准确的 DFT 计算、经验公式、廉价实验
- **关键洞察**：CEP 不需要准确，只要与目标性质有统计相关性（Pearson > ~0.5）即可
- 效果：案例 1（带隙预测）RMSE 降低 33%，且模型 DoF 从 12 降到 9

**三个验证案例**：
1. 二元半导体带隙（108 样本）— GGA-DFT 作为 CEP
2. 晶格热导率（93 样本）— Slack 经验模型作为 CEP
3. 沸石弹性模量（102 样本）— 经典力场作为 CEP

**对 SmallMatPrep 的启发**：
- inspect：诊断数据量是否触发"精度-DoF 关联"
- impute：插补时可用简单规则作为"粗略估计"辅助
- modeling：推荐模型时考虑数据规模，提示用户引入 CEP
- evaluate：Bootstrap 分解 bias/variance，定位问题是欠拟合还是过拟合

---

## 🧠 决策日志

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-06-03 | 引入 CEP（Crude Estimation）思路到 SmallMatPrep | 受 Zhang & Ling 2018 启发，小样本材料数据的核心痛点可通过"领域知识粗略估计"缓解，不需要增加模型复杂度 |
| 2026-06-03 | 安装 Kimi WebBridge + Playwright MCP | 用于自动化文献检索、网页数据抓取，提升研究效率 |
| 2026-06-03 | 采用 sklearn-only 的 MatImpute-lite + CEP MVP | 先用低依赖版本跑通"缺失修复 + 物理先验 + 对比验证" workflow，避免一开始引入元素指纹和额外材料库 |

---

## 🧪 实验记录

### 2026-06-03 — MatImpute-lite + CEP synthetic demo

命令：`python examples/run_matimpute_cep_demo.py`

| 组别 | 方法 | RMSE | MAE | R2 | 相对 baseline 的 RMSE 降幅 |
|------|------|------|-----|----|---------------------------|
| A | 中位数插补 | 2.919 | 2.438 | 0.787 | 0.0% |
| B | MatImpute-lite | 2.236 | 1.845 | 0.871 | 23.4% |
| C | MatImpute-lite + CEP | 1.237 | 0.831 | 0.960 | 57.6% |

结论：在可复现的 synthetic electrolyte 小样本 demo 中，MatImpute-lite 能明显改善缺失值处理，加入 CEP 后预测误差进一步下降。该结果用于验证 workflow，不等同于真实实验数据结论。

### 2026-06-03 — 真实电解质 Excel 初步实验

数据：`v2/data/bosailisi.xlsx` + `v2/data/supplement.xlsx`

清洗规则：`bosailisi.xlsx` 使用第 2 行作为真实表头、第 5 行开始作为样本；与 `supplement.xlsx` 合并。配方列使用 `A-X`，体系列使用 `CA/GR/体系方案`，目标列测试 `P15/P17`。

数据规模：合并后 68 条，`P15/P17` 同时有效 62 条；`A-X` 空白率 50.4%；`P15/P17` Pearson 相关系数 0.576。

| 目标 | 策略 | RMSE | MAE | R2 | 相对 rule zero 的 RMSE 降幅 |
|------|------|------|-----|----|-----------------------------|
| P17 | rule zero | 163.71 | 127.56 | 0.340 | 0.0% |
| P17 | median as missing | 164.85 | 127.47 | 0.309 | -0.7% |
| P17 | KNN as missing | 164.84 | 126.66 | 0.325 | -0.7% |
| P17 | MatImpute-lite | 179.85 | 145.45 | 0.193 | -9.9% |
| P17 | rule zero + P15 proxy | 157.43 | 123.12 | 0.385 | 3.8% |
| P15 | rule zero | 159.98 | 109.92 | 0.512 | 0.0% |
| P15 | median as missing | 154.18 | 112.99 | 0.540 | 3.6% |
| P15 | KNN as missing | 153.84 | 112.40 | 0.548 | 3.8% |
| P15 | MatImpute-lite | 168.48 | 121.77 | 0.425 | -5.3% |
| P15 | rule zero + P17 proxy | 160.23 | 113.65 | 0.506 | -0.2% |

结论：真实配方数据中，`A-X` 空白更像"组分未添加"而不是"未知缺失"，因此默认应使用 `rule zero`。MatImpute-lite 不适合直接用于这类配方空白；它更适合温度、密度、黏度等真正未测的连续特征。`P15` 作为 `P17` 的 proxy/CEP-like 特征有小幅提升，但是否合理取决于 `P15` 是否比 `P17` 更便宜或更早可获得。

---

### 2026-06-03 — 材料约束填充对比实验（constrained impute benchmark）

命令：`python examples/run_constrained_impute_demo.py`

数据：合成电解质数据（n=72, 特征缺失率 ~10%）
验证：5-fold CV, 下游模型 RandomForest

| # | 方法 | RMSE | MAE | R2 | ImputeRMSE | RMSE提升 |
|---|------|------|-----|----|-----------|---------|
| ① | Median (baseline) | 2.9190 | 2.4383 | 0.7874 | 4.3603 | +0.00% |
| ② | MatImpute-lite | 2.2362 | 1.8448 | 0.8715 | 0.7177 | +23.39% |
| ③ | + Range constraints | 2.2362 | 1.8448 | 0.8715 | 0.7177 | +23.39% |
| ④ | + Range + Similarity | 2.3102 | 1.9160 | 0.8661 | 0.8046 | +20.86% |
| ⑤ | MatImpute-lite + CEP | 1.2368 | 0.8311 | 0.9599 | 0.7177 | +57.63% |
| ⑥ | + Constraints + CEP ★ | 1.3532 | 0.9299 | 0.9538 | 0.8046 | +53.64% |
| ⑦ | + Uncertainty + CEP | 1.2310 | 0.8213 | 0.9603 | 0.7218 | +57.83% |

**关键发现**：

1. **Range 约束在合成数据上无效果（③=②）**：因为合成数据的真实值已天然满足物理范围约束。预期在真实数据上 Range 约束可防止插补产生负密度、负浓度等不合理值。
2. **Similarity 约束轻微降低性能（④<②）**：KNN 平滑将插补值拉向邻居均值，会抹去合理方差。对于小样本数据，该约束过强。
3. **CEP 特征仍是主导因素（⑤ :独一档 +57.63%）**：符合 Zhang & Ling 2018 的核心发现。
4. **约束 + CEP（⑥=53.64%）低于纯 CEP（⑤=57.63%）**：约束将插补值推向物理合理区域，但与 MSE 最优解存在偏移。
5. **不确定性输出（⑦=+57.83%）略优于单次 CEP（⑤=+57.63%）**：多轮不同随机种子的 ExtraTrees 平均后方差降低，边际提升 0.20 个百分点。

**结论**：物理约束不提升合成数据性能（因为数据本身已"物理合理"），但对真实数据可防止非物理值。约束宜作为**安全网**而非性能优化手段。CEP 是提升小样本预测精度的核心策略。

---

## ⚠️ 踩坑记录

### 约束模块开发

- 行列式：numpy 数组只读（read-only）问题。`apply_range_constraint`、`apply_similarity_constraint`、`apply_composition_constraint` 中需使用 `.to_numpy(dtype=float, copy=True)` 避免赋值错误。
- 不确定性 bootstrap 需要每次调用使用不同 random_state，否则多轮结果完全相同。`impute_with_uncertainty` 计算了 `seed` 但未传给 `imputer_fn`，需在闭包中手动管理种子递增。

---

## 🔗 相关资源

- 论文 PDF：`paper.pdf`
- 论文提取文本：`paper.txt`、`paper_extracted.txt`
- 最小 demo：`python examples/run_minimal_demo.py`
- synthetic 方法 demo：`python examples/run_matimpute_cep_demo.py`
- 真实数据 demo：`python examples/run_real_electrolyte_demo.py`
- AI demo：`python examples/run_ai_demo.py`（需设置对应 API key）
