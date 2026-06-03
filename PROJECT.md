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

---

## 🔴 进行中

- [ ] 领域感知插补策略设计（CEP 思路引入）
- [ ] inspect 模块：小样本欠拟合诊断功能

---

## ⬜ 待办

- [ ] 实验：在电解质数据集上测试 CEP 增强特征的效果
- [ ] evaluate 模块：Bootstrap bias-variance 分解
- [ ] modeling 模块：数据量敏感的模型推荐逻辑
- [ ] 补充单元测试

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

---

## ⚠️ 踩坑记录

（暂无）

---

## 🔗 相关资源

- 论文 PDF：`paper.pdf`import smallmatprep as smp
import os

# 方式1：OpenAI
api_key = "sk-..."
report = smp.ai.diagnose(df, target="conductivity", api_key=api_key)

# 方式2：Kimi
api_key = os.getenv("MOONSHOT_API_KEY")
config = smp.ai.generate_config(
    "预测电解质电导率，温度列缺失严重",
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
    model="moonshot-v1-8k",
)

# 方式3：直接运行 demo
# python examples/run_ai_demo.py   （需先设置 OPENAI_API_KEY 环境变量）（Zhang & Ling 2018，2026-06-03 下载）
