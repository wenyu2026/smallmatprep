# SmallMatPrep

面向**小样本材料科学数据**的 Python 工具包 — 缺失值诊断、智能插补、模型推荐、AI 辅助配置。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 为什么用 SmallMatPrep？

材料实验数据通常样本量小（n < 100）、缺失多、特征维度高。传统 ML 工具面向大数据设计，容易在小样本上过拟合。SmallMatPrep 专为这类场景优化：

- **保守策略** — 优先中位数插补、Ridge/SVR 等正则化模型
- **数据诊断** — 缺失模式分析、样本量评估
- **智能分组** — 按材料业务含义对特征分组
- **AI 辅助** — 自然语言生成配置、自动诊断报告

## 安装

```bash
# 基础安装
cd v3_python
pip install -e .

# 含 AI 功能（需要 openai 包）
pip install -e ".[ai]"
```

## 快速上手

### 纯代码模式

```python
import smallmatprep as smp

# 1. 加载数据 & 配置
df = smp.load_csv("your_data.csv")
cfg = smp.load_config("your_config.json")

# 2. 缺失值诊断
report = smp.missing_report(df)

# 3. 插补
df_clean = smp.impute_median(df, feature_cols=["col1", "col2"])

# 或：KNN 初始化 + ExtraTrees 迭代精修
df_clean = smp.impute_knn_extratrees(df, feature_cols=["col1", "col2"])

# 4. 加入 CEP 理论先验特征
df_clean = smp.add_cep_feature(df_clean, smp.simple_electrolyte_cep)

# 5. 模型推荐
recs = smp.recommend_model(n_samples=45, n_features=8)

# 6. 建模 & 评估
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
scores = smp.evaluate_model(y_test, pred)

# 7. 汇总
print(smp.build_summary(report, scores, recs[0]["model"]))
```

### AI 模式（自然语言生成配置）

```bash
$env:DEEPSEEK_API_KEY='sk-你的key'
python examples/run_ai_demo.py
```

运行后自动生成 `examples/my_ai_config.json`，可直接用于训练。

## 模块一览

| 模块 | 功能 | 主要函数 |
|------|------|----------|
| `data` | 数据加载 | `load_csv`, `load_config` |
| `inspect` | 缺失值诊断 | `missing_report` |
| `impute` | 插补策略 | `impute_median`, `impute_knn`, `impute_knn_extratrees`, `impute_group_median` |
| `features` | 物理先验特征 | `add_cep_feature`, `simple_electrolyte_cep` |
| `modeling` | 模型推荐 | `recommend_model` |
| `evaluate` | 模型评估 | `evaluate_model` |
| `report` | 报告汇总 | `build_summary` |
| `ai` | AI 辅助（可选） | `generate_config`, `generate_config_with_columns`, `diagnose`, `save_config` |

## 配置文件格式

```json
{
  "target": "conductivity",
  "id_column": "sample_id",
  "feature_groups": {
    "solvent": ["solvent_A", "solvent_B"],
    "salt": ["salt_S"]
  },
  "blank_as_zero_columns": ["additive_C"],
  "categorical_columns": [],
  "drop_columns": [],
  "impute": {
    "method": "median",
    "columns": {
      "additive_C": {"method": "zero"},
      "temperature": {"method": "group_median", "group_col": "solvent_A"}
    }
  },
  "model": {
    "task": "regression",
    "test_size": 0.2,
    "random_state": 42
  }
}
```

参考模板：`examples/config_template.json`

## 支持的 AI 厂商

| 厂商 | 推荐度 | base_url | 备注 |
|------|--------|----------|------|
| DeepSeek | ⭐⭐⭐ | `https://api.deepseek.com/v1` | 国内直连、便宜、中文好 |
| OpenAI | ⭐⭐ | 默认 | 需科学上网 |
| Kimi/Moonshot | ⭐⭐ | `https://api.moonshot.cn/v1` | 国产 |
| 通义千问 | ⭐⭐ | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 阿里云 |
| Ollama | ⭐ | `http://localhost:11434/v1` | 本地免费 |
| vLLM | ⭐ | `http://localhost:8000/v1` | 自部署 |

## 开发

```bash
# 运行测试
python -m pytest tests/ -v

# 运行演示
python examples/run_minimal_demo.py

# MatImpute-lite + CEP 对比演示
python examples/run_matimpute_cep_demo.py
```

## 许可证

MIT License
