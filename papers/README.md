# SmallMatPrep 相关论文

> 生成日期: 2026-06-10

---

## ✅ 已下载

| # | 论文 | 文件 | 备注 |
|---|------|------|------|
| 1 | **Small Data Machine Learning in Materials Science** — Xu et al. 2023, *npj Computational Materials* (被引 774) | `Xu2023_SmallDataML_Materials.pdf` | 开放获取 ✅ |
| 2 | **A Strategy to Apply Machine Learning to Small Datasets in Materials Science** — Zhang & Ling 2018, *npj Computational Materials* (被引 917+) | `../paper.pdf` | 已有，开放获取 ✅ |
| 3 | **OverNaN: NaN-Aware Oversampling for Imbalanced Learning with Meaningful Missingness** — Barnard 2026, arXiv:2605.11525 | `OverNaN_2026.pdf` | arXiv 免费 ✅ |
| 4 | **Predicting Missing Values: A Good Idea?** — van Buuren 2026, arXiv:2605.03733 | `PredictingMissingValues_vanBuuren_2026.pdf` | arXiv 免费 ✅ |
| 5 | **Meta-Imputation Balanced (MIB): An Ensemble Approach for Handling Missing Data** — Azad et al. 2025, arXiv:2509.03316 | `MIB_MetaImputation_2025.pdf` | arXiv 免费 ✅ |
| 6 | **Bridging Design Gaps: A Parametric Data Completion Approach with Graph Guided Diffusion Models** — Zhou et al. 2024, IDETC, arXiv:2406.11934 | `BridgingDesignGaps_Diffusion_2024.pdf` | arXiv 免费 ✅ |
| 7 | **GAIN: Generative Adversarial Imputation Networks** — Yoon et al. 2018, *ICML* (被引 2000+) | `GAIN_ICML2018.pdf` | arXiv 免费 ✅ |
| 8 | **HyperMM: Robust Multimodal Learning with Varying-sized Inputs** — Chaptoukaev et al. 2024, arXiv:2407.20768 | `HyperMM_Multimodal_2024.pdf` | arXiv 免费 ✅ |

---

## ⛔ 付费墙未能下载（需手动获取）

> 以下论文在付费墙后，且无法通过脚本自动下载。请通过机构访问权限或以下链接手动获取。

| # | 论文 | 链接 | 建议获取方式 |
|---|------|------|------------|
| 1 | **MatImpute: Imputation of Missing Data in Materials Science through Nearest Neighbors and Iterative Predictions** — Xie et al. 2024, *JCTC* | https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c01237 | 机构订阅 / 联系作者 / ResearchGate |
| 2 | **MissForest—Non-parametric Missing Value Imputation for Mixed-Type Data** — Stekhoven & Bühlmann 2012, *Bioinformatics* (被引 7296) | https://academic.oup.com/bioinformatics/article-abstract/28/1/112/219101 | 机构订阅 / R包文档自带说明 |
| 3 | **Uncertainty-Quantified Primary Particle Size Prediction in Li-Rich NCM Materials via ML and Chemistry-Aware Imputation** — Madika et al. 2026, *Advanced Science* | https://onlinelibrary.wiley.com/doi/abs/10.1002/advs.202515694 | Wiley / 开放获取可能性需确认 |
| 4 | **Artificial Intelligence for Materials Discovery, Development, and Optimization** — Madika et al. 2025, *ACS Nano* (被引 128) | https://pubs.acs.org/doi/abs/10.1021/acsnano.5c04200 | 机构订阅 |

---

## 📋 论文与 impute 模块的对应关系

| SmallMatPrep 模块 | 相关论文 | 关系 |
|------------------|---------|------|
| `impute_median` | — | 通用 baseline，无数论文使用 |
| `impute_knn` | MatImpute (Xie 2024) | KNN 是 MatImpute 第一阶段 |
| `impute_group_median` | — | **你的独特贡献**：材料领域分组先验 |
| `KNNExtraTreesImputer` | MissForest (Stekhoven 2012), MatImpute (Xie 2024) | MissForest用RF→你用ExtraTrees；MatImpute用KNN+迭代 |
| impute + CEP 组合 | Zhang & Ling (2018), Xu et al. (2023) | **你的最大创新点**：插补 + 物理先验协同 |
