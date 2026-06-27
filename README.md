# AI Approaches to Analysing Recruitment Demand: Machine Learning Insights from European Pharmaceutical Job Postings

**Author:** Kashmira Bhoir
**Institution:** GISMA University of Applied Sciences
**Program:** MSc Data Science, Artificial Intelligence and Digital Business
**Year:** 2025–2026

---

## Overview

This repository holds the technical implementation for my MSc dissertation, analysing the European pharmaceutical job market using NLP (Natural Language Processing) and Machine Learning.

The dataset combines three verified sources spanning an eight-year period: a large 2018 baseline extract from eMedCareers (Zenodo record 1228606), supplemented by two further extracts collected via JobSpikr in 2021 and 2026. After cleaning and pharma-relevance filtering, the combined dataset comprises **9,520 job postings**.

The study addresses three research questions using multi-task transformers, hierarchical clustering, and explainable ML (XGBoost + SHAP).

---

## Repository Structure
```
pharma-job-market-analysis/
│
├── notebooks/
│   ├── Data_Inspection.ipynb         # Raw data quality check (30,516 rows)
│   ├── Data_Cleaning.ipynb           # Cleaning pipeline : 9,520-rows, 16-columns dataset
│   ├── RQ1_Salary_Prediction.ipynb   # Multi-task transformer (job role + salary)
│   ├── RQ2_Hybrid_Roles.ipynb        # Hierarchical clustering (hybrid roles)
│   └── RQ3_Mismatch_Detection.ipynb  # XGBoost + SHAP (location-skill mismatch detection)
│
├── data/
│   ├── raw/                          # Combined raw extract (30,516 rows, 3 verified sources)
│   └── processed/                    # Cleaned dataset (9,520 rows, 16 columns)
│
├── results/
│   └── figures/                      # Output charts (RQ2, RQ3)
│
├── requirements.txt
└── README.md
```

---

## Research Questions

### RQ1 — Role-Salary Patterns
> To what extent can multi-task transformers accurately predict pharmaceutical job roles and associated salary levels from posting text?

**Method:** Sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim) → Keras dual-head network (role classification + salary regression), seeded for reproducibility (`tf.random.set_seed(42)`)
**Dataset:** 1,199 postings with both a top-5 role label and a parseable salary (train: 959, test: 240)
**Result:** 90.4% role-classification accuracy, €9,627 overall salary MAE
**Gap addressed:** Ather et al. (2024) and Agrawal et al. (2025) focused exclusively on the US AI/ML job market — this is the first multi-task transformer for joint role + salary prediction on European pharmaceutical job posting data.

### RQ2 — Hybrid Role Discovery
> What hybrid pharmaceutical roles emerge from hierarchical clustering of European job postings, and how do their salary and location profiles differ?

**Method:** Sentence-transformer embeddings → UMAP (384→50 dims) → Agglomerative clustering (Ward linkage), k selected by silhouette score (search range k=6–30); TF-IDF for cluster naming
**Dataset:** Full 9,520 cleaned postings
**Result:** k=30 (silhouette = 0.542); 19 of 30 clusters classified as hybrid roles (63%); salary range €36,933–€96,547 across clusters
**Gap addressed:** Puente Agueda (2024) applied NLP to job postings for gender analysis, not role/salary clustering; Lukauskas (2023) was limited to NLP job-ad segmentation in Lithuania — this discovers data-driven hybrid roles across the entire European pharma job market.

### RQ3 — Location-Skill Mismatch
> Which location-skill combinations exhibit labour-market mismatches detectable via explainable ML?

**Method:** Regex-based skill extraction against a curated pharma skill taxonomy → XGBoost regression → SHAP explainability
**Dataset:** 1,680 postings with a parseable EUR salary (train: 1,344, test: 336)
**Result:** €17,437 test MAE (R² = 0.420); top SHAP driver is Category (€9,184), ahead of Location (€5,875) and Job Type (€5,184); 34 of 336 test postings (10.1%) flagged as salary mismatches, average gap €23,568
**Gap addressed:** Catanese et al. (2023) used descriptive statistics on the Italian market only, with no explainability; Romanko & O'Mahony (2022) called for explainability in labour-market analysis without implementing it — this delivers an XGBoost + SHAP analysis across three verified European sources.

---

## Key Findings

| Finding | Detail |
|---|---|
| Total cleaned postings | 9,520 (from 30,516 raw rows, 3 sources) |
| Salary disclosed | 1,680 of 9,520 postings (17.6%) — remainder listed as "Competitive"/"Negotiable"/blank |
| RQ1 — role accuracy | 90.4% across 5 role categories (n=240 test set) |
| RQ1 — salary MAE | €9,627 overall; best per-role MAE €7,356 (Pharmaceutical, Healthcare and Medical Sales) |
| RQ2 — clusters | 30 clusters (silhouette 0.542); 19 hybrid (63%) |
| RQ2 — salary range | €36,933–€96,547 across clusters |
| RQ2 — largest cluster | Pharmaceutical Sales Rep (1,316 postings, €54,568 avg) |
| RQ2 — highest-paid cluster | Medical Affairs Manager (€96,547 avg) |
| RQ3 — top SHAP driver | Category (€9,184), ahead of Location (€5,875) |
| RQ3 — mismatch rate | 34 of 336 test postings (10.1%), avg gap €23,568 |
| RQ3 — model fit | R² = 0.420, test MAE €17,437 |

---

## Dataset

| Detail | Value |
|---|---|
| Raw rows (combined) | 30,516 |
| Sources | eMedCareers 2018 (Zenodo 1228606): 30,000 · JobSpikr sandbox 2021: 52 · JobSpikr API 2026: 464 |
| Duplicate rows removed | 20,848 |
| Non-pharma rows removed (keyword filter) | 145 |
| Final cleaned rows | 9,520 |
| Final columns | 16 |
| Countries represented | 14 individual countries, plus an aggregate "Europe (General)" category for postings without a specific country match |
| Salary disclosed | 1,680 rows (17.6%); mean €69,374, std €36,400 |

> Country mix is dominated by the United Kingdom and the "Europe (General)" aggregate category (together roughly 90% of postings prior to the final pharma-relevance filter), with the remainder spread across Switzerland, Germany, France, and 10 other countries.

---

## RQ1 — Per-Role Results (Test Set, n=240)

| Role | Precision | Recall | F1 | N | Actual € | Predicted € | MAE € |
|---|---|---|---|---|---|---|---|
| Pharmaceutical, Healthcare and Medical Sales | 0.96 | 0.95 | 0.96 | 114 | 52,352 | 52,669 | 7,356 |
| Pharmaceutical Marketing | 0.86 | 0.86 | 0.86 | 28 | 50,825 | 49,593 | 8,030 |
| Regulatory Affairs | 0.88 | 0.88 | 0.88 | 24 | 73,488 | 73,907 | 10,612 |
| Manufacturing & Operations | 0.86 | 0.88 | 0.87 | 41 | 66,969 | 70,138 | 13,392 |
| Clinical Research | 0.82 | 0.85 | 0.84 | 33 | 80,703 | 74,660 | 13,435 |
| **Overall (weighted avg)** | **0.91** | **0.90** | **0.90** | **240** | — | — | **9,627** |

---

## RQ2 — Top 5 Clusters by Size

| Cluster | Jobs | Avg Salary € | Hybrid | Top Location | Top Keywords |
|---|---|---|---|---|---|
| Pharmaceutical Sales Rep | 1,316 | 54,568 | Yes | UK | sales, medical, manager, territory |
| QA Manufacturing Engineer | 1,057 | 78,738 | Yes | Europe (General) | quality, engineer, assurance |
| Pharmaceutical Sales Rep — Representative | 867 | 63,926 | Yes | UK | representative, sales, medical sales |
| Product Marketing Manager | 831 | 78,189 | Yes | UK | marketing, manager, product |
| Medical Affairs Manager | 826 | 96,547 | Yes | UK | medical, pharmacovigilance, medical affairs |

Full 30-cluster breakdown is in `RQ2_Hybrid_Roles.ipynb`.

---

## RQ3 — Top 10 SHAP Features

| Feature | Type | Mean SHAP Value (€) |
|---|---|---|
| Category | Structural | 9,184 |
| Location | Structural | 5,875 |
| Job Type | Structural | 5,184 |
| Leadership | Soft/Management Skill | 4,793 |
| Clinical Trials | Technical Skill | 1,384 |
| Compliance | Technical Skill | 1,062 |
| Medical Writing | Soft/Management Skill | 907 |
| Regulatory Affairs | Technical Skill | 891 |
| Validation | Technical Skill | 763 |
| GCP | Technical Skill | 536 |

---

## How to Recreate Findings

### Option 1 — Google Colab (recommended)
1. Open any notebook in the `notebooks/` folder on GitHub
2. Click the **"Open in Colab"** badge at the top
3. Upload the relevant data file to `/content/`:
   - `Data_Inspection.ipynb` / `Data_Cleaning.ipynb` → raw file from `data/raw/`
   - `RQ1`, `RQ2`, `RQ3` notebooks → `Europe_Pharma_Jobs_Cleaned.xlsx` from `data/processed/`
4. Runtime → Run all

### Option 2 — Local environment
```bash
git clone https://github.com/kashbhoir-cmd/pharma-job-market-analysis.git
cd pharma-job-market-analysis
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## Requirements

See `requirements.txt` for the full list. Key libraries:

| Library | Used in | Purpose |
|---|---|---|
| pandas, numpy | All notebooks | Data manipulation |
| sentence-transformers | RQ1, RQ2 | Text embeddings |
| tensorflow | RQ1 | Dual-head neural network |
| scikit-learn | RQ1, RQ2, RQ3 | Clustering, splitting, metrics |
| umap-learn, scipy | RQ2 | Dimensionality reduction, dendrograms |
| xgboost, shap | RQ3 | Salary prediction, explainability |
| matplotlib | RQ2, RQ3 | Charts |

---

## Citation

If you use this work, please cite:
```
Kashmira Bhoir (2026). AI Approaches to Analysing Recruitment Demand:
Machine Learning Insights from European Pharmaceutical Job Postings.
MSc Dissertation, GISMA University of Applied Sciences.
GitHub: https://github.com/kashbhoir-cmd/pharma-job-market-analysis
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
