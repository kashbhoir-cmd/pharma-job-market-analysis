# AI Approaches to Analysing Recruitment Demand: Machine learning insights from European Pharmaceutical Job Postings


**Author:** Kashmira Bhoir 
**Institution:** GISMA University of Applied Sciences 
**Program:** MSc Data Science, Artificial Intelligence and Digital Business 
**Year:** 2025-2026

---

## Overview

This repository holds the entire technical implementations of my 
MSc Data Science, Artificial Intelligence and Digital Business dissertation analysing the European pharmaceutical job postings 
using NLP(Natural Language Processing), ML (Machine Learning). The data was colledt by Kaggle dataset and further added more data through web scraping.

The study is mainly focusing three research gap questions using 8,826 European 
pharmaceutical job postings collected from 10 job boards for (2018 - 2026).

---

## Repository Structure
```
pharma-job-market-analysis/
│
├── notebooks/
│   ├── Web_Scraper.ipynb         # Data collected from 10 European job portals
│   ├── Data_Inspection.ipynb      # Dataset loaded and inspected
│   ├── Data_Cleaning.ipynb        # Data cleaned (missing values/ duplicate values removed
│   ├── RQ1_Salary_Prediction+0.1.ipynb # Multi-task transformer (accurately predict job roles and associated salary levels from posting)
│   ├── RQ2_Hybrid_Roles_0.1.ipynb     # Hierarchical clustering (Discovering Hybrid Role)
│   └── RQ3_Mismatch_Detection_0.1.ipynb # With the help of (XGBoost + SHAP) detecting the mismatch
│
├── data/
│   ├── raw/                             # Original 9 CSV files
│   └── processed/                       # Cleaned dataset
│
├── results/
│   └── figures/                         # All output charts (8 PNG files)
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

---

## Research Questions

### RQ1 — Role-Salary Patterns
> To what extent can multi-task transformers accurately predict 
> pharmaceutical job roles and associated salary levels from posting text?

**Method:** sentence-transformers (all-MiniLM-L6-v2) → Keras dual-head model  
**Dataset:** 1,090 joint rows (salary + role label available)  
**Result:** 80.3% role classification accuracy, overall salary MAE €8,224  
**Gap addressed:** Ather et al. (2024) and Agrawal et al. (2025) focused 
on US/UK tech only — this is the first multi-task transformer for 
European pharma role + salary prediction jointly.

---

### RQ2 — Hybrid Role Discovery
> What hybrid pharmaceutical roles emerge from hierarchical clustering 
> of European job postings, and how do their salary and location profiles differ?

**Method:** UMAP(50) → Agglomerative Clustering (Ward linkage) → TF-IDF  
**Dataset:** Entire 9,420 pharmaceutical job postings  
**Result:** 16 clusters discovered, 14 hybrid (87.5% hybrid rate),
salary range €45,000 – €80,000  
**Gap addressed:** Puente Agueda (2024) used k-means on 4 fixed sectors — 
this discovers 16 data-driven hybrid roles across the full European market.

### RQ3 — Location-Skill Mismatch
> Which location-skill combinations exhibit labour market mismatches 
> detectable via explainable ML?

**Method:** KeyBERT skill extraction → XGBoost → SHAP  
**Dataset:** 1,444 rows with parseable salary (15.3% of dataset)  
**Result:** Location (SHAP €6,847) dominates salary over all skills.
2 mismatches detected — avg gap €18,506.  
**Gap addressed:** Romanko (2023) called for explainability without 
implementing it — this delivers first XGBoost + SHAP pharma analysis 
across European market.

---

## Key Findings

| Finding | Detail |
|---|---|
| Salary Opacity | 84.7% of European Phaarmaceutical salaries were hidden as "Competitive"/ "Negotiable" |
| Total postings | 9,420 jobs from 10 European pharma job boards |
| RQ1 accuracy | 80.3% role classification across 5 Pharmaceutical categories |
| RQ1 salary MAE | €8,224 overall — best role €5,364 (Pharmaceutical, Healthcare and Sales) |
| RQ2 clusters | 16 clusters discovered — 14 hybrid (87.5% hybrid rate) |
| RQ2 salary range | €45,000 – €80,000 across hybrid role clusters |
| RQ2 key hybrid | Clinical Data Analyst — merges Data Management + Clinical Research |
| RQ3 top driver | Location (SHAP €6,847) dominates over all skills combined |
| RQ3 2nd driver | Category (SHAP €5,225) — role type nearly as important as geography |
| RQ3 top skill | GMP (SHAP €460) + Medical Writer (SHAP €493) |
| RQ3 mismatch | 2 mismatches detected — with average gap of €18,506 |
| Mismatch locations | Europe General (gap is €28,374) + France (gap is €8,638) |

---

## Dataset

| Detail | Value |
|---|---|
| Total job postings after Filtering Pharmaceutical industry | 9,420 |
| Sources | 10 European pharma job portals |
| Countries | UK (38.2%), Europe General (34.2%), Germany (10.1%), Switzerland (5.9%), France (5.7%), Austria (1.7%), Netherlands (1.7%), Italy (1.5%), Spain (0.7%) |
| Salary parseable | 1,444 rows (15.3%) |
| Salary hidden | 84.7% — reported as "Competitive" or "Negotiable" |
| Salary mean | €56,620 (standard salary: €28,938) |
| Columns | 13 — category, company_name, job_description, job_title, job_type, location, post_date, salary_offered, salary_num, salary_min, salary_max, job_type_clean, country |

> The raw combined CSV exceeds GitHub's 25MB file limit.  
> A 1,000 rows of sample is provided in `data/raw/`.  
> The 1,000 rows of cleaned dataset is available in `data/processed/`, since the original cleaned file again exceed GitHub's 25 MB file limit.

---

## RQ1 Results — Details Per Role

| Role | Precision | Recall | F1 | N | Actual € | Predicted € | MAE € |
|---|---|---|---|---|---|---|---|
| Pharma Healthcare & Sales | 89% | 95% | 0.92 | 102 | 45,716 | 45,658 | 5,364 |
| Clinical Research | 74% | 76% | 0.75 | 33 | 68,470 | 61,066 | 11,735 |
| Manufacturing & Operations | 79% | 65% | 0.71 | 34 | 59,838 | 62,350 | 9,781 |
| Pharmaceutical Marketing | 73% | 64% | 0.68 | 25 | 45,680 | 43,919 | 5,924 |
| Regulatory Affairs | 60% | 62% | 0.61 | 24 | 65,500 | 53,505 | 15,745 |
| **Overall** | **80%** | **80%** | **0.80** | **218** | — | — | **8,224** |

---

## RQ2 Results — Profiles Clustered

| Cluster | Jobs | Hybrid | Top Keywords |
|---|---|---|---|
| Medical Sales Liaison | 1,861 | Yes | sales, medical, healthcare, territory |
| Medical Affairs Manager | 1,827 | Yes | clinical, research, medical, project |
| QA Manufacturing Engineer | 1,242 | Yes | quality, engineer, assurance, qa |
| Product Marketing Manager | 1,136 | Yes | marketing, manager, market, product |
| Key Account Manager | 972 | No | account, account manager, key account |
| Regulatory Affairs Specialist | 584 | Yes | regulatory, affairs, affairs manager |
| Clinical Data Analyst | 693 | Yes | data, statistical, programmer, clinical |
| Medical Affairs Manager (Europe) | 547 | Yes | writer, medical writer, writing |

---

## RQ3 Results — Importance of SHAP Feature

| Feature | SHAP € | Type |
|---|---|---|
| Location | 6,847 | Structural |
| Category | 5,225 | Structural |
| Job Type | 1,063 | Structural |
| Medical Writer | 493 | Technical Skill |
| Account Director | 476 | Technical Skill |
| GMP | 460 | Technical Skill |
| Medical Communications | 409 | Technical Skill |
| Account Manager | 392 | Soft Skill |

---

## How to Recreate Findings

### Option 1 — Google Colab (recommended)

1. Open any notebook in `notebooks/` folder on GitHub
2. Click **"Open in Colab"** button at top of notebook
3. Upload `Combined_Pharma_Jobs_Cleaned.csv` to `/content/`
4. Run all cells — `Runtime → Run all`
   
### Option 2 — Local environment
```bash
# Clone the repository
git clone https://github.com/[your-username]/pharma-job-market-analysis.git
cd pharma-job-market-analysis

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```
---

## Requirements

See `requirements.txt` for full list. Key libraries:

| Library | Version | Purpose |
|---|---|---|
| tensorflow | ≥2.12 | Keras dual-head model (RQ1) |
| sentence-transformers | ≥2.2 | Text embeddings (RQ1, RQ2) |
| umap-learn | ≥0.5 | Dimensionality reduction (RQ2) |
| scikit-learn | ≥1.2 | Clustering and metrics (RQ2, RQ3) |
| xgboost | ≥1.7 | Salary predictor (RQ3) |
| shap | ≥0.41 | Explainability (RQ3) |
| keybert | ≥0.7 | Skill extraction (RQ3) |
| pandas | ≥1.5 | Data manipulation |
| numpy | ≥1.23 | Numerical computing |
| matplotlib | ≥3.6 | Visualisation |

---

## Citation

If you use this work please cite:
```
Kashmira Bhoir (2025). European Pharmaceutical Job Postings Demand Analysis
. MSc Dissertation, GISMA Univesity of Applied Sciences.
GitHub: https://github.com/kashbhoir-cmd/pharma-job-market-analysis
```

---

## License

This project is licensed under the MIT License.  
See `LICENSE` for details.
```

---

### After pasting

1. Scroll down to **"Commit changes"**
2. Commit message:
```
Add comprehensive README with results and reproduction steps
