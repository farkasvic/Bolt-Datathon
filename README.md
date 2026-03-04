# Vancouver City FC Revenue Insight

**A data science case study from the Bolt Datathon** — analyzing fan engagement, merchandise sales, and stadium operations to segment customers and build a predictive clustering model for revenue optimization.

---

## Overview

Vancouver City FC has been facing declining revenue across three pillars: **stadium operations**, **fanbase engagement**, and **merchandise sales**, alongside challenges engaging international audiences. This project applies exploratory data analysis, customer segmentation, and machine learning to derive actionable insights and a deployable fan-segment prediction model.

| | |
|---|---|
| **Project** | Vancouver City FC Revenue Insight |
| **Source** | Bolt Case Package (UBC First Byte) |
| **Date Range** | Dec 2023 – Dec 2024 |
| **Tools** | Python, pandas, scikit-learn, Altair, Seaborn |

---

## Key Deliverables

- **4 distinct fan segments** with interpretable profiles and business implications
- **Cluster prediction model** (Decision Tree / Random Forest) to classify new fans by demographics and behavior
- **Lifetime Value (LTV) analysis** combining ticket revenue and merchandise spend
- **Data documentation** and reproducible cleaning pipeline

---

## Fan Segments

| Segment | Size | Avg LTV | Avg Games | Avg Merch | Season Pass |
|---------|------|---------|-----------|-----------|-------------|
| **Merch Enthusiasts** | 33,009 | $366 | 4.5 | $123 | 0% |
| **Disengaged Fans** | 23,791 | $245 | 4.5 | $0 | 0% |
| **Superfans / VIPs** | 8,451 | $707 | 4.5 | $462 | 0% |
| **Loyal Attendees** | 4,749 | $1,343 | 22.5 | $124 | 100% |

---

## Project Structure

```
Bolt-Datathon/
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── BOLT UBC First Byte - Stadium Operations.csv
│   │   ├── BOLT UBC First Byte - Fanbase Engagement.csv
│   │   └── BOLT UBC First Byte - Merchandise Sales.csv
│   └── processed/              # Cleaned & merged outputs
│       ├── clean_stadium.csv
│       ├── fanbase_clean.csv
│       ├── cleaned_merch.csv
│       └── merch_fanbase_merged.csv
├── model/                      # Trained models & artifacts
│   ├── cluster_prediction_model_v2.pkl
│   ├── feature_columns.pkl
│   └── ...
├── img/                        # Visualizations
├── Cleaning.ipynb              # Data cleaning pipeline
├── EDA.ipynb                   # Exploratory data analysis
├── Analysis.ipynb              # LTV, clustering, prediction model
├── Clean_pres.ipynb           # Presentation / summary
├── Data_Bible.ipynb           # Data dictionary & documentation
├── requirements.txt
└── README.md
```

---

## Methodology

1. **Data cleaning** — Standardized formats, handled missing values (MAR for size/arrival date), and merged fanbase with merchandise data.
2. **Exploratory analysis** — Distribution of revenue sources, fan demographics, and purchase behavior.
3. **LTV calculation** — Combined ticket revenue (estimated from stadium data) and merchandise spend per member.
4. **Clustering** — K-means on scaled features (games attended, merch spend, seasonal pass, purchase frequency); validated with silhouette score and elbow method.
5. **Prediction model** — Decision Tree / Random Forest to predict cluster from age, region, seasonal pass, games attended, and merchandise LTV.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter (for notebooks)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/Bolt-Datathon.git
cd Bolt-Datathon
pip install -r requirements.txt
```

### Run the pipeline

1. **Cleaning** — Run `Cleaning.ipynb` to produce processed CSVs (or use existing `data/processed/`).
2. **EDA** — Run `EDA.ipynb` for exploratory visualizations.
3. **Analysis** — Run `Analysis.ipynb` to compute LTV, fit clusters, and train the prediction model.

### Use the prediction model

```python
import joblib
import pandas as pd

model = joblib.load('model/cluster_prediction_model_v2.pkl')
feature_columns = joblib.load('model/feature_columns.pkl')

# One-hot encode input (age, region, seasonal_pass, games_attended, merch_ltv)
# and predict cluster
```

---

## Data Sources

| Dataset | Records | Description |
|---------|---------|-------------|
| Fanbase Engagement | ~70,000 | Member profiles, age, region, seasonal pass, games attended |
| Merchandise Sales | ~68,000 | Purchase history, product categories, channels, promotions |
| Stadium Operations | 144 | Monthly revenue by source (tickets, food, etc.) |

---

## License

See [LICENSE](LICENSE) for details.
