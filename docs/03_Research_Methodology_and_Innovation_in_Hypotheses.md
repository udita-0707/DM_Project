# Research Methodology and Innovation in Hypotheses

This document details our technical approach and innovative hypotheses, addressing **Research Methodology** (20 Marks) and **Innovation in Hypotheses** (20 Marks).

## 1. Research Methodology

Our methodology is grounded in a robust data science pipeline using Python's scientific stack (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`).

### 1.1. Data Pre-processing Pipeline

| Step | Technique | Justification |
| --- | --- | --- |
| Data Loading | Iterative JSON parsing | Handles multi-gigabyte files without memory issues |
| Temporal Extraction | Regex parsing of `versions` field | Extracts submission dates for time-series analysis |
| Categorization | Splitting `categories` field | Standardizes arXiv codes into main disciplines |
| Collaboration Proxy | Author list parsing | Calculates co-author count for collaboration intensity |

### 1.2. Analysis Techniques

- **Growth Analysis:** Rolling averages on publication counts to identify trends and bursts
- **Collaboration Analysis:** Co-occurrence matrix of (Field × Country) visualized as heatmap
- **Citation Half-Life:** Exponential decay model on citation patterns over time (requires Semantic Scholar API)
- **Emerging Keywords:** TF-IDF to identify keywords uniquely frequent in recent time windows

### 1.3. Visualization Strategy

- **Static:** `matplotlib`/`seaborn` charts (line charts, heatmaps, box plots) saved to `data/processed/analysis_results/`
- **Interactive:** Flask-based word cloud for dynamic keyword exploration

### 1.4. Predictive Model

- **Goal:** Predict citation count based on paper features
- **Model:** Linear Regression or Random Forest Regressor
- **Features:** Submission year, author count, discipline count, TF-IDF keywords

## 2. Innovative Hypotheses

We will test two novel hypotheses:

### Hypothesis 1: "Interdisciplinary Premium"

> Papers classified under **three or more distinct disciplines** will exhibit a **significantly longer citation half-life** than single-discipline papers, even with lower initial citation rates.

**Rationale:** Interdisciplinary work may start slow but has lasting impact by bridging communities and becoming foundational across fields.

### Hypothesis 2: "Collaboration-Burst Lag"

> Bursts in **international collaboration** (measured by countries per paper) in a field will **lag** publication volume bursts by **2-3 years**.

**Rationale:** Initial breakthroughs stem from smaller local teams; international collaboration follows once the field matures, providing predictive insight into field maturity.

