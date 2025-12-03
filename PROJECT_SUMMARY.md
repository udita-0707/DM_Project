# Data Mining Project: Scientific Research Trends Analysis

## Project Overview

This project analyzes **2.2+ million scientific papers** from arXiv to uncover patterns in scientific research, collaboration, and impact. We combine arXiv publication data with real citation metrics from Semantic Scholar API to provide comprehensive insights into research trends.

## What We Have Achieved

### 1. **Complete Dataset Processing**
- ✅ Processed **2,884,305 papers** from arXiv (1986-2025, 39 years)
- ✅ Analyzed **38 research categories** across multiple disciplines
- ✅ Enriched **10,000 papers** with real citation data from Semantic Scholar API (99.3% success rate)

### 2. **Comprehensive Analysis (6 Core Questions)**
1. **Research Area Growth**: Identified fastest-growing fields and breakthrough events
   - Analyzes publication trends across research categories over time
   - Calculates year-over-year growth rates for each category
   - Detects breakthrough events (sudden growth spikes >50% year-over-year)
   - Visualizes top 10 categories with growth trends and breakthrough markers
   - Generates: `research_growth_with_breakthroughs.png`, `research_growth_data.csv`, `breakthrough_events.csv`

2. **International Collaboration**: Analyzed geographic distribution of research
   - Simulates country assignment based on submission patterns (20 countries)
   - Creates country-category heatmaps showing dominance patterns
   - Two visualizations: percentage distribution and absolute counts
   - Generates: `international_collaboration_heatmap.png`, `country_category_percentage.csv`, `country_category_counts.csv`

3. **Interdisciplinary Impact**: Compared citation patterns between single vs multi-discipline papers
   - Classifies papers as single-discipline (1 category) vs interdisciplinary (2+ categories)
   - Uses real citation data when available, citation proxy otherwise
   - Performs Mann-Whitney U test for statistical significance
   - Four-panel visualization: box plots, histograms, category count analysis, summary statistics
   - Generates: `interdisciplinary_citation_analysis.png`, `interdisciplinary_analysis_results.json`

4. **Citation Half-Life**: Analyzed how citation patterns change over time
   - Calculates citation metrics by paper age (years since submission)
   - Estimates peak citation age and half-life (age where citations drop to half of peak)
   - Analyzes trends by publication year (last 15 years)
   - Two-panel visualization: citation by age with half-life markers, and trend analysis
   - Generates: `citation_half_life_analysis.png`, `citation_by_age.csv`, `half_life_trends.csv`

5. **Emerging Keywords**: Detected new research frontiers
   - Extracts keywords from titles and abstracts (4+ character words, filtered stop words)
   - Compares keyword frequencies: historical (pre-2020) vs recent (2020+)
   - Calculates growth rates and frequency increases
   - Identifies top 30 emerging keywords by frequency increase
   - Two-panel visualization: top keywords bar chart, frequency scatter plot
   - Generates: `emerging_keywords_analysis.png`, `emerging_keywords.csv`

6. **Predictive Modeling**: Built ML models to predict citation counts
   - Random Forest Regressor with 100 estimators, max_depth=10
   - Features: submission_year, num_authors, num_categories, title_length, abstract_length, paper_age, top 10 category one-hot encodings
   - Temporal train/test split (80/20 by year)
   - Four-panel visualization: feature importance, training predictions, test predictions, residuals
   - Generates: `predictive_model_analysis.png`, `predictive_model_results.json`, `feature_importance.csv`

### 3. **Full Dataset Analysis**
- ✅ Batch processing system for analyzing 2.2M+ papers efficiently
- ✅ Memory-efficient chunked processing (100K rows per chunk)
- ✅ Generated comprehensive statistics and visualizations from full dataset

### 4. **Interactive Dashboard**
- ✅ Flask web application with interactive word cloud generator
- ✅ Comprehensive results page (`/results`) showcasing all analyses and visualizations
- ✅ On-demand word cloud generation with keyword search and year filtering
- ✅ Statistics panel showing paper counts, top categories, and year ranges
- ✅ Professional, responsive design with smooth navigation

## Preprocessing Techniques Used

### 1. **Data Cleaning & Extraction**

**Why:** Raw arXiv data comes as a large JSON file (3.5GB) with nested structures. We need to extract relevant fields and convert to a structured format.

**How:**
- **Chunked Processing**: Process data in chunks of 50,000 records to manage memory efficiently
- **Field Selection**: Extract only relevant columns (id, title, abstract, authors, categories, dates, etc.)
- **Error Handling**: Skip malformed JSON lines and records with missing critical fields
- **Output Format**: Convert to CSV for easier analysis with pandas

**Code Location:** `src/data_acquisition/arxiv_dataset.py`

### 2. **Date Parsing & Year Extraction**

**Why:** Submission dates are stored as strings in various formats. We need consistent year values for time-based analysis.

**How:**
- Parse date strings from version timestamps (format: 'Mon, 20 Oct 2008 11:36:39 GMT')
- Extract submission year from first version (earliest submission date)
- Handle multiple date formats with fallback parsing
- Create `submission_year` column for temporal analysis

**Code Location:** `src/data_acquisition/arxiv_dataset.py` → `get_submission_year()`

### 3. **Category Simplification**

**Why:** arXiv categories are hierarchical (e.g., "cs.AI", "math.CO"). We need simplified category names for analysis.

**How:**
- Split category strings by spaces to get individual categories
- Extract main category prefix (before the dot) - e.g., "cs.AI" → "cs"
- Convert to list format: ["cs", "math", "physics"]
- Create `main_categories` column as a list of simplified categories

**Code Location:** `src/data_acquisition/arxiv_dataset.py` → line 117-119

### 4. **Citation Data Enrichment**

**Why:** arXiv doesn't provide citation counts. We need citation data to analyze research impact.

**How:**
- **API Integration**: Use Semantic Scholar API to fetch real citation counts
- **Rate Limiting**: Respect API limits (1 request/second) to avoid blocking
- **Caching**: Cache API responses in `data/cache/semanticscholar_cache.json` to avoid redundant requests
- **Batch Processing**: Use batch API endpoint (up to 500 papers per request) for efficiency
- **ArXiv ID Extraction**: Clean arXiv IDs from various formats ("arXiv:2101.12345v1" → "2101.12345")
- **Resume Capability**: Can resume interrupted enrichment runs
- **Progress Tracking**: Shows progress with success rate and remaining papers
- **Fallback Proxy**: Create citation proxy when real data unavailable (uses paper age + version count)

**Code Location:** 
- `src/data_acquisition/semanticscholar_api.py` - API client with caching
- `src/data_acquisition/enrich_with_citations.py` - Enrichment script with sampling strategies
- `src/analysis/statistical_tests.py` → `create_citation_proxy()` - Proxy creation
- `src/utils/data_loader.py` → `load_processed_data()` - Automatic enriched data detection

### 5. **Citation Proxy Creation**

**Why:** Not all papers can be enriched with real citations (API limits, papers not in Semantic Scholar). We need a proxy metric.

**How:**
- Calculate `paper_age` = current_year - submission_year (older papers likely have more citations)
- Count `version_count` = number of paper versions (more versions = more interest)
- Normalize both metrics to 0-1 scale
- Combine: `citation_proxy = (paper_age_normalized * 0.6) + (version_count_normalized * 0.4)`
- Scale to 0-100 range
- Use real citations when available, proxy otherwise

**Code Location:** `src/analysis/statistical_tests.py` → `create_citation_proxy()`

### 6. **Feature Engineering for ML Models**

**Why:** Raw data needs transformation into numerical features that ML models can use.

**How:**
- **Author Count**: Parse author list (handles both JSON list format and comma-separated strings) and count number of authors
- **Discipline Count**: Count number of categories per paper (interdisciplinarity measure)
- **Text Length**: Calculate title_length and abstract_length (character counts using `.str.len()`)
- **Paper Age**: Calculate years since submission (current_year - submission_year)
- **Version Count**: Count number of paper versions (indicator of interest/updates)
- **One-Hot Encoding**: Convert top 10 categories to binary features (category_cs, category_math, etc.)
- **Missing Value Handling**: Remove rows with missing critical features
- **Feature Scaling**: StandardScaler for Linear Regression (Random Forest doesn't need scaling)

**Code Location:** 
- `src/analysis/predictive_models.py` → `prepare_features()`
- `src/analysis/comprehensive_analysis.py` → `build_predictive_models()` (similar feature engineering)

### 7. **Data Sampling Strategies**

**Why:** Processing 2.2M papers with API calls would take ~25 days. We need strategic sampling.

**How:**
- **Random Sampling**: Simple random selection for quick tests (default: 10K papers)
- **Stratified Sampling**: Balance across years and categories for representative sample
- **Year-Based Sampling**: Focus on specific time periods (e.g., 2020-2024)
- **Category-Based Sampling**: Focus on top N categories
- **All Data**: Option to process all papers (for smaller datasets or testing)

**Code Location:** `src/data_acquisition/enrich_with_citations.py` → `sample_papers_strategically()`

**Usage Example:**
```bash
# Random sample of 10K papers
python3 enrich_with_citations.py --sample-size 10000 --strategy random

# Papers from specific years
python3 enrich_with_citations.py --years 2020 2021 2022 --strategy year

# Top 5 categories
python3 enrich_with_citations.py --top-categories 5 --strategy category
```

### 8. **Memory-Efficient Batch Processing**

**Why:** Full dataset (2.2M papers) is too large to load into memory at once.

**How:**
- **Chunked Reading**: Use pandas `read_csv()` with `chunksize` parameter (default: 100K rows)
- **Streaming Processing**: Process each chunk independently (category counts, year distribution, collaboration stats)
- **Result Aggregation**: Merge results from all chunks at the end (dictionary aggregation)
- **Parallel Processing**: Optional parallel processing for independent chunks using multiprocessing
- **Output Files**: Timestamped output files for full dataset statistics

**Code Location:** `src/analysis/batch_full_dataset_analysis.py`

**Usage:**
```bash
python3 batch_full_dataset_analysis.py --chunk-size 100000
python3 batch_full_dataset_analysis.py --full --parallel
```

### 9. **Data Type Conversion**

**Why:** Some fields are stored as strings but need to be parsed as lists or numbers.

**How:**
- **List Parsing**: Use `ast.literal_eval()` to convert string representations of lists
- **Author Parsing**: Handle both JSON list format and comma-separated strings
- **Category Parsing**: Convert category strings to lists
- **Type Checking**: Verify data types before processing

**Code Location:** Multiple files - used throughout analysis scripts

### 10. **Missing Value Handling**

**Why:** Real-world data has missing values that can break analysis.

**How:**
- **Selective Dropping**: Remove rows only when critical features are missing
- **Fill Strategies**: Use appropriate defaults (0 for counts, empty list for categories)
- **Citation Handling**: Use proxy when real citations unavailable
- **Report Missing Data**: Track and report missing value statistics

**Code Location:** Used throughout analysis scripts (comprehensive_analysis.py, predictive_models.py, etc.)

## Technical Stack

- **Python 3.8+**: Core programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **scikit-learn**: Machine learning models (RandomForestRegressor, LinearRegression)
- **scipy**: Statistical tests (Mann-Whitney U, ANOVA, correlation tests)
- **Flask**: Web dashboard framework
- **gunicorn**: Production web server (4 workers)
- **requests**: HTTP API calls (Semantic Scholar API)
- **wordcloud**: Interactive word cloud generation
- **json**: Data serialization (API responses, cache)

## Project Structure

```
DM_Project/
├── data/
│   ├── raw/                    # Original arXiv JSON files
│   │   └── arxiv_sample_10k.json  # Sample file for word cloud
│   ├── processed/              # Cleaned CSV files
│   │   ├── arxiv_processed.csv # Main processed dataset (2.2M+ papers)
│   │   ├── enriched/           # Datasets with citation data
│   │   │   └── arxiv_enriched_*.csv  # Enriched datasets with citations
│   │   └── analysis_results/   # All analysis outputs
│   │       ├── comprehensive_analysis/  # 6 core questions results
│   │       ├── predictive_models/       # ML model results
│   │       ├── exploratory_analysis/    # Initial exploration
│   │       ├── full_dataset_analysis/    # Full dataset statistics
│   │       └── statistical_tests/       # Statistical test results
│   └── cache/                  # API response cache
│       └── semanticscholar_cache.json
├── src/
│   ├── data_acquisition/       # Data processing scripts
│   │   ├── arxiv_dataset.py   # Main preprocessing (chunked processing)
│   │   ├── enrich_with_citations.py  # Citation enrichment with sampling
│   │   ├── semanticscholar_api.py    # Semantic Scholar API client
│   │   ├── create_sample_data.py    # Sample data creation
│   │   └── optimized_batch_enrichment.py  # Batch enrichment utilities
│   ├── analysis/               # Analysis and visualization scripts
│   │   ├── comprehensive_analysis.py  # Main analysis (6 questions)
│   │   ├── predictive_models.py       # ML models (citation prediction, forecasting)
│   │   ├── statistical_tests.py       # Statistical hypothesis testing
│   │   ├── batch_full_dataset_analysis.py  # Full dataset batch processing
│   │   ├── exploratory_analysis.py    # Initial data exploration
│   │   └── full_dataset_visualizations.py  # Full dataset visualizations
│   ├── dashboard/              # Flask web application
│   │   ├── app.py              # Main Flask app (word cloud + results page)
│   │   └── templates/
│   │       └── results.html    # Comprehensive results page
│   └── utils/                  # Utility functions
│       └── data_loader.py      # Data loading with enriched data detection
├── venv/                       # Python virtual environment
├── requirements.txt            # Python dependencies
├── run.sh                      # Helper script for common tasks
├── run_full_dataset_analysis.sh  # Full dataset analysis script
├── README.md                   # Project setup and usage guide
└── PROJECT_SUMMARY.md          # This file
```

## Key Results Summary

- **Total Papers Analyzed**: 2,884,305 (full dataset: 1986-2025, 39 years)
- **Enriched Sample**: 10,000 papers with real citations (99.3% success rate)
- **Research Categories**: 38 categories analyzed
- **Core Questions Answered**: 6/6 comprehensive analyses
- **ML Model Performance**: 
  - Random Forest: Training R² = 0.83, Test RMSE = 94.34
  - Linear Regression: Also trained for comparison
- **Additional Analyses**:
  - Research growth forecasting (2026-2028)
  - Emerging keyword classification
  - Statistical hypothesis testing (interdisciplinarity, correlations)

## Main Analysis Scripts

### Comprehensive Analysis
```bash
python3 src/analysis/comprehensive_analysis.py
```
Runs all 6 core questions and generates:
- Research growth with breakthrough detection
- International collaboration heatmaps
- Interdisciplinary citation analysis
- Citation half-life analysis
- Emerging keywords detection
- Predictive modeling for citations

### Predictive Models
```bash
python3 src/analysis/predictive_models.py
```
Trains ML models and generates:
- Citation prediction models (Random Forest, Linear Regression)
- Research growth forecasting (2026-2028)
- Emerging keyword classification

### Full Dataset Analysis
```bash
python3 src/analysis/batch_full_dataset_analysis.py --chunk-size 100000
```
Processes full 2.2M+ dataset in chunks and generates:
- Category distribution statistics
- Year distribution
- Collaboration statistics
- Top keywords

### Dashboard
```bash
./run.sh dashboard
# Or: gunicorn -w 4 -b 0.0.0.0:8080 "src.dashboard.app:app"
```
Access at `http://localhost:8080`:
- `/` - Redirects to `/results`
- `/results` - Comprehensive results page with all visualizations
- `/wordcloud` - Interactive word cloud generator (POST endpoint)
  - Accepts: keyword, year_min, year_max
  - Returns: Base64-encoded word cloud image, statistics, top categories
  - Loads data on-demand from `data/raw/arxiv_sample_10k.json` (10K papers)
  - Generates enhanced visualization with word cloud, top 20 keywords bar chart, and statistics panel

## Future Enhancements

- Expand citation enrichment to larger samples (50K-100K papers)
- Add more sophisticated ML models (neural networks, ensemble methods)
- Implement real-time dashboard updates
- Add more interactive visualizations (D3.js, Plotly)
- Extend analysis to other data sources (PubMed, Google Scholar)
- Add author network analysis
- Implement topic modeling (LDA, BERTopic)

---

*This project demonstrates comprehensive data mining techniques including data preprocessing, feature engineering, statistical analysis, machine learning, and interactive visualization.*
