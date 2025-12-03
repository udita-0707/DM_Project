# Scientific Research Trends Analysis

A comprehensive data mining project analyzing **2.2+ million scientific papers** from arXiv to uncover patterns in scientific research, collaboration, and impact.

## 📊 Project Overview

This project combines arXiv publication data with real citation metrics from Semantic Scholar API to provide insights into:
- Research area growth and breakthrough detection
- International collaboration patterns
- Interdisciplinary research impact
- Citation half-life analysis
- Emerging keywords and research frontiers
- Predictive modeling for citation counts

**📖 For detailed information about achievements, preprocessing techniques, and results, see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd DM_Project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Process Data

```bash
# Process full dataset (2.2M+ papers)
python3 src/data_acquisition/arxiv_dataset.py --full

# Or process sample for testing
python3 src/data_acquisition/arxiv_dataset.py --sample
```

### 3. Run Analysis

```bash
# Run comprehensive analysis
python3 src/analysis/comprehensive_analysis.py

# Run predictive models
python3 src/analysis/predictive_models.py
```

### 4. Start Dashboard

```bash
# Using helper script (recommended)
./run.sh dashboard

# Or manually
PYTHONPATH=$PWD gunicorn -w 4 -b 0.0.0.0:8080 "src.dashboard.app:app"
```

Access dashboard at: **http://localhost:8080**

## 📁 Project Structure

```
DM_Project/
├── data/
│   ├── raw/                    # Original arXiv JSON files
│   ├── processed/              # Cleaned CSV files
│   │   ├── arxiv_processed.csv # Main processed dataset (2.2M+ papers)
│   │   ├── enriched/           # Datasets with citation data
│   │   └── analysis_results/   # All analysis outputs and visualizations
├── src/
│   ├── data_acquisition/       # Data processing and enrichment scripts
│   │   ├── arxiv_dataset.py   # Main preprocessing script
│   │   ├── enrich_with_citations.py  # Citation enrichment
│   │   └── semanticscholar_api.py   # API client
│   ├── analysis/               # Analysis and visualization scripts
│   │   ├── comprehensive_analysis.py  # 6 core questions
│   │   ├── predictive_models.py       # ML models
│   │   └── batch_full_dataset_analysis.py  # Full dataset analysis
│   ├── dashboard/              # Flask web application
│   │   ├── app.py              # Main Flask app
│   │   └── templates/          # HTML templates
│   └── utils/                  # Utility functions
├── venv/                       # Python virtual environment
├── requirements.txt            # Python dependencies
├── run.sh                      # Helper script for common tasks
├── README.md                   # This file
└── PROJECT_SUMMARY.md          # Detailed project documentation
```

## 🛠️ Technologies

- **Python 3.8+** - Core programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib/seaborn** - Data visualization
- **scikit-learn** - Machine learning models
- **Flask** - Web dashboard framework
- **gunicorn** - Production web server
- **requests** - HTTP API calls
- **wordcloud** - Interactive word cloud generation

## 📈 Key Results

- **Total Papers Analyzed**: 2,884,305 (full dataset: 1986-2025)
- **Enriched Sample**: 10,000 papers with real citations (99.3% success rate)
- **Research Categories**: 38 categories analyzed
- **Core Questions**: 6/6 answered
- **ML Model Performance**: R² = 0.83 (training), RMSE = 94.34 (test)

## 🎯 Main Features

### 1. Comprehensive Analysis
- Research area growth trends and breakthrough detection
- International collaboration analysis
- Interdisciplinary citation impact
- Citation half-life analysis
- Emerging keywords detection
- Predictive modeling

### 2. Interactive Dashboard
- **Main Dashboard**: Interactive word cloud generator with keyword search
- **Results Page**: Comprehensive HTML page with all analyses and visualizations

### 3. Full Dataset Support
- Batch processing for 2.2M+ papers
- Memory-efficient chunked processing
- Parallel processing options

## 📝 Usage Examples

### Process Full Dataset
```bash
python3 src/data_acquisition/arxiv_dataset.py --full
```

### Enrich with Citations (requires API key)
```bash
export SEMANTIC_SCHOLAR_API_KEY='your_api_key'
python3 src/data_acquisition/enrich_with_citations.py --sample-size 10000 --strategy random
```

### Run All Analyses
```bash
./run.sh analyze
```

### Analyze Full Dataset
```bash
python3 src/analysis/batch_full_dataset_analysis.py --chunk-size 100000 --parallel
```

## 📚 Documentation

- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete project documentation including:
  - Detailed achievements
  - Preprocessing techniques explained
  - Results summary
  - Technical details

## ⚠️ Requirements

- Python 3.8 or higher
- 8GB+ RAM (for full dataset processing)
- Internet connection (for API enrichment)
- Semantic Scholar API key (optional, for citation enrichment)

## 🔧 Helper Scripts

The `run.sh` script provides convenient commands:

```bash
./run.sh help          # Show all available commands
./run.sh process-data  # Process arXiv dataset
./run.sh analyze       # Run all analyses
./run.sh dashboard     # Start web dashboard
./run.sh full          # Complete pipeline
```

## 📄 License

Educational project for Data Mining course.

---

**For detailed information about preprocessing techniques, achievements, and results, please refer to [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**
