# Scientific Research Trends Analysis

We are analyzing large-scale scientific publication data from **arXiv** and **Semantic Scholar** to uncover patterns of scientific progress, collaboration, and influence. Our project addresses exploratory questions related to research growth, international collaboration, interdisciplinarity, and emerging keywords.

## Technologies

- **Python 3.8+**, **pandas**, **matplotlib/seaborn**
- **Flask** - Web dashboard
- **wordcloud** - Interactive visualizations
- **scikit-learn** - ML and statistical analysis
- **requests** - API integration

## Project Structure

| Directory | Purpose |
| :--- | :--- |
| `data/raw` | Original datasets (e.g., `arxiv-metadata-oai-snapshot.json`) |
| `data/processed` | Cleaned data and analysis results |
| `src/data_acquisition` | Data downloading, cleaning, and processing scripts |
| `src/analysis` | Exploratory analysis and static visualizations |
| `src/dashboard` | Flask application with interactive word cloud |
| `src/utils` | Utility functions and helpers |
| `docs` | Documentation and evaluation materials |

## Team Roles

### Data Pre-processing Lead (3-4 hours)
- Download and process arXiv dataset
- Create sample extraction (5K-10K records)
- Document data quality

### Research & Analysis Lead (5-6 hours)
- Implement exploratory analysis and test hypotheses
- Add statistical tests (t-test, ANOVA)
- Generate static visualizations

### Visualization Lead (4-5 hours)
- Develop Flask dashboard and interactive word cloud
- Integrate analysis results
- Ensure responsive UI/UX

## Quick Setup (15 Minutes)

```bash
# Clone and install
git clone https://github.com/Kkt04/DM_Project.git
cd DM_Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test installation
python3 -c "import pandas, flask, matplotlib; print('✓ Dependencies installed successfully!')"
```

## Data Acquisition

### Dataset Options

| Type | Records | Size | Use Case |
|------|---------|------|----------|
| Mock (provided) | 7 | 2 KB | Initial testing |
| Sample (create) | 10K | ~50 MB | Development |
| Full (download) | 2.2M+ | 3.5 GB | Final run |

### Steps

1. Download `arxiv-metadata-oai-snapshot.json` from [Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Place in `data/raw/`
3. Extract sample:
```bash
python3 src/data_acquisition/create_sample_data.py \
    --input data/raw/arxiv-metadata-oai-snapshot.json \
    --output data/raw/arxiv_sample_10k.json \
    --size 10000
```
4. Process data:
```bash
python3 src/data_acquisition/arxiv_dataset.py
```

## Running Analysis and Dashboard

### Using Helper Script (Recommended)

```bash
./run.sh process-data  # Process data
./run.sh analyze       # Run analysis
./run.sh dashboard     # Start web server (http://localhost:8080)
./run.sh full          # Complete pipeline
./run.sh help          # View all commands
```

### Manual Execution

```bash
# Run analysis
python3 src/analysis/exploratory_analysis.py

# Start dashboard
PYTHONPATH=$PWD gunicorn -w 4 -b 0.0.0.0:8080 "src.dashboard.app:app"
# Access at http://localhost:8080
```

## Exploratory Questions

1. ✅ Research Area Growth - `analyze_research_growth()`
2. ✅ International Collaboration - `analyze_collaboration()`
3. ✅ Interdisciplinarity vs Citations - `analyze_interdisciplinarity()`
4. ⚠️ Citation Half-Life - Requires Semantic Scholar API
5. ✅ Emerging Keywords - `analyze_emerging_keywords()`
6. ✅ Interactive Word Cloud - Flask dashboard

## Known Limitations

1. **Citation Data**: ArXiv lacks citation counts. Use Semantic Scholar API (rate-limited) or version count as proxy
2. **Country Data**: No structured location data. Focus on co-authorship analysis
3. **Large Dataset**: 3.5 GB dataset. Develop on 10K sample, final run on full dataset

## Git Workflow

We commit frequently with clear messages:
- `FEAT:` - New features
- `FIX:` - Bug fixes  
- `DATA:` - Data processing
- `DOCS:` - Documentation
- `VIZ:` - Visualizations

Work on feature branches and merge to `main` after review.

## Evaluation Criteria

| Criterion | Marks | Documentation |
|-----------|-------|---------------|
| Work Planning & Division | 15 | `docs/01_Work_Planning_and_Division.md` |
| Problem Understanding | 20 | `docs/02_Problem_Understanding_and_Identification.md` |
| Data Pre-processing | 10 | `src/data_acquisition/arxiv_dataset.py` |
| Innovation in Hypotheses | 20 | `docs/03_Research_Methodology...md` |
| Research Methodology | 20 | `docs/03_Research_Methodology...md` |
| Consistency (Git) | 15 | Git history |

## License

Educational project for Data Mining course.

