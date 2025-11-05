# Data Acquisition Guide - ArXiv Dataset

This guide provides step-by-step instructions for downloading the arXiv dataset and extracting a 10,000-record sample for development.

---

## 📥 Step 1: Download the ArXiv Dataset from Kaggle

### Prerequisites
- Kaggle account (free to create)
- Kaggle API key (optional, but recommended for command-line download)

### Option A: Manual Download (Recommended for Beginners)

1. **Visit the Kaggle Dataset Page:**
   ```
   https://www.kaggle.com/datasets/Cornell-University/arxiv
   ```

2. **Sign in or Create a Kaggle Account**
   - If you don't have an account, click "Register" (it's free)
   - Sign in with your credentials

3. **Download the Dataset:**
   - Click the **"Download"** button (top right of the page)
   - This will download: `arxiv-metadata-oai-snapshot.json` (~3.5 GB)
   - **Note:** The download may take 10-30 minutes depending on your internet speed

4. **Move the File to Your Project:**
   ```bash
   # Navigate to your project directory
   cd /Users/dally/Data-Mining-Project/DM_project_v1
   
   # Move the downloaded file to data/raw/
   # (Replace ~/Downloads with your actual download location)
   mv ~/Downloads/arxiv-metadata-oai-snapshot.json data/raw/
   ```

5. **Verify the File:**
   ```bash
   # Check if the file exists and its size
   ls -lh data/raw/arxiv-metadata-oai-snapshot.json
   
   # Expected output: ~3.5 GB file
   ```

---

### Option B: Command-Line Download Using Kaggle API

**Prerequisites:**
1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. **Set Up Kaggle API Credentials:**
   - Go to: https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`
   - Place it in the correct location:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download the Dataset:**
   ```bash
   cd /Users/dally/Data-Mining-Project/DM_project_v1
   
   # Download using Kaggle CLI
   kaggle datasets download -d Cornell-University/arxiv -p data/raw/
   
   # Unzip the file
   unzip data/raw/arxiv.zip -d data/raw/
   ```

---

## 🔬 Step 2: Extract a 10,000-Record Sample

Once you have the full dataset, extract a sample for faster development and testing.

### Why Use a Sample?
- **Full Dataset:** 2.2M+ records, ~3.5 GB → Takes hours to process
- **10K Sample:** 10,000 records, ~50 MB → Takes seconds to process
- **Strategy:** Develop with sample, run full dataset only once at the end

---

### Running the Sample Extraction Script

The script `src/data_acquisition/create_sample_data.py` is already created and ready to use.

#### Basic Usage (Default: 10,000 records)

```bash
cd /Users/dally/Data-Mining-Project/DM_project_v1

# Extract 10,000 random records
python3 src/data_acquisition/create_sample_data.py
```

This will:
- Read from: `data/raw/arxiv-metadata-oai-snapshot.json`
- Save to: `data/raw/arxiv_sample_10k.json`
- Extract: 10,000 random records
- Time: ~5-10 minutes (depending on your system)

---

#### Advanced Usage (Custom Parameters)

**Extract a Different Sample Size:**
```bash
# Extract 5,000 records
python3 src/data_acquisition/create_sample_data.py --size 5000

# Extract 50,000 records (for more comprehensive testing)
python3 src/data_acquisition/create_sample_data.py --size 50000
```

**Specify Custom Input/Output Paths:**
```bash
python3 src/data_acquisition/create_sample_data.py \
    --input data/raw/arxiv-metadata-oai-snapshot.json \
    --output data/raw/arxiv_sample_custom.json \
    --size 10000
```

**Change Random Seed (for different samples):**
```bash
# Use seed=42 (default, reproducible)
python3 src/data_acquisition/create_sample_data.py --seed 42

# Use seed=123 (different sample)
python3 src/data_acquisition/create_sample_data.py --seed 123
```

**View All Options:**
```bash
python3 src/data_acquisition/create_sample_data.py --help
```

---

### Expected Output

When you run the script, you'll see output like this:

```
============================================================
ArXiv Sample Data Extraction
============================================================
Input file: data/raw/arxiv-metadata-oai-snapshot.json
Output file: data/raw/arxiv_sample_10k.json
Sample size: 10000

Step 1: Counting total records in dataset...
  Total records found: 2,241,611

Step 2: Generating 10000 random indices...
  Generated 10000 unique indices

Step 3: Extracting sample records...
  Processed 100,000 records... (found 437 samples so far)
  Processed 200,000 records... (found 892 samples so far)
  ...
  Extraction complete: 10000 records extracted

Step 4: Saving sample to data/raw/arxiv_sample_10k.json...
  Sample saved successfully!
  File size: 52.34 MB

============================================================
Summary
============================================================
Total records in full dataset: 2,241,611
Sample size: 10,000
Sampling rate: 0.45%
Output file: data/raw/arxiv_sample_10k.json

Next steps:
1. Process the sample: python3 src/data_acquisition/arxiv_dataset.py
2. Run analysis: python3 src/analysis/exploratory_analysis.py
```

---

## 🔧 Step 3: Process the Sample Data

After extracting the sample, you need to clean and transform it into a structured format.

### Modify the Processing Script

Edit `src/data_acquisition/arxiv_dataset.py` to use your sample file:

```python
# Line 10-11 (approximately)
# Change from:
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')

# To use the sample file, modify the download_and_process_arxiv_data function
# Or simply pass the sample file path when calling the function
```

**Quick Fix:** Create a new function in `arxiv_dataset.py`:

```python
def process_sample_data():
    """Process the sample data instead of the full dataset."""
    raw_file_path = os.path.join(RAW_DATA_DIR, 'arxiv_sample_10k.json')
    
    if not os.path.exists(raw_file_path):
        print(f"Error: Sample file not found at {raw_file_path}")
        print("Please run create_sample_data.py first.")
        return
    
    # Use the same processing logic as download_and_process_arxiv_data
    # but with the sample file...
```

### Run the Processing Script

```bash
cd /Users/dally/Data-Mining-Project/DM_project_v1

# Process the sample data
python3 src/data_acquisition/arxiv_dataset.py
```

This creates: `data/processed/arxiv_processed.csv`

---

## 📊 Step 4: Verify the Processed Data

```bash
# Check the processed file
head -20 data/processed/arxiv_processed.csv

# Check the number of records
wc -l data/processed/arxiv_processed.csv
# Expected: ~10,001 lines (10,000 + 1 header)
```

Or use Python:

```python
import pandas as pd

# Load the processed data
df = pd.read_csv('data/processed/arxiv_processed.csv')

print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few records:")
print(df.head())
print(f"\nSubmission years range: {df['submission_year'].min()} - {df['submission_year'].max()}")
print(f"Unique categories: {df['main_categories'].nunique()}")
```

---

## 🚨 Troubleshooting

### Problem: "ERROR: Input file not found"

**Solution:**
```bash
# Check if the file exists
ls -lh data/raw/

# If not, download it from Kaggle (see Step 1)
```

---

### Problem: "MemoryError" when processing

**Solution:**
- Use a smaller sample size (e.g., 5,000 instead of 10,000)
- Close other applications to free up RAM
- Process the data in chunks (modify the script)

---

### Problem: Script is taking too long (>30 minutes)

**Solution:**
- This is normal for the full dataset (2.2M records)
- For the 10K sample, it should take 5-10 minutes
- Ensure you're using the sample file, not the full file
- Check your system's disk I/O (close unnecessary programs)

---

### Problem: Malformed JSON records

**Solution:**
The script automatically skips malformed records and reports them:
```
WARNING: Skipped malformed record at line 12345
```
This is expected for large datasets. The script will continue and extract the remaining valid records.

---

## 📝 Data Format Reference

### Raw Data Format (JSON Lines)

Each line in `arxiv-metadata-oai-snapshot.json` is a JSON object:

```json
{
  "id": "0704.0001",
  "submitter": "Pavel Nadolsky",
  "authors": "Pavel Nadolsky, C. R. Schmidt",
  "title": "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies",
  "comments": "37 pages with 15 figures",
  "journal-ref": null,
  "doi": null,
  "report-no": null,
  "categories": "hep-ph",
  "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
  "abstract": "A fully differential calculation in perturbative quantum chromodynamics...",
  "versions": [{"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"}],
  "update_date": "2008-11-26"
}
```

### Processed Data Format (CSV)

After processing, the data is structured as:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | ArXiv paper ID |
| `submitter` | string | Submitter name |
| `authors` | string (list) | List of author names |
| `title` | string | Paper title |
| `categories` | string | ArXiv categories (space-separated) |
| `abstract` | string | Paper abstract |
| `versions` | string (list) | Version history |
| `update_date` | string | Last update date |
| `submission_year` | int | **Extracted:** Year of first submission |
| `main_categories` | string (list) | **Extracted:** Main discipline codes |

---

## 🎯 Quick Reference: Full Workflow

```bash
# 1. Download dataset from Kaggle (manual or CLI)
# Visit: https://www.kaggle.com/datasets/Cornell-University/arxiv

# 2. Extract 10K sample
cd /Users/dally/Data-Mining-Project/DM_project_v1
python3 src/data_acquisition/create_sample_data.py --size 10000

# 3. Process the sample
python3 src/data_acquisition/arxiv_dataset.py

# 4. Verify
head data/processed/arxiv_processed.csv

# 5. Run analysis
python3 src/analysis/exploratory_analysis.py

# 6. Launch dashboard
cd src/dashboard
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

---

## 📅 Recommended Sample Sizes for Different Use Cases

| Use Case | Sample Size | Processing Time | File Size |
|----------|-------------|-----------------|-----------|
| **Script Testing** | 100 | <1 min | ~500 KB |
| **Development** | 5,000 | ~3 min | ~25 MB |
| **Full Testing** | 10,000 | ~5 min | ~50 MB |
| **Comprehensive Analysis** | 50,000 | ~20 min | ~250 MB |
| **Near-Production** | 500,000 | ~3 hours | ~2.5 GB |
| **Final Run** | 2,241,611 (all) | ~12 hours | ~3.5 GB |

---

## ✅ Checklist

Before proceeding to analysis, ensure:

- [ ] ArXiv dataset downloaded (3.5 GB file in `data/raw/`)
- [ ] Sample extraction script runs successfully
- [ ] 10K sample created (`data/raw/arxiv_sample_10k.json`)
- [ ] Sample processed (`data/processed/arxiv_processed.csv`)
- [ ] Processed data has ~10,000 records
- [ ] Data includes `submission_year` and `main_categories` columns
- [ ] No critical errors during processing

---

## 📞 Need Help?

- **Kaggle Download Issues:** https://www.kaggle.com/docs/api
- **Script Errors:** Check error messages and ensure dependencies are installed
- **Data Quality Issues:** Document them in your analysis report

---

**Next Steps:** After completing data acquisition, proceed to `src/analysis/exploratory_analysis.py` to run the analysis!

