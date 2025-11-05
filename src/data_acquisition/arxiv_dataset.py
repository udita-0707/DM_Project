import pandas as pd
import os
import json
from datetime import datetime

# Configuration
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def download_and_process_arxiv_data(kaggle_json_path=None, use_sample=False, sample_filename='arxiv_sample_10k.json'):
    """
    Process the arXiv dataset from either the full dataset or a sample.
    The actual download from Kaggle must be done manually or using the Kaggle API
    with proper authentication, which is outside the scope of this script.
    
    Args:
        kaggle_json_path: Legacy parameter (kept for compatibility)
        use_sample: If True, process the sample file instead of full dataset
        sample_filename: Name of the sample file to process (default: arxiv_sample_10k.json)
    
    The expected file is 'arxiv-metadata-oai-snapshot.json' from the Kaggle dataset,
    or a sample file created by create_sample_data.py.
    """
    print("--- ArXiv Dataset Processing ---")
    
    # 1. Download/Locate the raw data
    if use_sample:
        raw_file_path = os.path.join(RAW_DATA_DIR, sample_filename)
        print(f"Processing SAMPLE data from: {sample_filename}")
    else:
        if kaggle_json_path is None:
            print("NOTE: The raw arXiv dataset (arxiv-metadata-oai-snapshot.json) must be downloaded manually from Kaggle.")
            print("Please place the JSON file in the 'data/raw' directory.")
            print(f"Expected path: {RAW_DATA_DIR}/arxiv-metadata-oai-snapshot.json")
            print("\nTIP: To process a sample instead, run:")
            print("  python3 src/data_acquisition/create_sample_data.py")
            print("  Then call this function with use_sample=True")
            return
        raw_file_path = os.path.join(RAW_DATA_DIR, 'arxiv-metadata-oai-snapshot.json')
    
    if not os.path.exists(raw_file_path):
        print(f"Error: Raw file not found at {raw_file_path}.")
        if use_sample:
            print("Please run create_sample_data.py first to create a sample.")
        else:
            print("Please download the full dataset from Kaggle.")
        return

    # 2. Load and process the large JSON file
    print("Loading and processing raw arXiv JSON file. This may take a while...")
    
    # The file is a JSON file where each line is a separate JSON object
    data = []
    try:
        with open(raw_file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping malformed line: {line[:50]}...")
                    continue
    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        return

    df = pd.DataFrame(data)
    print(f"Raw DataFrame loaded with {len(df)} records.")

    # 3. Clean and select relevant columns
    # Focus on columns needed for the exploratory questions
    df_processed = df[[
        'id', 'submitter', 'authors', 'title', 'categories', 'abstract', 
        'versions', 'update_date', 'comments', 'journal-ref', 'doi', 
        'report-no', 'license'
    ]].copy()
    
    # Extract submission year from 'versions' (first version date)
    def get_submission_year(versions):
        if versions:
            # Versions is a list of dicts, the first one is the original submission
            date_str = versions[0]['created']  # e.g., 'Mon, 20 Oct 2008 11:36:39 GMT'
            try:
                # Try parsing the full date string first
                return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z').year
            except ValueError:
                try:
                    # Fallback: parse just the date part without time
                    date_parts = ' '.join(date_str.split()[:4])  # "Mon, 20 Oct 2008"
                    return datetime.strptime(date_parts, '%a, %d %b %Y').year
                except ValueError:
                    return None
        return None

    df_processed['submission_year'] = df_processed['versions'].apply(get_submission_year)
    
    # Simplify categories into a list of main fields
    df_processed['main_categories'] = df_processed['categories'].apply(lambda x: [c.split('.')[0] for c in x.split(' ')])
    
    # 4. Save the processed data
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, 'arxiv_processed.csv')
    df_processed.to_csv(processed_file_path, index=False)
    print(f"Processed data saved to: {processed_file_path}")
    print(f"Total records processed: {len(df_processed)}")

if __name__ == "__main__":
    import sys
    
    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        print("Processing SAMPLE data (10K records)...")
        download_and_process_arxiv_data(use_sample=True)
    elif len(sys.argv) > 1 and sys.argv[1] == '--full':
        print("Processing FULL dataset (2.2M+ records)...")
        download_and_process_arxiv_data(kaggle_json_path='data/raw/arxiv-metadata-oai-snapshot.json')
    else:
        # Check if sample exists, default to that
        sample_path = os.path.join(RAW_DATA_DIR, 'arxiv_sample_10k.json')
        if os.path.exists(sample_path):
            print("Sample file found. Processing sample data...")
            print("(Use --full flag to process complete dataset instead)")
            download_and_process_arxiv_data(use_sample=True)
        else:
            print("ArXiv processing script")
            print("\nUsage:")
            print("  python3 arxiv_dataset.py --sample   # Process sample data (recommended)")
            print("  python3 arxiv_dataset.py --full     # Process full dataset")
            print("\nNo sample file found. Please run create_sample_data.py first.")

