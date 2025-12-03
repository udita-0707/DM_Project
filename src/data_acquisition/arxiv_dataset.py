"""
ArXiv Dataset Processing Module

Processes raw arXiv JSON data into a cleaned CSV format suitable for analysis.
Supports both full dataset (2.2M+ papers) and sample datasets.
"""

import pandas as pd
import os
import json
from datetime import datetime

# Directory configuration
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def get_submission_year(versions):
    """
    PREPROCESSING: Extract submission year from paper versions data.
    
    Why: We need the submission year for temporal analysis (research growth over time,
    citation half-life, etc.). The year is stored in the first version's timestamp.
    
    How: Parse the date string from the first version (earliest submission) and extract
    the year. Uses fallback parsing if the primary format fails.
    
    Args:
        versions: List of version dictionaries with 'created' timestamps
    
    Returns:
        int: Year of first submission, or None if parsing fails
    """
    if versions:
        # Get the first version's creation date (earliest submission)
        date_str = versions[0]['created']  # Format: 'Mon, 20 Oct 2008 11:36:39 GMT'
        try:
            # Try parsing the full date-time format
            return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z').year
        except ValueError:
            try:
                # Fallback: parse date without time (in case format differs)
                date_parts = ' '.join(date_str.split()[:4])
                return datetime.strptime(date_parts, '%a, %d %b %Y').year
            except ValueError:
                return None
    return None


def download_and_process_arxiv_data(kaggle_json_path=None, use_sample=False, sample_filename='arxiv_sample_10k.json'):
    """
    Process arXiv dataset from JSON file into cleaned CSV format.
    
    Extracts key fields, parses submission years, and simplifies category classifications.
    The raw JSON file must be downloaded manually from Kaggle and placed in data/raw/.
    
    Args:
        kaggle_json_path: Legacy parameter (kept for compatibility)
        use_sample: If True, process sample file instead of full dataset
        sample_filename: Name of sample file to process (default: arxiv_sample_10k.json)
    
    Returns:
        None: Saves processed data to data/processed/arxiv_processed.csv
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

    # PREPROCESSING: Load and process the large JSON file in chunks (memory-efficient)
    # Why: The full dataset is 3.5GB+ and cannot fit in memory. We process in chunks.
    # How: Read JSON line-by-line, accumulate records, process in batches of 50K.
    print("Loading and processing raw arXiv JSON file. This may take a while...")
    print("Processing in chunks to manage memory efficiently...")
    
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, 'arxiv_processed.csv')
    chunk_size = 50000  # Process 50K records at a time to manage memory
    total_processed = 0
    first_chunk = True
    
    try:
        with open(raw_file_path, 'r') as f:
            chunk_data = []
            
            # Read JSON file line-by-line (each line is a JSON object)
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    chunk_data.append(record)
                    
                    # PREPROCESSING: Process chunk when it reaches chunk_size
                    # This prevents memory overflow when processing 2.2M+ records
                    if len(chunk_data) >= chunk_size:
                        df_chunk = pd.DataFrame(chunk_data)
                        
                        # PREPROCESSING: Select only relevant columns
                        # Why: Raw data has many fields we don't need. This reduces memory usage.
                        df_processed_chunk = df_chunk[[
                            'id', 'submitter', 'authors', 'title', 'categories', 'abstract', 
                            'versions', 'update_date', 'comments', 'journal-ref', 'doi', 
                            'report-no', 'license'
                        ]].copy()
                        
                        # PREPROCESSING: Extract submission year from version timestamps
                        # Why: Needed for temporal analysis (research growth, citation half-life)
                        df_processed_chunk['submission_year'] = df_processed_chunk['versions'].apply(get_submission_year)
                        
                        # PREPROCESSING: Simplify categories (e.g., "cs.AI math.CO" -> ["cs", "math"])
                        # Why: Hierarchical categories are complex. We need main category names for analysis.
                        # How: Split by space, then extract prefix before the dot
                        df_processed_chunk['main_categories'] = df_processed_chunk['categories'].apply(
                            lambda x: [c.split('.')[0] for c in x.split(' ')]
                        )
                        
                        # Append to CSV (append mode after first chunk)
                        df_processed_chunk.to_csv(
                            processed_file_path, 
                            mode='w' if first_chunk else 'a',
                            header=first_chunk,
                            index=False
                        )
                        
                        total_processed += len(df_processed_chunk)
                        print(f"Processed {total_processed:,} records... (chunk of {len(chunk_data):,})")
                        
                        # Reset for next chunk
                        chunk_data = []
                        first_chunk = False
                        
                except json.JSONDecodeError:
                    print(f"Skipping malformed line {line_num}: {line[:50]}...")
                    continue
                except KeyError as e:
                    print(f"Skipping record {line_num} with missing field: {e}")
                    continue
            
            # Process remaining records in final chunk
            if chunk_data:
                df_chunk = pd.DataFrame(chunk_data)
                df_processed_chunk = df_chunk[[
                    'id', 'submitter', 'authors', 'title', 'categories', 'abstract', 
                    'versions', 'update_date', 'comments', 'journal-ref', 'doi', 
                    'report-no', 'license'
                ]].copy()
                
                df_processed_chunk['submission_year'] = df_processed_chunk['versions'].apply(get_submission_year)
                df_processed_chunk['main_categories'] = df_processed_chunk['categories'].apply(
                    lambda x: [c.split('.')[0] for c in x.split(' ')]
                )
                
                df_processed_chunk.to_csv(
                    processed_file_path,
                    mode='a',
                    header=False,
                    index=False
                )
                
                total_processed += len(df_processed_chunk)
                print(f"Processed final chunk: {len(chunk_data):,} records")
        
        print(f"\n✓ Processing complete!")
        print(f"Processed data saved to: {processed_file_path}")
        print(f"Total records processed: {total_processed:,}")
        
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        import traceback
        traceback.print_exc()
        return

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

