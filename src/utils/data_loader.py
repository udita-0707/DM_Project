"""
Data Loader Utility Module

Provides functions to load processed arXiv data, with automatic detection of
enriched datasets containing citation data from Semantic Scholar API.
"""

import pandas as pd
import os
import glob
from typing import Optional


def load_processed_data(mock=True, use_enriched=True):
    """
    Load processed arXiv dataset with automatic enriched data detection.
    
    Priority order:
    1. Enriched data (if use_enriched=True and available)
    2. Full processed data (if mock=False)
    3. Mock data (for testing)
    
    Args:
        mock: If True, prefer mock data for testing (default: True)
        use_enriched: If True, automatically load enriched data with citation counts (default: True)
    
    Returns:
        pd.DataFrame: Processed arXiv data with optional citation enrichment
    
    Raises:
        FileNotFoundError: If no data files are found
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_path = os.path.join(base_dir, 'data', 'processed', 'arxiv_processed.csv')
    mock_path = os.path.join(base_dir, 'data', 'processed', 'arxiv_processed_mock.csv')
    enriched_dir = os.path.join(base_dir, 'data', 'processed', 'enriched')
    
    # Try to load enriched data first if requested
    if use_enriched and os.path.exists(enriched_dir):
        enriched_files = glob.glob(os.path.join(enriched_dir, 'arxiv_enriched_*.csv'))
        if enriched_files:
            # Get most recent enriched file
            latest_enriched = max(enriched_files, key=os.path.getmtime)
            print(f"Loading enriched data with citation counts from: {latest_enriched}")
            df = pd.read_csv(latest_enriched, low_memory=False)
            print(f"  Found {df['citation_data_fetched'].sum() if 'citation_data_fetched' in df.columns else 0} papers with real citation data")
            return df
    
    # Fall back to regular processed data
    if os.path.exists(processed_path) and not mock:
        print(f"Loading processed data from: {processed_path}")
        return pd.read_csv(processed_path, low_memory=False)
    elif os.path.exists(mock_path):
        print(f"Loading mock data from: {mock_path}")
        return pd.read_csv(mock_path, low_memory=False)
    else:
        raise FileNotFoundError("Processed data file not found. Please run the data acquisition script or ensure mock data is present.")

if __name__ == "__main__":
    # Example usage
    try:
        df = load_processed_data(mock=True)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(df.head())
    except FileNotFoundError as e:
        print(e)

