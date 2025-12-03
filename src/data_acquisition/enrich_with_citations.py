#!/usr/bin/env python3
"""
Batch Citation Enrichment Script

This script enriches arXiv papers with citation data from Semantic Scholar API.
Uses intelligent sampling strategies to work within API rate limits.

Usage:
    # Enrich a random sample of 10K papers
    python3 enrich_with_citations.py --sample-size 10000 --strategy random
    
    # Enrich papers from specific years
    python3 enrich_with_citations.py --years 2020 2021 2022 --strategy year
    
    # Enrich papers from top categories
    python3 enrich_with_citations.py --top-categories 5 --strategy category
    
    # Resume from previous run
    python3 enrich_with_citations.py --resume
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
from datetime import datetime
from typing import List, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_acquisition.semanticscholar_api import SemanticScholarAPI
from src.utils.data_loader import load_processed_data

# Configuration
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
ENRICHED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'enriched')
CACHE_DIR = os.path.join(project_root, 'data', 'cache')
os.makedirs(ENRICHED_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_arxiv_id(paper_id: str) -> str:
    """
    Extract clean arXiv ID from various formats.
    
    Handles formats like "arXiv:2101.12345v1", "2101.12345", or NaN values.
    
    Args:
        paper_id: ArXiv ID string or NaN/None
    
    Returns:
        str: Clean arXiv ID (e.g., "2101.12345") or empty string if invalid
    """
    # Handle NaN/None values
    if pd.isna(paper_id) or paper_id is None:
        return ""
    
    # Convert to string if not already
    paper_id = str(paper_id)
    
    # Handle formats like "arXiv:2101.12345v1" or "2101.12345"
    if ':' in paper_id:
        paper_id = paper_id.split(':')[1]
    if 'v' in paper_id:
        paper_id = paper_id.split('v')[0]
    return paper_id.strip()

def sample_papers_strategically(df: pd.DataFrame, strategy: str, 
                                sample_size: Optional[int] = None,
                                years: Optional[List[int]] = None,
                                top_categories: Optional[int] = None,
                                random_seed: int = 42) -> pd.DataFrame:
    """
    Sample papers using different strategies.
    
    Args:
        df: DataFrame with processed arXiv data
        strategy: 'random', 'year', 'category', 'stratified', or 'all'
        sample_size: Number of papers to sample (for random/stratified)
        years: List of years to filter (for year strategy)
        top_categories: Number of top categories to include (for category strategy)
        random_seed: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    np.random.seed(random_seed)
    
    if strategy == 'all':
        return df.copy()
    
    elif strategy == 'random':
        if sample_size is None:
            sample_size = min(10000, len(df))
        return df.sample(n=min(sample_size, len(df)), random_state=random_seed)
    
    elif strategy == 'year':
        if years:
            filtered = df[df['submission_year'].isin(years)]
            if sample_size:
                return filtered.sample(n=min(sample_size, len(filtered)), random_state=random_seed)
            return filtered
        else:
            print("Warning: --years not specified, using random strategy instead")
            return sample_papers_strategically(df, 'random', sample_size, random_seed=random_seed)
    
    elif strategy == 'category':
        # Get top categories by paper count
        import ast
        df['main_categories'] = df['main_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['first_category'] = df['main_categories'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown'
        )
        
        if top_categories:
            top_cats = df['first_category'].value_counts().head(top_categories).index.tolist()
            filtered = df[df['first_category'].isin(top_cats)]
            if sample_size:
                return filtered.sample(n=min(sample_size, len(filtered)), random_state=random_seed)
            return filtered
        else:
            print("Warning: --top-categories not specified, using random strategy instead")
            return sample_papers_strategically(df, 'random', sample_size, random_seed=random_seed)
    
    elif strategy == 'stratified':
        # Stratified sampling by year and category
        import ast
        df['main_categories'] = df['main_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['first_category'] = df['main_categories'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown'
        )
        
        if sample_size is None:
            sample_size = min(10000, len(df))
        
        # Sample proportionally from each year-category combination
        sampled = df.groupby(['submission_year', 'first_category'], group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))), 
                             random_state=random_seed)
        )
        
        # If we got fewer than requested, add random samples
        if len(sampled) < sample_size:
            remaining = df[~df.index.isin(sampled.index)]
            additional = remaining.sample(n=min(sample_size - len(sampled), len(remaining)), 
                                        random_state=random_seed)
            sampled = pd.concat([sampled, additional])
        
        return sampled.head(sample_size)
    
    else:
        print(f"Unknown strategy: {strategy}. Using random strategy.")
        return sample_papers_strategically(df, 'random', sample_size, random_seed=random_seed)

def enrich_papers_with_citations(df: pd.DataFrame, api: SemanticScholarAPI, 
                                 resume_file: Optional[str] = None,
                                 use_batch: bool = False) -> pd.DataFrame:
    """
    Enrich papers with citation data from Semantic Scholar API.
    
    Args:
        df: DataFrame with papers to enrich
        api: SemanticScholarAPI instance
        resume_file: Optional path to resume from previous run
    
    Returns:
        DataFrame with added 'citation_count' and 'reference_count' columns
    """
    # Load previous progress if resuming
    enriched_ids = set()
    if resume_file and os.path.exists(resume_file):
        print(f"Resuming from {resume_file}...")
        resume_df = pd.read_csv(resume_file)
        enriched_ids = set(resume_df['id'].astype(str))
        print(f"Found {len(enriched_ids)} already enriched papers")
    
    # Initialize new columns
    df['citation_count'] = None
    df['reference_count'] = None
    df['citation_data_fetched'] = False
    
    # Extract arXiv IDs
    df['arxiv_id_clean'] = df['id'].apply(extract_arxiv_id)
    
    # Filter out rows with empty/invalid IDs
    valid_mask = df['arxiv_id_clean'].str.len() > 0
    df_valid = df[valid_mask].copy()
    invalid_count = (~valid_mask).sum()
    
    if invalid_count > 0:
        print(f"Warning: {invalid_count} papers have invalid/missing IDs and will be skipped")
    
    total = len(df_valid)
    enriched = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"Enriching {total} papers with citation data")
    if use_batch:
        print(f"Using batch API mode (recommended for large datasets)")
    print(f"{'='*60}\n")
    
    # Use batch API if enabled and dataset is large enough
    if use_batch and total > 50:
        return _enrich_with_batch_api(df, df_valid, valid_mask, invalid_count, 
                                     enriched_ids, api, resume_file)
    
    # Use itertuples for better performance than iterrows
    # Get column indices for faster access
    id_col_idx = df_valid.columns.get_loc('id')
    arxiv_id_col_idx = df_valid.columns.get_loc('arxiv_id_clean')
    
    citation_counts = []
    reference_counts = []
    fetched_flags = []
    
    for i, row in enumerate(df_valid.itertuples(index=False)):
        # Access columns by index for reliability
        row_id = row[id_col_idx]
        arxiv_id = row[arxiv_id_col_idx]
        
        # Skip if already enriched
        if str(row_id) in enriched_ids:
            citation_counts.append(df_valid.iloc[i]['citation_count'] if 'citation_count' in df_valid.columns else None)
            reference_counts.append(df_valid.iloc[i]['reference_count'] if 'reference_count' in df_valid.columns else None)
            fetched_flags.append(df_valid.iloc[i]['citation_data_fetched'] if 'citation_data_fetched' in df_valid.columns else False)
            enriched += 1
            continue
        
        # Skip if ID is empty
        if not arxiv_id or len(arxiv_id) == 0:
            citation_counts.append(0)
            reference_counts.append(0)
            fetched_flags.append(False)
            failed += 1
            continue
        
        # Get paper data from API
        try:
            paper_data = api.get_paper_data(arxiv_id)
            
            if paper_data:
                citation_counts.append(paper_data.get('citationCount', 0))
                reference_counts.append(paper_data.get('referenceCount', 0))
                fetched_flags.append(True)
                enriched += 1
            else:
                citation_counts.append(0)
                reference_counts.append(0)
                fetched_flags.append(False)
                failed += 1
        except Exception as e:
            print(f"Error processing {arxiv_id}: {e}")
            citation_counts.append(0)
            reference_counts.append(0)
            fetched_flags.append(False)
            failed += 1
        
        # Progress update every 100 papers
        if (enriched + failed) % 100 == 0:
            progress = (enriched + failed) / total * 100
            print(f"Progress: {enriched + failed}/{total} ({progress:.1f}%) | "
                  f"Enriched: {enriched} | Failed: {failed} | "
                  f"API calls: {api.total_requests}")
            
            # Save intermediate progress (only every 500 papers to reduce I/O)
            if resume_file and (enriched + failed) % 500 == 0:
                # Update DataFrame with current progress
                df_valid['citation_count'] = citation_counts + [None] * (len(df_valid) - len(citation_counts))
                df_valid['reference_count'] = reference_counts + [None] * (len(df_valid) - len(reference_counts))
                df_valid['citation_data_fetched'] = fetched_flags + [False] * (len(df_valid) - len(fetched_flags))
                
                # Merge with invalid rows if needed
                if invalid_count > 0:
                    df_invalid = df[~valid_mask].copy()
                    df_invalid['citation_count'] = None
                    df_invalid['reference_count'] = None
                    df_invalid['citation_data_fetched'] = False
                    df_temp = pd.concat([df_valid, df_invalid], ignore_index=True)
                else:
                    df_temp = df_valid.copy()
                
                df_temp.to_csv(resume_file, index=False)
                print(f"  → Saved progress to {resume_file}")
    
    # Update DataFrame with all results
    df_valid['citation_count'] = citation_counts
    df_valid['reference_count'] = reference_counts
    df_valid['citation_data_fetched'] = fetched_flags
    
    # Force save cache at the end
    api.save_cache()
    
    print(f"\n{'='*60}")
    print(f"Enrichment complete!")
    print(f"  Total papers processed: {total}")
    print(f"  Successfully enriched: {enriched}")
    print(f"  Failed/not found: {failed}")
    if invalid_count > 0:
        print(f"  Skipped (invalid IDs): {invalid_count}")
    print(f"  Total API calls: {api.total_requests}")
    print(f"  Cached entries: {len(api.cache)}")
    print(f"{'='*60}\n")
    
    # Merge back with invalid rows (they'll have null citation data)
    if invalid_count > 0:
        df_invalid = df[~valid_mask].copy()
        df_invalid['citation_count'] = None
        df_invalid['reference_count'] = None
        df_invalid['citation_data_fetched'] = False
        df = pd.concat([df_valid, df_invalid], ignore_index=True)
    else:
        df = df_valid
    
    return df

def _enrich_with_batch_api(df: pd.DataFrame, df_valid: pd.DataFrame, 
                          valid_mask: pd.Series, invalid_count: int,
                          enriched_ids: set, api: SemanticScholarAPI,
                          resume_file: Optional[str] = None) -> pd.DataFrame:
    """
    Enrich papers using batch API endpoint (more efficient for large datasets).
    
    Args:
        df: Original DataFrame
        df_valid: DataFrame with valid arXiv IDs
        valid_mask: Boolean mask for valid rows
        invalid_count: Number of invalid rows
        enriched_ids: Set of already enriched paper IDs
        api: SemanticScholarAPI instance
        resume_file: Optional resume file path
    
    Returns:
        Enriched DataFrame
    """
    # Collect arXiv IDs that need enrichment
    arxiv_ids_to_fetch = []
    row_indices = []
    
    for idx, row in df_valid.iterrows():
        if str(row['id']) not in enriched_ids:
            arxiv_id = row['arxiv_id_clean']
            if arxiv_id and len(arxiv_id) > 0:
                arxiv_ids_to_fetch.append(arxiv_id)
                row_indices.append(idx)
    
    total_to_fetch = len(arxiv_ids_to_fetch)
    print(f"Fetching {total_to_fetch} papers using batch API...")
    
    # Fetch papers in batches
    papers_data = api.batch_get_papers(arxiv_ids_to_fetch, 
                                       fields="citationCount,referenceCount")
    
    # Update DataFrame with results
    enriched = 0
    failed = 0
    
    for i, (arxiv_id, idx) in enumerate(zip(arxiv_ids_to_fetch, row_indices)):
        paper_data = papers_data.get(arxiv_id)
        
        if paper_data:
            df_valid.at[idx, 'citation_count'] = paper_data.get('citationCount', 0)
            df_valid.at[idx, 'reference_count'] = paper_data.get('referenceCount', 0)
            df_valid.at[idx, 'citation_data_fetched'] = True
            enriched += 1
        else:
            df_valid.at[idx, 'citation_count'] = 0
            df_valid.at[idx, 'reference_count'] = 0
            df_valid.at[idx, 'citation_data_fetched'] = False
            failed += 1
        
        # Progress update
        if (i + 1) % 100 == 0 or (i + 1) == total_to_fetch:
            progress = (i + 1) / total_to_fetch * 100
            print(f"Progress: {i + 1}/{total_to_fetch} ({progress:.1f}%) | "
                  f"Enriched: {enriched} | Failed: {failed} | "
                  f"API calls: {api.total_requests}")
            
            # Save intermediate progress
            if resume_file and (i + 1) % 500 == 0:
                if invalid_count > 0:
                    df_invalid = df[~valid_mask].copy()
                    df_invalid['citation_count'] = None
                    df_invalid['reference_count'] = None
                    df_invalid['citation_data_fetched'] = False
                    df_temp = pd.concat([df_valid, df_invalid], ignore_index=True)
                else:
                    df_temp = df_valid.copy()
                df_temp.to_csv(resume_file, index=False)
                print(f"  → Saved progress to {resume_file}")
    
    # Force save cache
    api.save_cache()
    
    print(f"\n{'='*60}")
    print(f"Enrichment complete!")
    print(f"  Total papers processed: {total_to_fetch}")
    print(f"  Successfully enriched: {enriched}")
    print(f"  Failed/not found: {failed}")
    if invalid_count > 0:
        print(f"  Skipped (invalid IDs): {invalid_count}")
    print(f"  Total API calls: {api.total_requests}")
    print(f"  Cached entries: {len(api.cache)}")
    print(f"{'='*60}\n")
    
    # Merge back with invalid rows
    if invalid_count > 0:
        df_invalid = df[~valid_mask].copy()
        df_invalid['citation_count'] = None
        df_invalid['reference_count'] = None
        df_invalid['citation_data_fetched'] = False
        df = pd.concat([df_valid, df_invalid], ignore_index=True)
    else:
        df = df_valid
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Enrich arXiv papers with Semantic Scholar citation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random sample of 10K papers
  python3 enrich_with_citations.py --sample-size 10000 --strategy random
  
  # Papers from 2020-2022
  python3 enrich_with_citations.py --years 2020 2021 2022 --strategy year
  
  # Top 5 categories, 5K papers each
  python3 enrich_with_citations.py --top-categories 5 --sample-size 5000 --strategy category
  
  # Stratified sample (balanced across years/categories)
  python3 enrich_with_citations.py --sample-size 20000 --strategy stratified
        """
    )
    
    parser.add_argument('--sample-size', type=int, help='Number of papers to sample')
    parser.add_argument('--strategy', choices=['random', 'year', 'category', 'stratified', 'all'],
                       default='random', help='Sampling strategy')
    parser.add_argument('--years', nargs='+', type=int, help='Years to filter (for year strategy)')
    parser.add_argument('--top-categories', type=int, help='Number of top categories (for category strategy)')
    parser.add_argument('--api-key', type=str, help='Semantic Scholar API key (or set SEMANTIC_SCHOLAR_API_KEY env var)')
    parser.add_argument('--resume', type=str, help='Resume from previous run (path to CSV file)')
    parser.add_argument('--output', type=str, help='Output file path (default: auto-generated)')
    parser.add_argument('--cache-file', type=str, 
                       default=os.path.join(CACHE_DIR, 'semanticscholar_cache.json'),
                       help='Path to API cache file')
    parser.add_argument('--use-batch', action='store_true',
                       help='Use batch API endpoint (faster for large datasets, recommended for 1000+ papers)')
    
    args = parser.parse_args()
    
    # Initialize API
    api = SemanticScholarAPI(api_key=args.api_key, cache_file=args.cache_file)
    
    if not api.api_key:
        print("WARNING: No API key provided!")
        print("Set SEMANTIC_SCHOLAR_API_KEY environment variable or use --api-key")
        print("Continuing without API key (will use unauthenticated requests)...")
    
    # Load processed data
    print("Loading processed arXiv data...")
    try:
        df = load_processed_data(mock=False)
        print(f"Loaded {len(df)} papers")
    except FileNotFoundError:
        print("Error: Processed data not found. Please run arxiv_dataset.py first.")
        return
    
    # Sample papers based on strategy
    if args.resume:
        print(f"Loading data from resume file: {args.resume}")
        df_enriched = pd.read_csv(args.resume)
    else:
        print(f"\nSampling papers using '{args.strategy}' strategy...")
        df_sampled = sample_papers_strategically(
            df, 
            strategy=args.strategy,
            sample_size=args.sample_size,
            years=args.years,
            top_categories=args.top_categories
        )
        print(f"Selected {len(df_sampled)} papers for enrichment")
        df_enriched = df_sampled.copy()
    
    # Enrich with citation data
    df_enriched = enrich_papers_with_citations(df_enriched, api, 
                                               resume_file=args.resume,
                                               use_batch=args.use_batch)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_str = args.strategy
        size_str = f"_{len(df_enriched)}" if args.sample_size else ""
        output_path = os.path.join(ENRICHED_DATA_DIR, f"arxiv_enriched_{strategy_str}{size_str}_{timestamp}.csv")
    
    df_enriched.to_csv(output_path, index=False)
    print(f"Enriched data saved to: {output_path}")
    
    # Save API stats
    stats = api.get_stats()
    stats_path = output_path.replace('.csv', '_api_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"API statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()

