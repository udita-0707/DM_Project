#!/usr/bin/env python3
"""
Optimized Batch Citation Enrichment Script

Enhanced version with better batch processing, parallelization, and progress tracking.
Designed to efficiently enrich larger samples (50K-100K papers) from the full dataset.

Usage:
    # Enrich 50K papers with optimized batch processing
    python3 optimized_batch_enrichment.py --sample-size 50000 --strategy stratified
    
    # Enrich with parallel processing (faster but uses more API calls)
    python3 optimized_batch_enrichment.py --sample-size 100000 --parallel --workers 4
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_acquisition.semanticscholar_api import SemanticScholarAPI
from src.data_acquisition.enrich_with_citations import (
    extract_arxiv_id, 
    sample_papers_strategically,
    enrich_papers_with_citations
)
from src.utils.data_loader import load_processed_data

# Configuration
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
ENRICHED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'enriched')
CACHE_DIR = os.path.join(project_root, 'data', 'cache')
os.makedirs(ENRICHED_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Thread-safe progress tracking
progress_lock = threading.Lock()
progress_stats = {
    'enriched': 0,
    'failed': 0,
    'total': 0,
    'api_calls': 0
}


def enrich_batch_parallel(arxiv_ids: List[str], api: SemanticScholarAPI, 
                         batch_size: int = 500) -> dict:
    """
    Enrich a batch of papers using parallel API calls (respecting rate limits).
    
    Args:
        arxiv_ids: List of arXiv IDs to enrich
        api: SemanticScholarAPI instance
        batch_size: Size of each API batch (max 500)
    
    Returns:
        Dictionary mapping arXiv IDs to paper data
    """
    global progress_stats
    
    results = {}
    
    # Split into batches
    for i in range(0, len(arxiv_ids), batch_size):
        batch_ids = arxiv_ids[i:i+batch_size]
        
        # Use batch API endpoint
        batch_results = api.batch_get_papers(batch_ids, batch_size=min(batch_size, 500))
        
        with progress_lock:
            progress_stats['api_calls'] += 1
        
        # Process results
        for arxiv_id, paper_data in batch_results.items():
            if paper_data:
                results[arxiv_id] = paper_data
                with progress_lock:
                    progress_stats['enriched'] += 1
            else:
                with progress_lock:
                    progress_stats['failed'] += 1
        
        # Rate limiting: 1 request per second
        if i + batch_size < len(arxiv_ids):
            time.sleep(1)
    
    return results


def enrich_chunk_worker(chunk_data: tuple, api_key: str) -> pd.DataFrame:
    """
    Worker function to enrich a chunk of papers.
    
    Args:
        chunk_data: Tuple of (chunk_index, DataFrame chunk)
        api_key: Semantic Scholar API key
    
    Returns:
        Enriched DataFrame chunk
    """
    chunk_idx, df_chunk = chunk_data
    
    # Create API instance for this worker
    api = SemanticScholarAPI(api_key=api_key, cache_dir=CACHE_DIR)
    
    # Extract arXiv IDs
    df_chunk['arxiv_id_clean'] = df_chunk['id'].apply(extract_arxiv_id)
    valid_ids = df_chunk[df_chunk['arxiv_id_clean'] != '']['arxiv_id_clean'].tolist()
    
    if not valid_ids:
        return df_chunk
    
    # Enrich batch
    enriched_data = enrich_batch_parallel(valid_ids, api)
    
    # Add citation data to DataFrame
    df_chunk['citation_count'] = 0
    df_chunk['reference_count'] = 0
    df_chunk['citation_data_fetched'] = False
    
    for idx, row in df_chunk.iterrows():
        arxiv_id = row['arxiv_id_clean']
        if arxiv_id in enriched_data and enriched_data[arxiv_id]:
            paper_data = enriched_data[arxiv_id]
            df_chunk.at[idx, 'citation_count'] = paper_data.get('citationCount', 0) or 0
            df_chunk.at[idx, 'reference_count'] = paper_data.get('referenceCount', 0) or 0
            df_chunk.at[idx, 'citation_data_fetched'] = True
    
    return df_chunk


def optimized_enrichment(df: pd.DataFrame, sample_size: int, strategy: str,
                         api_key: str, chunk_size: int = 10000, 
                         n_workers: int = 1, resume_file: Optional[str] = None):
    """
    Optimized enrichment with chunked processing and optional parallelization.
    
    Args:
        df: Full DataFrame to sample from
        sample_size: Number of papers to enrich
        strategy: Sampling strategy ('random', 'stratified', etc.)
        api_key: Semantic Scholar API key
        chunk_size: Size of chunks for processing
        n_workers: Number of parallel workers (1 = sequential)
        resume_file: Path to resume from previous run
    """
    global progress_stats
    
    print("="*80)
    print("OPTIMIZED BATCH ENRICHMENT")
    print("="*80)
    print(f"Total papers available: {len(df):,}")
    print(f"Sample size: {sample_size:,}")
    print(f"Strategy: {strategy}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Parallel workers: {n_workers}")
    print()
    
    # Sample papers
    print("Sampling papers...")
    df_sampled = sample_papers_strategically(
        df, strategy=strategy, sample_size=sample_size, random_seed=42
    )
    print(f"Selected {len(df_sampled):,} papers for enrichment")
    
    # Check for resume
    enriched_ids = set()
    if resume_file and os.path.exists(resume_file):
        resume_df = pd.read_csv(resume_file)
        enriched_ids = set(resume_df[resume_df['citation_data_fetched'] == True]['id'].astype(str))
        print(f"Resuming: Found {len(enriched_ids):,} already enriched papers")
    
    # Filter out already enriched
    df_to_enrich = df_sampled[~df_sampled['id'].astype(str).isin(enriched_ids)]
    print(f"Papers remaining to enrich: {len(df_to_enrich):,}")
    
    if len(df_to_enrich) == 0:
        print("All papers already enriched!")
        return resume_file
    
    # Initialize progress
    progress_stats['total'] = len(df_to_enrich)
    progress_stats['enriched'] = len(enriched_ids)
    progress_stats['failed'] = 0
    progress_stats['api_calls'] = 0
    
    # Split into chunks
    chunks = []
    for i in range(0, len(df_to_enrich), chunk_size):
        chunk = df_to_enrich.iloc[i:i+chunk_size]
        chunks.append((i // chunk_size, chunk))
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Process chunks
    enriched_chunks = []
    start_time = time.time()
    
    if n_workers > 1:
        # Parallel processing
        print(f"Using {n_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(enrich_chunk_worker, chunk, api_key): chunk 
                      for chunk in chunks}
            
            for future in as_completed(futures):
                chunk_idx, _ = futures[future]
                try:
                    enriched_chunk = future.result()
                    enriched_chunks.append(enriched_chunk)
                    
                    elapsed = time.time() - start_time
                    rate = progress_stats['enriched'] / elapsed if elapsed > 0 else 0
                    remaining = (progress_stats['total'] - progress_stats['enriched']) / rate if rate > 0 else 0
                    
                    print(f"Chunk {chunk_idx+1}/{len(chunks)} complete | "
                          f"Enriched: {progress_stats['enriched']:,}/{progress_stats['total']:,} | "
                          f"Rate: {rate:.1f} papers/sec | "
                          f"ETA: {remaining/60:.1f} min")
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
    else:
        # Sequential processing
        for chunk_idx, chunk in chunks:
            enriched_chunk = enrich_chunk_worker((chunk_idx, chunk), api_key)
            enriched_chunks.append(enriched_chunk)
            
            elapsed = time.time() - start_time
            rate = progress_stats['enriched'] / elapsed if elapsed > 0 else 0
            remaining = (progress_stats['total'] - progress_stats['enriched']) / rate if rate > 0 else 0
            
            print(f"Chunk {chunk_idx+1}/{len(chunks)} complete | "
                  f"Enriched: {progress_stats['enriched']:,}/{progress_stats['total']:,} | "
                  f"Rate: {rate:.1f} papers/sec | "
                  f"ETA: {remaining/60:.1f} min")
    
    # Combine chunks
    print("\nCombining results...")
    df_enriched = pd.concat(enriched_chunks, ignore_index=True)
    
    # Merge with resume data if exists
    if resume_file and os.path.exists(resume_file):
        resume_df = pd.read_csv(resume_file)
        df_enriched = pd.concat([resume_df, df_enriched], ignore_index=True)
        df_enriched = df_enriched.drop_duplicates(subset=['id'], keep='last')
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_str = strategy if strategy else 'random'
    size_str = f"_{len(df_enriched)}"
    output_path = os.path.join(ENRICHED_DATA_DIR, 
                              f"arxiv_enriched_{strategy_str}{size_str}_{timestamp}.csv")
    
    df_enriched.to_csv(output_path, index=False)
    
    # Save API stats
    api_stats = {
        'total_requests': progress_stats['api_calls'],
        'cached_entries': 0,  # Would need to track this separately
        'api_key_configured': bool(api_key),
        'enriched_count': progress_stats['enriched'],
        'failed_count': progress_stats['failed'],
        'success_rate': progress_stats['enriched'] / progress_stats['total'] * 100 if progress_stats['total'] > 0 else 0
    }
    
    stats_path = output_path.replace('.csv', '_api_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(api_stats, f, indent=2)
    
    print(f"\n✓ Enrichment complete!")
    print(f"  Output: {output_path}")
    print(f"  Enriched: {progress_stats['enriched']:,} papers")
    print(f"  Failed: {progress_stats['failed']:,} papers")
    print(f"  Success rate: {api_stats['success_rate']:.2f}%")
    print(f"  Total API calls: {progress_stats['api_calls']:,}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Optimized batch citation enrichment')
    parser.add_argument('--sample-size', type=int, required=True,
                       help='Number of papers to enrich')
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random', 'stratified', 'year', 'category'],
                       help='Sampling strategy')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for processing (default: 10000)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--resume', type=str,
                       help='Resume from previous enrichment file')
    parser.add_argument('--api-key', type=str,
                       default=os.environ.get('SEMANTIC_SCHOLAR_API_KEY'),
                       help='Semantic Scholar API key')
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API key required. Set SEMANTIC_SCHOLAR_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Load data
    print("Loading processed data...")
    try:
        df = load_processed_data(mock=False, use_enriched=False)
        print(f"Loaded {len(df):,} papers")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run: python3 src/data_acquisition/arxiv_dataset.py --full")
        sys.exit(1)
    
    # Run enrichment
    output_path = optimized_enrichment(
        df=df,
        sample_size=args.sample_size,
        strategy=args.strategy,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        n_workers=args.workers if args.parallel else 1,
        resume_file=args.resume
    )
    
    print(f"\n✓ Done! Enriched data saved to: {output_path}")


if __name__ == "__main__":
    main()
