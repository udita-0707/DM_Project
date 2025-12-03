#!/usr/bin/env python3
"""
Batch Full Dataset Analysis Script

Analyzes the complete 2.2M+ arXiv dataset efficiently using chunked processing.
This script performs all analyses that don't require citation data on the full dataset.

Usage:
    python3 batch_full_dataset_analysis.py --chunk-size 100000
    python3 batch_full_dataset_analysis.py --full --parallel
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import ast
from datetime import datetime
from typing import Dict, List, Optional
import multiprocessing as mp
from functools import partial

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data

# Output directory
OUTPUT_DIR = os.path.join(project_root, 'data', 'processed', 'analysis_results', 'full_dataset_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_research_growth_chunk(df_chunk: pd.DataFrame) -> Dict:
    """Analyze research growth for a chunk of data."""
    results = {}
    
    # Parse main_categories and extract first category
    if 'main_categories' in df_chunk.columns:
        try:
            df_chunk = df_chunk.copy()
            df_chunk['main_categories'] = df_chunk['main_categories'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x
            )
            df_chunk['first_category'] = df_chunk['main_categories'].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown'
            )
            
            category_counts = df_chunk['first_category'].value_counts().to_dict()
            results['category_counts'] = category_counts
            
            # Category-year counts for breakthrough detection
            if 'submission_year' in df_chunk.columns:
                category_year_counts = df_chunk.groupby(['first_category', 'submission_year']).size().to_dict()
                results['category_year_counts'] = category_year_counts
        except Exception as e:
            print(f"Warning: Could not parse categories: {e}")
    
    # Year distribution
    if 'submission_year' in df_chunk.columns:
        year_counts = df_chunk['submission_year'].value_counts().sort_index().to_dict()
        results['year_counts'] = year_counts
    
    return results


def analyze_collaboration_chunk(df_chunk: pd.DataFrame) -> Dict:
    """Analyze collaboration patterns for a chunk."""
    results = {}
    
    # Compute author count from authors column
    if 'authors' in df_chunk.columns:
        try:
            df_chunk = df_chunk.copy()
            df_chunk['num_authors'] = df_chunk['authors'].apply(
                lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[')
                else len(x.split(',')) if isinstance(x, str) else 0
            )
            
            results['author_count_stats'] = {
                'mean': float(df_chunk['num_authors'].mean()),
                'median': float(df_chunk['num_authors'].median()),
                'std': float(df_chunk['num_authors'].std()),
                'max': int(df_chunk['num_authors'].max()),
            }
        except Exception as e:
            print(f"Warning: Could not parse authors: {e}")
    
    # Category distribution (if available)
    if 'main_categories' in df_chunk.columns:
        try:
            df_chunk = df_chunk.copy()
            df_chunk['main_categories'] = df_chunk['main_categories'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x
            )
            df_chunk['first_category'] = df_chunk['main_categories'].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown'
            )
            results['category_distribution'] = df_chunk['first_category'].value_counts().to_dict()
        except Exception as e:
            pass
    
    return results


def analyze_interdisciplinarity_chunk(df_chunk: pd.DataFrame) -> Dict:
    """Analyze interdisciplinarity for a chunk."""
    results = {}
    
    if 'main_categories' in df_chunk.columns:
        try:
            df_chunk = df_chunk.copy()
            df_chunk['main_categories'] = df_chunk['main_categories'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x
            )
            df_chunk['discipline_count'] = df_chunk['main_categories'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            results['discipline_distribution'] = df_chunk['discipline_count'].value_counts().to_dict()
            results['mean_disciplines'] = float(df_chunk['discipline_count'].mean())
            results['interdisciplinary_count'] = int((df_chunk['discipline_count'] > 1).sum())
            results['single_discipline_count'] = int((df_chunk['discipline_count'] == 1).sum())
        except Exception as e:
            print(f"Warning: Could not parse disciplines: {e}")
    
    return results


def analyze_keywords_chunk(df_chunk: pd.DataFrame) -> Dict:
    """Extract keywords from a chunk."""
    from collections import Counter
    
    results = {}
    
    # Combine title and abstract
    if 'title' in df_chunk.columns and 'abstract' in df_chunk.columns:
        text_data = (df_chunk['title'].fillna('') + ' ' + df_chunk['abstract'].fillna('')).str.lower()
        
        # Extract words (simple approach)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'this', 'that', 'these', 'those', 'paper', 'study', 'research', 'analysis'}
        
        all_words = []
        for text in text_data:
            words = [w.strip('.,!?;:()[]{}"\'-') for w in str(text).split() 
                    if len(w.strip('.,!?;:()[]{}"\'-')) > 3 and w.lower() not in stop_words]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        results['keyword_counts'] = dict(word_counts.most_common(100))
    
    return results


def process_chunk(chunk_data: tuple, analysis_type: str) -> Dict:
    """Process a single chunk of data."""
    chunk_idx, df_chunk = chunk_data
    
    try:
        if analysis_type == 'growth':
            return analyze_research_growth_chunk(df_chunk)
        elif analysis_type == 'collaboration':
            return analyze_collaboration_chunk(df_chunk)
        elif analysis_type == 'interdisciplinarity':
            return analyze_interdisciplinarity_chunk(df_chunk)
        elif analysis_type == 'keywords':
            return analyze_keywords_chunk(df_chunk)
        else:
            return {}
    except Exception as e:
        print(f"Error processing chunk {chunk_idx}: {e}")
        return {}


def merge_results(all_results: List[Dict]) -> Dict:
    """Merge results from multiple chunks."""
    merged = {}
    
    for result in all_results:
        if not result:  # Skip empty results
            continue
        for key, value in result.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, dict):
                # Merge dictionaries (e.g., category_counts, year_counts)
                if isinstance(merged[key], dict):
                    for k, v in value.items():
                        merged[key][k] = merged[key].get(k, 0) + v
                else:
                    merged[key] = value
            elif isinstance(value, (int, float)):
                # For numeric values, accumulate (but we'll recalculate stats anyway)
                if isinstance(merged[key], (int, float)):
                    merged[key] += value
                else:
                    merged[key] = value
    
    return merged


def analyze_full_dataset_chunked(file_path: str, chunk_size: int = 100000, 
                                 n_workers: int = 4, use_parallel: bool = False):
    """
    Analyze full dataset in chunks for memory efficiency.
    
    Args:
        file_path: Path to the full processed CSV file
        chunk_size: Number of rows to process per chunk
        n_workers: Number of parallel workers (if use_parallel=True)
        use_parallel: Whether to use parallel processing
    """
    print("="*80)
    print("BATCH FULL DATASET ANALYSIS")
    print("="*80)
    print(f"Input file: {file_path}")
    print(f"Chunk size: {chunk_size:,} rows")
    print(f"Parallel processing: {use_parallel} ({n_workers} workers)")
    print()
    
    # Get file size
    file_size = os.path.getsize(file_path) / (1024**3)  # GB
    print(f"File size: {file_size:.2f} GB")
    
    # Count total rows
    print("Counting total rows...")
    total_rows = 0
    with open(file_path, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    print(f"Total rows: {total_rows:,}")
    
    num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size > 0 else 0)
    print(f"Number of chunks: {num_chunks}")
    print()
    
    # Analysis types
    analysis_types = ['growth', 'collaboration', 'interdisciplinarity', 'keywords']
    
    all_results = {atype: [] for atype in analysis_types}
    
    # Process chunks
    print("Processing chunks...")
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    
    for chunk_idx, df_chunk in enumerate(chunk_iter, 1):
        print(f"Processing chunk {chunk_idx}/{num_chunks} ({len(df_chunk):,} rows)...", end='\r')
        
        if use_parallel and chunk_idx <= n_workers:
            # Process first n_workers chunks in parallel
            chunk_data = [(i, chunk) for i, chunk in enumerate([df_chunk])]
            with mp.Pool(n_workers) as pool:
                for atype in analysis_types:
                    results = pool.map(partial(process_chunk, analysis_type=atype), chunk_data)
                    all_results[atype].extend(results)
        else:
            # Sequential processing
            for atype in analysis_types:
                result = process_chunk((chunk_idx, df_chunk), atype)
                all_results[atype].append(result)
        
        if chunk_idx % 10 == 0:
            print(f"\n  Processed {chunk_idx} chunks ({chunk_idx * chunk_size:,} rows)")
    
    print(f"\n✓ Processed all {num_chunks} chunks")
    
    # Merge results
    print("\nMerging results...")
    merged_results = {}
    for atype in analysis_types:
        merged_results[atype] = merge_results(all_results[atype])
    
    # Save results
    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save category counts
    if 'category_counts' in merged_results['growth']:
        category_df = pd.DataFrame(list(merged_results['growth']['category_counts'].items()),
                                   columns=['category', 'count'])
        category_df = category_df.sort_values('count', ascending=False)
        category_df.to_csv(os.path.join(OUTPUT_DIR, f'category_counts_full_{timestamp}.csv'), index=False)
        print(f"  ✓ Saved category counts: {len(category_df)} categories")
    
    # Save year distribution
    if 'year_counts' in merged_results['growth']:
        year_df = pd.DataFrame(list(merged_results['growth']['year_counts'].items()),
                               columns=['year', 'count'])
        year_df = year_df.sort_values('year')
        year_df.to_csv(os.path.join(OUTPUT_DIR, f'year_distribution_full_{timestamp}.csv'), index=False)
        print(f"  ✓ Saved year distribution: {len(year_df)} years")
    
    # Save collaboration stats
    if 'author_count_stats' in merged_results['collaboration']:
        import json
        with open(os.path.join(OUTPUT_DIR, f'collaboration_stats_full_{timestamp}.json'), 'w') as f:
            json.dump(merged_results['collaboration'], f, indent=2)
        print(f"  ✓ Saved collaboration statistics")
    
    # Save interdisciplinarity stats
    if 'discipline_distribution' in merged_results['interdisciplinarity']:
        import json
        with open(os.path.join(OUTPUT_DIR, f'interdisciplinarity_stats_full_{timestamp}.json'), 'w') as f:
            json.dump(merged_results['interdisciplinarity'], f, indent=2)
        print(f"  ✓ Saved interdisciplinarity statistics")
    
    # Save top keywords
    if 'keyword_counts' in merged_results['keywords']:
        keyword_df = pd.DataFrame(list(merged_results['keywords']['keyword_counts'].items()),
                                  columns=['keyword', 'count'])
        keyword_df = keyword_df.sort_values('count', ascending=False)
        keyword_df.head(1000).to_csv(os.path.join(OUTPUT_DIR, f'top_keywords_full_{timestamp}.csv'), index=False)
        print(f"  ✓ Saved top keywords: {len(keyword_df)} keywords")
    
    print(f"\n✓ Analysis complete! Results saved to: {OUTPUT_DIR}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if 'category_counts' in merged_results['growth']:
        top_categories = sorted(merged_results['growth']['category_counts'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 Categories:")
        for cat, count in top_categories:
            print(f"  {cat}: {count:,} papers")
    
    if 'year_counts' in merged_results['growth']:
        years = sorted(merged_results['growth']['year_counts'].keys())
        print(f"\nYear Range: {min(years)} - {max(years)}")
        print(f"Total Years: {len(years)}")
    
    if 'author_count_stats' in merged_results['collaboration']:
        stats = merged_results['collaboration']['author_count_stats']
        print(f"\nAuthor Statistics:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Max: {stats['max']}")
    
    if 'discipline_distribution' in merged_results['interdisciplinarity']:
        inter = merged_results['interdisciplinarity']
        print(f"\nInterdisciplinarity:")
        print(f"  Single-discipline: {inter.get('single_discipline_count', 0):,}")
        print(f"  Interdisciplinary: {inter.get('interdisciplinary_count', 0):,}")
        print(f"  Mean disciplines per paper: {inter.get('mean_disciplines', 0):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Batch analysis of full arXiv dataset')
    parser.add_argument('--file', type=str, 
                       default='data/processed/arxiv_processed.csv',
                       help='Path to processed CSV file')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Number of rows per chunk (default: 100000)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Handle empty file path (from unset environment variable)
    if not args.file or args.file.strip() == '':
        args.file = 'data/processed/arxiv_processed.csv'
    
    # If path is already absolute, use it; otherwise join with project root
    if os.path.isabs(args.file):
        file_path = args.file
    else:
        file_path = os.path.join(project_root, args.file)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        print("Please run: python3 src/data_acquisition/arxiv_dataset.py --full")
        sys.exit(1)
    
    if not os.path.isfile(file_path):
        print(f"Error: Path is not a file: {file_path}")
        print("Please provide a valid CSV file path")
        sys.exit(1)
    
    analyze_full_dataset_chunked(
        file_path=file_path,
        chunk_size=args.chunk_size,
        n_workers=args.workers,
        use_parallel=args.parallel
    )


if __name__ == "__main__":
    main()
