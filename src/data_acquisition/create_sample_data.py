#!/usr/bin/env python3
"""
Sample Data Extraction Script
Extracts a representative sample from the full arXiv dataset for development and testing.

Usage:
    python3 create_sample_data.py --size 10000 --output data/raw/arxiv_sample_10k.json
"""

import json
import random
import argparse
import os
from datetime import datetime

def extract_sample(input_file, output_file, sample_size=10000, random_seed=42):
    """
    Extract a random sample from the large arXiv JSON file.
    
    Args:
        input_file: Path to the full arXiv JSON file
        output_file: Path to save the sample
        sample_size: Number of records to extract
        random_seed: Random seed for reproducibility
    """
    
    print(f"=" * 60)
    print("ArXiv Sample Data Extraction")
    print(f"=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Sample size: {sample_size}")
    print()
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Please download the arXiv dataset from Kaggle first.")
        return False
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Step 1: Count total lines (this may take a few minutes for large files)
    print("Step 1: Counting total records in dataset...")
    total_lines = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return False
    
    print(f"  Total records found: {total_lines:,}")
    print()
    
    if sample_size > total_lines:
        print(f"WARNING: Sample size ({sample_size}) is larger than total records ({total_lines})")
        print(f"Using all {total_lines} records instead.")
        sample_size = total_lines
    
    # Step 2: Generate random line indices to sample
    print(f"Step 2: Generating {sample_size} random indices...")
    sample_indices = set(random.sample(range(total_lines), sample_size))
    print(f"  Generated {len(sample_indices)} unique indices")
    print()
    
    # Step 3: Extract the sampled lines
    print("Step 3: Extracting sample records...")
    sampled_records = []
    skipped_records = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx in sample_indices:
                    try:
                        record = json.loads(line)
                        sampled_records.append(record)
                    except json.JSONDecodeError:
                        skipped_records += 1
                        print(f"  WARNING: Skipped malformed record at line {idx}")
                
                # Progress update every 100K records
                if (idx + 1) % 100000 == 0:
                    print(f"  Processed {idx + 1:,} records... (found {len(sampled_records)} samples so far)")
    
    except Exception as e:
        print(f"ERROR during extraction: {e}")
        return False
    
    print(f"  Extraction complete: {len(sampled_records)} records extracted")
    if skipped_records > 0:
        print(f"  Note: {skipped_records} malformed records were skipped")
    print()
    
    # Step 4: Save the sample
    print(f"Step 4: Saving sample to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in sampled_records:
                f.write(json.dumps(record) + '\n')
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Sample saved successfully!")
        print(f"  File size: {file_size_mb:.2f} MB")
        print()
    except Exception as e:
        print(f"ERROR saving file: {e}")
        return False
    
    # Step 5: Summary statistics
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total records in full dataset: {total_lines:,}")
    print(f"Sample size: {len(sampled_records):,}")
    print(f"Sampling rate: {(len(sampled_records) / total_lines * 100):.2f}%")
    print(f"Output file: {output_file}")
    print()
    print("Next steps:")
    print("1. Process the sample: python3 src/data_acquisition/arxiv_dataset.py")
    print("2. Run analysis: python3 src/analysis/exploratory_analysis.py")
    print()
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Extract a sample from the arXiv dataset for development and testing."
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/arxiv-metadata-oai-snapshot.json',
        help='Path to the full arXiv JSON file (default: data/raw/arxiv-metadata-oai-snapshot.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/arxiv_sample_10k.json',
        help='Path to save the sample file (default: data/raw/arxiv_sample_10k.json)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=10000,
        help='Number of records to sample (default: 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    success = extract_sample(
        input_file=args.input,
        output_file=args.output,
        sample_size=args.size,
        random_seed=args.seed
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())

