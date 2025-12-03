#!/usr/bin/env python3
"""
Quick script to regenerate only the citation half-life analysis graph
with the improved visualization that removes the "Insufficient data" message.
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data
from src.analysis.comprehensive_analysis import analyze_citation_half_life

def main():
    print("="*80)
    print("REGENERATING CITATION HALF-LIFE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        df = load_processed_data(mock=False, use_enriched=True)
        print(f"Loaded {len(df):,} papers")
        print(f"Date range: {df['submission_year'].min()} - {df['submission_year'].max()}")
        
        if 'citation_data_fetched' in df.columns:
            enriched_count = df['citation_data_fetched'].sum()
            print(f"Papers with real citation data: {enriched_count:,} ({enriched_count/len(df)*100:.1f}%)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Run only citation half-life analysis
    print("\nRunning citation half-life analysis...")
    try:
        result = analyze_citation_half_life(df)
        print("\n" + "="*80)
        print("CITATION HALF-LIFE ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nGraph saved to: data/processed/analysis_results/comprehensive_analysis/citation_half_life_analysis.png")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
