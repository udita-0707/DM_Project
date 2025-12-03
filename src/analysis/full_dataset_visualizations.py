#!/usr/bin/env python3
"""
Generate Visualizations from Full Dataset Analysis Results

This script creates comprehensive visualizations using the full 2.2M+ dataset
analysis results instead of the 10K sample.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import glob
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Output directory
FULL_DATASET_DIR = os.path.join(project_root, 'data', 'processed', 'analysis_results', 'full_dataset_analysis')
OUTPUT_DIR = os.path.join(project_root, 'data', 'processed', 'analysis_results', 'comprehensive_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


def load_latest_full_dataset_results():
    """Load the most recent full dataset analysis results."""
    results = {}
    
    # Find latest files
    category_files = glob.glob(os.path.join(FULL_DATASET_DIR, 'category_counts_full_*.csv'))
    year_files = glob.glob(os.path.join(FULL_DATASET_DIR, 'year_distribution_full_*.csv'))
    collab_files = glob.glob(os.path.join(FULL_DATASET_DIR, 'collaboration_stats_full_*.json'))
    inter_files = glob.glob(os.path.join(FULL_DATASET_DIR, 'interdisciplinarity_stats_full_*.json'))
    keyword_files = glob.glob(os.path.join(FULL_DATASET_DIR, 'top_keywords_full_*.csv'))
    
    if category_files:
        latest_cat = max(category_files, key=os.path.getmtime)
        results['categories'] = pd.read_csv(latest_cat)
        print(f"Loaded category counts from: {os.path.basename(latest_cat)}")
    
    if year_files:
        latest_year = max(year_files, key=os.path.getmtime)
        results['years'] = pd.read_csv(latest_year)
        print(f"Loaded year distribution from: {os.path.basename(latest_year)}")
    
    if collab_files:
        latest_collab = max(collab_files, key=os.path.getmtime)
        with open(latest_collab, 'r') as f:
            results['collaboration'] = json.load(f)
        print(f"Loaded collaboration stats from: {os.path.basename(latest_collab)}")
    
    if inter_files:
        latest_inter = max(inter_files, key=os.path.getmtime)
        with open(latest_inter, 'r') as f:
            results['interdisciplinarity'] = json.load(f)
        print(f"Loaded interdisciplinarity stats from: {os.path.basename(latest_inter)}")
    
    if keyword_files:
        latest_keyword = max(keyword_files, key=os.path.getmtime)
        results['keywords'] = pd.read_csv(latest_keyword)
        print(f"Loaded keywords from: {os.path.basename(latest_keyword)}")
    
    return results


def plot_research_growth_full_dataset(results):
    """Plot research growth trends from full dataset."""
    if 'years' not in results or 'categories' not in results:
        print("Missing data for research growth plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Year distribution
    ax1 = axes[0]
    year_df = results['years'].sort_values('year')
    ax1.plot(year_df['year'], year_df['count'], linewidth=2.5, marker='o', markersize=5, color='#2E86AB')
    ax1.fill_between(year_df['year'], year_df['count'], alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
    ax1.set_title('arXiv Publication Growth Over Time (Full Dataset: 2.2M+ Papers)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(year_df['year'].min(), year_df['year'].max())
    
    # Add total count annotation
    total = year_df['count'].sum()
    ax1.text(0.02, 0.98, f'Total Papers: {total:,}', 
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Top categories
    ax2 = axes[1]
    cat_df = results['categories'].head(15).sort_values('count', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(cat_df)))
    bars = ax2.barh(cat_df['category'], cat_df['count'], color=colors)
    ax2.set_xlabel('Number of Papers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Category', fontsize=12, fontweight='bold')
    ax2.set_title('Top 15 Research Categories (Full Dataset)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(cat_df.iterrows()):
        ax2.text(row['count'] + row['count']*0.01, i, f"{row['count']:,}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'research_growth_full_dataset.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    plt.close()


def plot_category_growth_trends(results):
    """Plot category growth trends over time (requires full processed data)."""
    # This would need the full processed CSV to calculate year-by-year category growth
    # For now, we'll create a simplified version
    if 'categories' not in results:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cat_df = results['categories'].head(10)
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_df)))
    
    bars = ax.bar(cat_df['category'], cat_df['count'], color=colors)
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Papers', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Research Categories - Full Dataset (2.2M+ Papers)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height):,}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'top_categories_full_dataset.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    plt.close()


def plot_collaboration_stats(results):
    """Plot collaboration statistics from full dataset."""
    if 'collaboration' not in results:
        return
    
    collab = results['collaboration']
    if 'author_count_stats' not in collab:
        return
    
    stats = collab['author_count_stats']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Author statistics
    ax1 = axes[0]
    metrics = ['Mean', 'Median', 'Max']
    values = [stats['mean'], stats['median'], stats['max']]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax1.bar(metrics, values, color=colors)
    ax1.set_ylabel('Number of Authors', fontsize=12, fontweight='bold')
    ax1.set_title('Author Count Statistics (Full Dataset)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}' if val < 1000 else f'{int(val):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Top categories by paper count
    ax2 = axes[1]
    if 'category_distribution' in collab:
        cat_dist = collab['category_distribution']
        top_cats = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        categories = [cat[0] for cat in top_cats]
        counts = [cat[1] for cat in top_cats]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        bars = ax2.barh(categories, counts, color=colors)
        ax2.set_xlabel('Number of Papers', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Category', fontsize=12, fontweight='bold')
        ax2.set_title('Top 10 Categories (Full Dataset)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (cat, count) in enumerate(top_cats):
            ax2.text(count + count*0.01, i, f"{count:,}", 
                    va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'collaboration_stats_full_dataset.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    plt.close()


def plot_interdisciplinarity_stats(results):
    """Plot interdisciplinarity statistics from full dataset."""
    if 'interdisciplinarity' not in results:
        return
    
    inter = results['interdisciplinarity']
    if 'discipline_distribution' not in inter:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Discipline distribution
    ax1 = axes[0]
    disc_dist = inter['discipline_distribution']
    # Convert to sorted list
    disc_items = sorted([(int(k), v) for k, v in disc_dist.items()])
    disciplines = [str(d[0]) for d in disc_items]
    counts = [d[1] for d in disc_items]
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(disciplines)))
    bars = ax1.bar(disciplines, counts, color=colors)
    ax1.set_xlabel('Number of Disciplines', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Papers by Discipline Count (Full Dataset)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top bars
    for bar, count in zip(bars, counts):
        if count > max(counts) * 0.1:  # Only label significant bars
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}',
                   ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Single vs Interdisciplinary
    ax2 = axes[1]
    single = inter.get('single_discipline_count', 0)
    interdisc = inter.get('interdisciplinary_count', 0)
    
    labels = ['Single-Discipline', 'Interdisciplinary']
    values = [single, interdisc]
    colors_pie = ['#2E86AB', '#A23B72']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax2.pie(values, labels=labels, colors=colors_pie, 
                                       autopct='%1.1f%%', explode=explode,
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Single vs Interdisciplinary Papers (Full Dataset)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add total count
    total = single + interdisc
    ax2.text(0, -1.3, f'Total: {total:,} papers', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'interdisciplinarity_stats_full_dataset.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    plt.close()


def plot_top_keywords(results):
    """Plot top keywords from full dataset."""
    if 'keywords' not in results:
        return
    
    keyword_df = results['keywords'].head(30)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(keyword_df)))
    bars = ax.barh(keyword_df['keyword'], keyword_df['count'], color=colors)
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Keyword', fontsize=12, fontweight='bold')
    ax.set_title('Top 30 Keywords - Full Dataset (2.2M+ Papers)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(keyword_df.iterrows()):
        ax.text(row['count'] + row['count']*0.01, i, f"{row['count']:,}", 
               va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'top_keywords_full_dataset.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    plt.close()


def main():
    """Generate all visualizations from full dataset results."""
    print("="*80)
    print("GENERATING VISUALIZATIONS FROM FULL DATASET ANALYSIS")
    print("="*80)
    print()
    
    # Load results
    print("Loading full dataset analysis results...")
    results = load_latest_full_dataset_results()
    
    if not results:
        print("Error: No full dataset analysis results found!")
        print(f"Please run: python3 src/analysis/batch_full_dataset_analysis.py")
        return
    
    print(f"\nGenerating visualizations...")
    
    # Generate plots
    plot_research_growth_full_dataset(results)
    plot_category_growth_trends(results)
    plot_collaboration_stats(results)
    plot_interdisciplinarity_stats(results)
    plot_top_keywords(results)
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - research_growth_full_dataset.png")
    print("  - top_categories_full_dataset.png")
    print("  - collaboration_stats_full_dataset.png")
    print("  - interdisciplinarity_stats_full_dataset.png")
    print("  - top_keywords_full_dataset.png")


if __name__ == "__main__":
    main()
