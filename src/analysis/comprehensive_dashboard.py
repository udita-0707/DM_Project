#!/usr/bin/env python3
"""
Comprehensive Analysis Dashboard Generator

Creates a comprehensive dashboard visualization showing:
- Dataset overview statistics
- Missing values analysis
- Key distributions (year, categories, etc.)
- Data quality metrics

Designed to work with the full 2.8M paper dataset using chunked processing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import ast
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Output directory
OUTPUT_DIR = os.path.join(project_root, 'data', 'processed', 'analysis_results', 'comprehensive_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


def analyze_dataset_chunked(file_path: str, chunk_size: int = 100000):
    """
    Analyze full dataset in chunks to collect statistics.
    
    Args:
        file_path: Path to processed CSV file
        chunk_size: Number of rows per chunk
    
    Returns:
        Dictionary with aggregated statistics
    """
    print("="*80)
    print("COMPREHENSIVE DATASET ANALYSIS DASHBOARD")
    print("="*80)
    print(f"Analyzing: {file_path}")
    print(f"Chunk size: {chunk_size:,} rows")
    print()
    
    # Initialize aggregators
    total_rows = 0
    missing_counts = {}
    year_counts = {}
    category_counts = {}
    author_count_list = []
    title_lengths = []
    abstract_lengths = []
    discipline_counts = []
    
    # Process chunks
    print("Processing dataset in chunks...")
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    
    for chunk_idx, df_chunk in enumerate(chunk_iter, 1):
        total_rows += len(df_chunk)
        
        # Missing values
        for col in df_chunk.columns:
            if col not in missing_counts:
                missing_counts[col] = 0
            missing_counts[col] += df_chunk[col].isna().sum()
        
        # Year distribution
        if 'submission_year' in df_chunk.columns:
            year_data = df_chunk['submission_year'].value_counts().to_dict()
            for year, count in year_data.items():
                if pd.notna(year):
                    year_counts[int(year)] = year_counts.get(int(year), 0) + count
        
        # Category distribution
        if 'main_categories' in df_chunk.columns:
            try:
                df_chunk_copy = df_chunk.copy()
                df_chunk_copy['main_categories'] = df_chunk_copy['main_categories'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x
                )
                df_exploded = df_chunk_copy.explode('main_categories')
                cat_data = df_exploded['main_categories'].value_counts().to_dict()
                for cat, count in cat_data.items():
                    if pd.notna(cat):
                        category_counts[cat] = category_counts.get(cat, 0) + count
                
                # Discipline counts
                discipline_count_data = df_chunk_copy['main_categories'].apply(len).tolist()
                discipline_counts.extend(discipline_count_data)
            except Exception as e:
                pass
        
        # Author counts
        if 'authors' in df_chunk.columns:
            try:
                author_counts = df_chunk['authors'].apply(
                    lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[')
                    else len(x.split(',')) if isinstance(x, str) else 0
                )
                author_count_list.extend(author_counts.tolist())
            except Exception as e:
                pass
        
        # Text lengths
        if 'title' in df_chunk.columns:
            title_lengths.extend(df_chunk['title'].str.len().fillna(0).tolist())
        if 'abstract' in df_chunk.columns:
            abstract_lengths.extend(df_chunk['abstract'].str.len().fillna(0).tolist())
        
        if chunk_idx % 10 == 0:
            print(f"  Processed {chunk_idx} chunks ({total_rows:,} rows)...")
    
    print(f"\n✓ Processed {total_rows:,} total rows")
    
    # Compile results
    results = {
        'total_rows': total_rows,
        'missing_counts': missing_counts,
        'year_counts': year_counts,
        'category_counts': category_counts,
        'author_count_list': author_count_list,
        'title_lengths': title_lengths,
        'abstract_lengths': abstract_lengths,
        'discipline_counts': discipline_counts
    }
    
    return results


def create_comprehensive_dashboard(results: dict):
    """
    Create comprehensive dashboard visualization.
    
    Args:
        results: Dictionary with analysis results
    """
    print("\nCreating comprehensive dashboard visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, figure=fig)
    
    # Title
    fig.suptitle('Comprehensive Analysis Dashboard: Full arXiv Dataset (2.8M+ Papers)', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # ========== ROW 1: Dataset Overview ==========
    
    # 1. Dataset Statistics (Text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    stats_text = "DATASET OVERVIEW\n\n"
    stats_text += f"Total Papers: {results['total_rows']:,}\n"
    stats_text += f"Year Range: {min(results['year_counts'].keys()) if results['year_counts'] else 'N/A'} - {max(results['year_counts'].keys()) if results['year_counts'] else 'N/A'}\n"
    stats_text += f"Categories: {len(results['category_counts'])}\n"
    
    if results['author_count_list']:
        stats_text += f"\nAuthor Statistics:\n"
        stats_text += f"  Mean: {np.mean(results['author_count_list']):.1f}\n"
        stats_text += f"  Median: {np.median(results['author_count_list']):.0f}\n"
        stats_text += f"  Max: {np.max(results['author_count_list'])}\n"
    
    if results['title_lengths']:
        stats_text += f"\nTitle Length:\n"
        stats_text += f"  Mean: {np.mean(results['title_lengths']):.0f} chars\n"
        stats_text += f"  Median: {np.median(results['title_lengths']):.0f} chars\n"
    
    if results['abstract_lengths']:
        stats_text += f"\nAbstract Length:\n"
        stats_text += f"  Mean: {np.mean(results['abstract_lengths']):.0f} chars\n"
        stats_text += f"  Median: {np.median(results['abstract_lengths']):.0f} chars\n"
    
    ax1.text(0.1, 0.5, stats_text, fontsize=11, fontweight='bold',
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 2. Missing Values Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    
    missing_data = {k: v for k, v in results['missing_counts'].items() 
                   if v > 0 and k not in ['citation_count', 'citation_data_fetched']}
    if missing_data:
        missing_pct = {k: (v / results['total_rows']) * 100 for k, v in missing_data.items()}
        sorted_missing = sorted(missing_pct.items(), key=lambda x: x[1], reverse=True)[:10]
        
        cols = [x[0] for x in sorted_missing]
        vals = [x[1] for x in sorted_missing]
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(cols)))
        bars = ax2.barh(range(len(cols)), vals, color=colors)
        ax2.set_yticks(range(len(cols)))
        ax2.set_yticklabels(cols, fontsize=9)
        ax2.set_xlabel('Missing Values (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Missing Values by Column (Top 10)', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax2.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No Missing Values\nin Key Columns', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                transform=ax2.transAxes)
        ax2.axis('off')
    
    # 3. Year Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    
    if results['year_counts']:
        years = sorted(results['year_counts'].keys())
        counts = [results['year_counts'][y] for y in years]
        
        ax3.plot(years, counts, linewidth=2.5, marker='o', markersize=4, color='#2E86AB')
        ax3.fill_between(years, counts, alpha=0.3, color='#2E86AB')
        ax3.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Papers', fontsize=11, fontweight='bold')
        ax3.set_title('Publication Distribution Over Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='plain', axis='y')
    
    # ========== ROW 2: Category Analysis ==========
    
    # 4. Top Categories Bar Chart
    ax4 = fig.add_subplot(gs[1, 0])
    
    if results['category_counts']:
        top_cats = sorted(results['category_counts'].items(), key=lambda x: x[1], reverse=True)[:15]
        cats = [x[0] for x in top_cats]
        counts = [x[1] for x in top_cats]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(cats)))
        bars = ax4.barh(range(len(cats)), counts, color=colors)
        ax4.set_yticks(range(len(cats)))
        ax4.set_yticklabels(cats, fontsize=9)
        ax4.set_xlabel('Number of Papers', fontsize=11, fontweight='bold')
        ax4.set_title('Top 15 Research Categories', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax4.text(count + max(counts)*0.01, i, f'{count:,}', va='center', fontsize=8)
    
    # 5. Discipline Count Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    
    if results['discipline_counts']:
        disc_counts = Counter(results['discipline_counts'])
        disc_items = sorted(disc_counts.items())
        discs = [str(d[0]) for d in disc_items]
        counts = [d[1] for d in disc_items]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(discs)))
        bars = ax5.bar(discs, counts, color=colors)
        ax5.set_xlabel('Number of Disciplines per Paper', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Number of Papers', fontsize=11, fontweight='bold')
        ax5.set_title('Interdisciplinarity Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            if count > max(counts) * 0.05:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    # 6. Author Count Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    
    if results['author_count_list']:
        # Limit to reasonable range for visualization
        author_counts_filtered = [x for x in results['author_count_list'] if x <= 20]
        ax6.hist(author_counts_filtered, bins=20, color='#A23B72', edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Number of Authors', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Number of Papers', fontsize=11, fontweight='bold')
        ax6.set_title('Author Count Distribution (≤20 authors)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.axvline(np.mean(results['author_count_list']), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(results['author_count_list']):.1f}')
        ax6.legend()
    
    # ========== ROW 3: Text Analysis ==========
    
    # 7. Title Length Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    
    if results['title_lengths']:
        # Filter outliers for better visualization
        title_lengths_filtered = [x for x in results['title_lengths'] if x <= 500]
        ax7.hist(title_lengths_filtered, bins=50, color='#06A77D', edgecolor='black', alpha=0.7)
        ax7.set_xlabel('Title Length (characters)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Number of Papers', fontsize=11, fontweight='bold')
        ax7.set_title('Title Length Distribution', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.axvline(np.mean(results['title_lengths']), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(results['title_lengths']):.0f}')
        ax7.legend()
    
    # 8. Abstract Length Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    
    if results['abstract_lengths']:
        # Filter outliers for better visualization
        abstract_lengths_filtered = [x for x in results['abstract_lengths'] if x <= 5000]
        ax8.hist(abstract_lengths_filtered, bins=50, color='#F18F01', edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Abstract Length (characters)', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Number of Papers', fontsize=11, fontweight='bold')
        ax8.set_title('Abstract Length Distribution', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.axvline(np.mean(results['abstract_lengths']), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(results['abstract_lengths']):.0f}')
        ax8.legend()
    
    # 9. Data Quality Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    quality_text = "DATA QUALITY METRICS\n\n"
    
    # Calculate completeness
    key_columns = ['title', 'abstract', 'authors', 'categories', 'submission_year']
    completeness = {}
    for col in key_columns:
        if col in results['missing_counts']:
            missing = results['missing_counts'][col]
            total = results['total_rows']
            completeness[col] = ((total - missing) / total) * 100
    
    for col, pct in completeness.items():
        quality_text += f"{col.capitalize()}: {pct:.1f}% complete\n"
    
    quality_text += f"\nTotal Records: {results['total_rows']:,}\n"
    quality_text += f"Categories Found: {len(results['category_counts'])}\n"
    
    if results['year_counts']:
        quality_text += f"Year Span: {max(results['year_counts'].keys()) - min(results['year_counts'].keys()) + 1} years\n"
    
    quality_text += f"\nInterdisciplinary Papers:\n"
    if results['discipline_counts']:
        multi_disc = sum(1 for x in results['discipline_counts'] if x > 1)
        pct_multi = (multi_disc / len(results['discipline_counts'])) * 100
        quality_text += f"  {pct_multi:.1f}% have 2+ categories\n"
    
    ax9.text(0.1, 0.5, quality_text, fontsize=11, fontweight='bold',
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ========== ROW 4: Additional Statistics ==========
    
    # 10. Category Percentage Bar Chart (Top 10) - Clearer than pie chart
    ax10 = fig.add_subplot(gs[3, 0])
    
    if results['category_counts']:
        top_cats = sorted(results['category_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        cats = [x[0] for x in top_cats]
        counts = [x[1] for x in top_cats]
        total_cat_papers = sum(results['category_counts'].values())
        percentages = [(c / total_cat_papers) * 100 for c in counts]
        
        # Use horizontal bar chart for better readability
        colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
        bars = ax10.barh(range(len(cats)), percentages, color=colors, edgecolor='black', linewidth=1.2)
        ax10.set_yticks(range(len(cats)))
        ax10.set_yticklabels(cats, fontsize=10, fontweight='bold')
        ax10.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax10.set_title('Top 10 Categories by Percentage', fontsize=12, fontweight='bold', pad=15)
        ax10.invert_yaxis()  # Highest at top
        ax10.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add percentage labels on bars
        for i, (bar, pct, count) in enumerate(zip(bars, percentages, counts)):
            # Label with both percentage and count for clarity
            label_text = f'{pct:.1f}%\n({count:,})'
            ax10.text(pct + 0.5, i, label_text, va='center', fontsize=9, fontweight='bold')
        
        # Set x-axis limit to accommodate labels
        ax10.set_xlim(0, max(percentages) * 1.15)
    
    # 11. Year-over-Year Growth
    ax11 = fig.add_subplot(gs[3, 1])
    
    if results['year_counts'] and len(results['year_counts']) > 1:
        years = sorted(results['year_counts'].keys())
        counts = [results['year_counts'][y] for y in years]
        
        # Calculate year-over-year growth
        yoy_growth = []
        for i in range(1, len(counts)):
            if counts[i-1] > 0:
                growth = ((counts[i] - counts[i-1]) / counts[i-1]) * 100
                yoy_growth.append(growth)
            else:
                yoy_growth.append(0)
        
        growth_years = years[1:]
        colors_growth = ['green' if x > 0 else 'red' for x in yoy_growth]
        bars = ax11.bar(range(len(growth_years)), yoy_growth, color=colors_growth, alpha=0.7)
        ax11.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax11.set_ylabel('Year-over-Year Growth (%)', fontsize=11, fontweight='bold')
        ax11.set_title('Publication Growth Rate Over Time', fontsize=12, fontweight='bold')
        ax11.set_xticks(range(0, len(growth_years), max(1, len(growth_years)//10)))
        ax11.set_xticklabels([growth_years[i] for i in range(0, len(growth_years), max(1, len(growth_years)//10))],
                             rotation=45, ha='right')
        ax11.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax11.grid(True, alpha=0.3, axis='y')
    
    # 12. Summary Statistics Table
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.axis('off')
    
    summary_text = "SUMMARY STATISTICS\n\n"
    summary_text += f"Dataset Size: {results['total_rows']:,} papers\n"
    summary_text += f"Time Period: {min(results['year_counts'].keys()) if results['year_counts'] else 'N/A'}-{max(results['year_counts'].keys()) if results['year_counts'] else 'N/A'}\n"
    summary_text += f"Research Fields: {len(results['category_counts'])}\n\n"
    
    if results['author_count_list']:
        summary_text += f"Avg Authors/Paper: {np.mean(results['author_count_list']):.1f}\n"
    
    if results['discipline_counts']:
        avg_disc = np.mean(results['discipline_counts'])
        summary_text += f"Avg Disciplines/Paper: {avg_disc:.1f}\n"
        single_disc = sum(1 for x in results['discipline_counts'] if x == 1)
        pct_single = (single_disc / len(results['discipline_counts'])) * 100
        summary_text += f"Single-Discipline: {pct_single:.1f}%\n"
        summary_text += f"Interdisciplinary: {100-pct_single:.1f}%\n"
    
    if results['year_counts']:
        peak_year = max(results['year_counts'], key=results['year_counts'].get)
        summary_text += f"\nPeak Publication Year: {peak_year}\n"
        summary_text += f"  ({results['year_counts'][peak_year]:,} papers)\n"
    
    ax12.text(0.1, 0.5, summary_text, fontsize=11, fontweight='bold',
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'comprehensive_analysis_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Dashboard saved to: {output_path}")
    
    return output_path


def main():
    """Main function to generate comprehensive dashboard."""
    # Path to processed data
    processed_file = os.path.join(project_root, 'data', 'processed', 'arxiv_processed.csv')
    
    if not os.path.exists(processed_file):
        print(f"Error: Processed data file not found at {processed_file}")
        print("Please run the data processing script first:")
        print("  python3 src/data_acquisition/arxiv_dataset.py --full")
        return
    
    # Analyze dataset
    results = analyze_dataset_chunked(processed_file, chunk_size=100000)
    
    # Create dashboard
    dashboard_path = create_comprehensive_dashboard(results)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DASHBOARD GENERATION COMPLETE!")
    print("="*80)
    print(f"\nDashboard saved to: {dashboard_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total Papers: {results['total_rows']:,}")
    print(f"  Categories: {len(results['category_counts'])}")
    print(f"  Year Range: {min(results['year_counts'].keys()) if results['year_counts'] else 'N/A'}-{max(results['year_counts'].keys()) if results['year_counts'] else 'N/A'}")


if __name__ == "__main__":
    main()
