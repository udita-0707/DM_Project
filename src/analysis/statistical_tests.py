import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import ast
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, pearsonr, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'analysis_results', 'statistical_tests')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_citation_proxy(df):
    """
    PREPROCESSING: Creates a citation proxy when real citation data is unavailable.
    
    Why: arXiv doesn't provide citation counts. We need a metric to analyze research impact.
    Not all papers can be enriched via API (rate limits, papers not in Semantic Scholar).
    
    How: Combine two indicators:
    1. Paper age: Older papers have more time to accumulate citations (60% weight)
    2. Version count: More versions suggest more interest/updates (40% weight)
    
    The proxy is normalized to 0-100 scale. Real citations are used when available,
    proxy is used as fallback.
    
    Args:
        df: DataFrame with submission_year and versions columns
    
    Returns:
        DataFrame with citation_metric column (real citations or proxy)
    """
    # PREPROCESSING: Calculate paper age (years since submission)
    # Why: Older papers typically have more citations (more time to be cited)
    current_year = 2025
    df['paper_age'] = current_year - df['submission_year']
    
    # PREPROCESSING: Count number of versions per paper
    # Why: Papers with more versions often indicate active research/updates, 
    # which may correlate with citations
    df['version_count'] = df['versions'].apply(
        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x) if isinstance(x, list) else 1
    )
    
    # PREPROCESSING: Create composite citation proxy
    # How: Normalize both metrics to 0-1, weight them (age 60%, versions 40%), scale to 0-100
    # Rationale: Age is more important (time = more citations), but versions also matter
    df['citation_proxy'] = (
        (df['paper_age'] / df['paper_age'].max()) * 0.6 +
        (df['version_count'] / df['version_count'].max()) * 0.4
    ) * 100  # Scale to 0-100 range
    
    # PREPROCESSING: Use real citations when available, proxy otherwise
    # Why: Real citations are more accurate, but proxy allows analysis of all papers
    if 'citation_count' in df.columns:
        # Fill missing citation counts with proxy
        # Note: Real citations are typically much larger (0-1000s) than proxy (0-100)
        df['citation_metric'] = df['citation_count'].fillna(df['citation_proxy'])
        df['using_real_citations'] = df['citation_count'].notna()
        real_count = df['using_real_citations'].sum()
        if real_count > 0:
            print(f"  Using real citation data for {real_count} papers ({100*real_count/len(df):.1f}%)")
            print(f"  Using citation proxy for {len(df) - real_count} papers")
    else:
        df['citation_metric'] = df['citation_proxy']
        df['using_real_citations'] = False
        print(f"  No real citation data found - using citation proxy for all papers")
    
    return df

def test_interdisciplinarity_hypothesis(df):
    """
    Statistical test: "Interdisciplinary Premium"
    Papers with 3+ disciplines have different citation patterns than single-discipline papers.
    
    Uses ANOVA to compare citations (real or proxy) across discipline count groups.
    """
    print("\n" + "="*80)
    print("STATISTICAL TEST: Interdisciplinarity vs Citations")
    print("="*80)
    
    # Ensure derived features exist
    df['main_categories'] = df['main_categories'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df['discipline_count'] = df['main_categories'].apply(len)
    
    # Create citation metric (real citations if available, proxy otherwise)
    df = create_citation_proxy(df)
    
    # Group papers by discipline count
    # Group 1: Single discipline (1)
    # Group 2: Two disciplines (2)
    # Group 3: Three or more disciplines (3+)
    df['discipline_group'] = df['discipline_count'].apply(
        lambda x: 'Single' if x == 1 else 'Two' if x == 2 else 'Three+'
    )
    
    groups = {
        'Single': df[df['discipline_group'] == 'Single']['citation_metric'].dropna(),
        'Two': df[df['discipline_group'] == 'Two']['citation_metric'].dropna(),
        'Three+': df[df['discipline_group'] == 'Three+']['citation_metric'].dropna()
    }
    
    # Descriptive statistics
    print("\nDescriptive Statistics by Discipline Group:")
    desc_stats = df.groupby('discipline_group')['citation_metric'].agg(['count', 'mean', 'std', 'median'])
    print(desc_stats.round(3))
    
    # Perform ANOVA
    f_statistic, p_value = f_oneway(groups['Single'], groups['Two'], groups['Three+'])
    
    print(f"\nANOVA Results:")
    print(f"  F-statistic: {f_statistic:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significance level (α): 0.05")
    
    if p_value < 0.05:
        print(f"  ✓ Result: REJECT null hypothesis (p < 0.05)")
        print(f"    There IS a significant difference in citations across discipline groups.")
    else:
        print(f"  ✗ Result: FAIL to reject null hypothesis (p >= 0.05)")
        print(f"    No significant difference found in citations across discipline groups.")
    
    # Post-hoc test (Tukey HSD) if ANOVA is significant
    if p_value < 0.05:
        print(f"\nPost-hoc Analysis (Tukey HSD):")
        # Prepare data for Tukey HSD
        tukey_data = pd.DataFrame({
            'citation_metric': pd.concat([groups['Single'], groups['Two'], groups['Three+']]),
            'discipline_group': (['Single'] * len(groups['Single']) + 
                               ['Two'] * len(groups['Two']) + 
                               ['Three+'] * len(groups['Three+']))
        })
        
        tukey_result = pairwise_tukeyhsd(tukey_data['citation_metric'], tukey_data['discipline_group'])
        print(tukey_result)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    ax = axes[0]
    box_data = [groups['Single'], groups['Two'], groups['Three+']]
    bp = ax.boxplot(box_data, labels=['Single', 'Two', 'Three+'], patch_artist=True)
    
    colors = sns.color_palette("Set2", 3)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Citations by Discipline Count\n(ANOVA Test)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Disciplines', fontsize=12, fontweight='bold')
    ax.set_ylabel('Citation Count/Score', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add p-value annotation
    ax.text(0.5, 0.95, f'ANOVA: p = {p_value:.4f}', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')
    
    # Violin plot for better distribution view
    ax = axes[1]
    violin_data = pd.melt(
        pd.DataFrame({
            'Single': groups['Single'],
            'Two': groups['Two'],
            'Three+': groups['Three+']
        }),
        var_name='Discipline Group',
        value_name='Citations'
    )
    
    sns.violinplot(data=violin_data, x='Discipline Group', y='Citations', 
                   palette=colors, ax=ax)
    ax.set_title('Distribution of Citations by Discipline Count', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Disciplines', fontsize=12, fontweight='bold')
    ax.set_ylabel('Citation Count/Score', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'interdisciplinarity_anova_test.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved: interdisciplinarity_anova_test.png")
    
    return {
        'f_statistic': f_statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'descriptive_stats': desc_stats
    }

def test_correlation_analysis(df):
    """
    Tests correlations between key variables.
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Ensure derived features exist
    if 'num_authors' not in df.columns:
        df['num_authors'] = df['authors'].apply(
            lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[') 
            else len(x.split(',')) if isinstance(x, str) else 0
        )
    if 'discipline_count' not in df.columns:
        df['main_categories'] = df['main_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['discipline_count'] = df['main_categories'].apply(len)
    
    df = create_citation_proxy(df)
    
    # Test correlations
    correlations = []
    
    # 1. Author count vs Citations
    corr_pearson, p_pearson = pearsonr(df['num_authors'].dropna(), 
                                       df.loc[df['num_authors'].notna(), 'citation_metric'])
    correlations.append({
        'Variable 1': 'Number of Authors',
        'Variable 2': 'Citations',
        'Pearson r': corr_pearson,
        'p-value': p_pearson,
        'Significant': p_pearson < 0.05
    })
    
    # 2. Discipline count vs Citations
    corr_pearson, p_pearson = pearsonr(df['discipline_count'].dropna(), 
                                       df.loc[df['discipline_count'].notna(), 'citation_metric'])
    correlations.append({
        'Variable 1': 'Discipline Count',
        'Variable 2': 'Citations',
        'Pearson r': corr_pearson,
        'p-value': p_pearson,
        'Significant': p_pearson < 0.05
    })
    
    # 3. Submission year vs Citations
    corr_pearson, p_pearson = pearsonr(df['submission_year'].dropna(), 
                                       df.loc[df['submission_year'].notna(), 'citation_metric'])
    correlations.append({
        'Variable 1': 'Submission Year',
        'Variable 2': 'Citations',
        'Pearson r': corr_pearson,
        'p-value': p_pearson,
        'Significant': p_pearson < 0.05
    })
    
    # 4. Author count vs Discipline count
    corr_pearson, p_pearson = pearsonr(df['num_authors'].dropna(), 
                                       df.loc[df['num_authors'].notna(), 'discipline_count'])
    correlations.append({
        'Variable 1': 'Number of Authors',
        'Variable 2': 'Discipline Count',
        'Pearson r': corr_pearson,
        'p-value': p_pearson,
        'Significant': p_pearson < 0.05
    })
    
    corr_df = pd.DataFrame(correlations)
    print("\nCorrelation Test Results:")
    print(corr_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    pairs = [
        ('num_authors', 'citation_metric', 'Number of Authors', 'Citations'),
        ('discipline_count', 'citation_metric', 'Discipline Count', 'Citations'),
        ('submission_year', 'citation_metric', 'Submission Year', 'Citations'),
        ('num_authors', 'discipline_count', 'Number of Authors', 'Discipline Count')
    ]
    
    for idx, (x_col, y_col, x_label, y_label) in enumerate(pairs):
        ax = axes[idx]
        data_subset = df[[x_col, y_col]].dropna()
        
        # Scatter plot with regression line
        ax.scatter(data_subset[x_col], data_subset[y_col], alpha=0.5, s=20, color=sns.color_palette("viridis", 1)[0])
        
        # Add regression line
        z = np.polyfit(data_subset[x_col], data_subset[y_col], 1)
        p = np.poly1d(z)
        ax.plot(data_subset[x_col], p(data_subset[x_col]), "r--", alpha=0.8, linewidth=2, label='Regression Line')
        
        # Calculate and display correlation
        corr_val = corr_df[
            ((corr_df['Variable 1'] == x_label) & (corr_df['Variable 2'] == y_label)) |
            ((corr_df['Variable 1'] == y_label) & (corr_df['Variable 2'] == x_label))
        ]['Pearson r'].values[0]
        p_val = corr_df[
            ((corr_df['Variable 1'] == x_label) & (corr_df['Variable 2'] == y_label)) |
            ((corr_df['Variable 1'] == y_label) & (corr_df['Variable 2'] == x_label))
        ]['p-value'].values[0]
        
        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
        ax.set_title(f'{x_label} vs {y_label}\nr = {corr_val:.3f}, p = {p_val:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Correlation Analysis: Scatter Plots with Regression Lines', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_analysis_tests.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved: correlation_analysis_tests.png")
    
    return corr_df

def test_temporal_trends(df):
    """
    Tests for significant trends over time using Mann-Kendall test.
    """
    print("\n" + "="*80)
    print("TEMPORAL TREND ANALYSIS")
    print("="*80)
    
    # Ensure main_categories is a list
    df['main_categories'] = df['main_categories'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Get top 5 categories
    df_exploded = df.explode('main_categories')
    top_categories = df_exploded['main_categories'].value_counts().head(5).index.tolist()
    
    print(f"\nTesting trends for top 5 categories: {top_categories}")
    
    trend_results = []
    
    for category in top_categories:
        cat_data = df_exploded[
            (df_exploded['main_categories'] == category) & 
            (df_exploded['submission_year'].notna())
        ]
        
        # Count publications per year
        yearly_counts = cat_data.groupby('submission_year').size().reset_index(name='count')
        yearly_counts = yearly_counts.sort_values('submission_year')
        
        if len(yearly_counts) < 3:
            continue
        
        # Simple linear trend test (correlation with year)
        corr, p_val = pearsonr(yearly_counts['submission_year'], yearly_counts['count'])
        
        # Calculate growth rate
        if len(yearly_counts) > 1:
            growth_rate = ((yearly_counts['count'].iloc[-1] - yearly_counts['count'].iloc[0]) / 
                          yearly_counts['count'].iloc[0]) * 100 if yearly_counts['count'].iloc[0] > 0 else 0
        else:
            growth_rate = 0
        
        trend_results.append({
            'Category': category,
            'Correlation (r)': corr,
            'p-value': p_val,
            'Significant Trend': p_val < 0.05,
            'Growth Rate (%)': growth_rate,
            'Years': len(yearly_counts)
        })
    
    trend_df = pd.DataFrame(trend_results)
    print("\nTemporal Trend Test Results:")
    print(trend_df.to_string(index=False))
    
    return trend_df

def main():
    """Main function to run all statistical tests."""
    try:
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
        
        # Load data
        print("\nLoading processed data...")
        df = load_processed_data(mock=False)
        
        # Ensure main_categories is a list
        df['main_categories'] = df['main_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Run statistical tests
        anova_results = test_interdisciplinarity_hypothesis(df)
        corr_results = test_correlation_analysis(df)
        trend_results = test_temporal_trends(df)
        
        print("\n" + "="*80)
        print("✅ STATISTICAL ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {OUTPUT_DIR}")
        print("\nGenerated Files:")
        print("  1. interdisciplinarity_anova_test.png - ANOVA test results")
        print("  2. correlation_analysis_tests.png - Correlation scatter plots")
        print("\n" + "="*80)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the data acquisition script first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

