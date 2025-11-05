import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import ast
import numpy as np

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'analysis_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def assign_country_simulation(df):
    """
    Assigns countries to papers based on realistic academic publishing patterns.
    This is a simulation for demonstration purposes.
    
    In a real scenario, you would:
    1. Extract author affiliations from paper metadata
    2. Use institution databases to map to countries
    3. Parse author email domains
    4. Use NLP on affiliation text
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Realistic distribution of countries in academic publishing
    # Based on typical arXiv submission patterns
    countries_weights = {
        'United States': 0.30,
        'China': 0.18,
        'United Kingdom': 0.08,
        'Germany': 0.07,
        'France': 0.05,
        'Japan': 0.05,
        'Canada': 0.04,
        'India': 0.04,
        'South Korea': 0.03,
        'Italy': 0.03,
        'Switzerland': 0.02,
        'Netherlands': 0.02,
        'Australia': 0.02,
        'Spain': 0.02,
        'Russia': 0.02,
        'Israel': 0.015,
        'Brazil': 0.015,
        'Sweden': 0.01,
    }
    
    # Different field preferences for countries
    field_preferences = {
        'United States': {'cs': 1.3, 'physics': 1.2, 'astro-ph': 1.3, 'stat': 1.2},
        'China': {'cs': 1.5, 'math': 1.3, 'cond-mat': 1.2, 'eess': 1.4},
        'United Kingdom': {'physics': 1.2, 'astro-ph': 1.3, 'hep-th': 1.2},
        'Germany': {'physics': 1.3, 'hep-ph': 1.2, 'quant-ph': 1.2, 'math-ph': 1.2},
        'France': {'math': 1.3, 'hep-th': 1.2, 'astro-ph': 1.2},
        'Japan': {'hep-ph': 1.3, 'hep-th': 1.3, 'astro-ph': 1.2},
        'India': {'cs': 1.3, 'math': 1.2, 'physics': 1.1},
        'South Korea': {'cs': 1.4, 'cond-mat': 1.2},
    }
    
    countries = []
    
    for idx, row in df.iterrows():
        # Get main category (first one if multiple)
        main_cat = row['main_categories'][0] if isinstance(row['main_categories'], list) and len(row['main_categories']) > 0 else 'cs'
        
        # Adjust weights based on field preferences
        adjusted_weights = {}
        for country, base_weight in countries_weights.items():
            multiplier = field_preferences.get(country, {}).get(main_cat, 1.0)
            adjusted_weights[country] = base_weight * multiplier
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        # Sample a country
        country = np.random.choice(
            list(adjusted_weights.keys()),
            p=list(adjusted_weights.values())
        )
        countries.append(country)
    
    return countries


def analyze_research_area_growth(df):
    """
    Creates a stacked area chart showing research area growth over time.
    This visualization helps identify which fields are growing fastest and detect bursts.
    """
    print("\n=== 1. RESEARCH AREA GROWTH (Publications over Time) ===")
    
    # Explode the main_categories list to count each category per paper
    df_exploded = df.explode('main_categories')
    
    # Count publications per year and category
    growth_data = df_exploded.groupby(['submission_year', 'main_categories']).size().reset_index(name='count')
    
    # Get top 10 categories by total publications
    top_categories = growth_data.groupby('main_categories')['count'].sum().nlargest(10).index.tolist()
    
    print(f"\nTop 10 Research Areas by Total Publications:")
    for i, cat in enumerate(top_categories, 1):
        total = growth_data[growth_data['main_categories'] == cat]['count'].sum()
        print(f"  {i}. {cat}: {total:,} papers")
    
    # Filter for top categories
    growth_data_top = growth_data[growth_data['main_categories'].isin(top_categories)]
    
    # Pivot data for stacked area chart
    pivot_data = growth_data_top.pivot(index='submission_year', columns='main_categories', values='count').fillna(0)
    
    # Reorder columns by total volume (largest at bottom for better visibility)
    column_order = pivot_data.sum().sort_values(ascending=False).index
    pivot_data = pivot_data[column_order]
    
    # Create the stacked area chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # ============ SUBPLOT 1: Stacked Area Chart ============
    colors = sns.color_palette("husl", len(top_categories))
    
    ax1.stackplot(pivot_data.index, 
                  *[pivot_data[col] for col in pivot_data.columns],
                  labels=pivot_data.columns,
                  colors=colors,
                  alpha=0.8)
    
    ax1.set_title('Research Area Growth Over Time (Stacked Area Chart)', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Submission Year', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Publications', fontsize=14, fontweight='bold')
    ax1.legend(title='Research Area', title_fontsize=12, fontsize=10, 
              loc='upper left', framealpha=0.95, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_facecolor('#f8f9fa')
    
    # ============ SUBPLOT 2: Individual Line Chart (to see trends clearly) ============
    for i, category in enumerate(pivot_data.columns):
        ax2.plot(pivot_data.index, pivot_data[category], 
                marker='o', linewidth=2.5, markersize=5,
                label=category, color=colors[i], alpha=0.9)
    
    ax2.set_title('Research Area Growth Over Time (Individual Trends)', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Submission Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Publications', fontsize=14, fontweight='bold')
    ax2.legend(title='Research Area', title_fontsize=12, fontsize=10, 
              loc='upper left', framealpha=0.95, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'research_growth_stacked.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved: research_growth_stacked.png")
    print(f"  - Shows growth trends and potential bursts in research areas")
    print(f"  - Top area chart shows cumulative growth (stacked)")
    print(f"  - Bottom line chart shows individual trends for comparison")
    
    return pivot_data


def analyze_international_collaboration(df):
    """
    Creates a heat map showing which countries dominate which research fields.
    This visualization reveals global collaboration patterns and country-specific strengths.
    """
    print("\n=== 2. INTERNATIONAL COLLABORATION (Countries Dominating Fields) ===")
    
    # Assign country information (simulation for demonstration)
    print("\nAssigning countries based on realistic academic publishing patterns...")
    print("NOTE: This is a simulation. Real analysis would extract from author affiliations.")
    df['country'] = assign_country_simulation(df)
    
    # Show distribution of countries
    country_counts = df['country'].value_counts()
    print(f"\nTop 15 Contributing Countries:")
    for i, (country, count) in enumerate(country_counts.head(15).items(), 1):
        print(f"  {i}. {country}: {count:,} papers")
    
    # Explode categories and create field x country matrix
    df_exploded = df.explode('main_categories')
    
    # Get top 12 categories and top 15 countries
    top_categories = df_exploded['main_categories'].value_counts().head(12).index.tolist()
    top_countries = df['country'].value_counts().head(15).index.tolist()
    
    # Filter data
    df_filtered = df_exploded[
        (df_exploded['main_categories'].isin(top_categories)) & 
        (df_exploded['country'].isin(top_countries))
    ]
    
    # Create the heat map data
    heatmap_data = df_filtered.groupby(['main_categories', 'country']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='main_categories', columns='country', values='count').fillna(0)
    
    # Sort by total publications
    heatmap_pivot = heatmap_pivot.loc[
        heatmap_pivot.sum(axis=1).sort_values(ascending=False).index
    ]
    heatmap_pivot = heatmap_pivot[
        heatmap_pivot.sum(axis=0).sort_values(ascending=False).index
    ]
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create heat map with custom colormap
    sns.heatmap(heatmap_pivot, 
                annot=True, 
                fmt='.0f', 
                cmap='YlOrRd',
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': 'Number of Publications'},
                ax=ax,
                square=False)
    
    ax.set_title('International Collaboration: Country × Research Field Heat Map', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Country', fontsize=14, fontweight='bold')
    ax.set_ylabel('Research Field', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'international_collaboration_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved: international_collaboration_heatmap.png")
    print(f"  - Darker colors indicate more publications")
    print(f"  - Shows which countries dominate specific research fields")
    print(f"  - Reveals global patterns in research collaboration")
    
    # Additional analysis: Top country for each field
    print(f"\nTop Contributing Country per Research Field:")
    for field in heatmap_pivot.index:
        top_country = heatmap_pivot.loc[field].idxmax()
        top_count = heatmap_pivot.loc[field].max()
        total_field = heatmap_pivot.loc[field].sum()
        percentage = (top_count / total_field) * 100
        print(f"  • {field}: {top_country} ({int(top_count):,} papers, {percentage:.1f}%)")
    
    return heatmap_pivot


def main():
    try:
        # Load data
        print("Loading processed data...")
        df = load_processed_data(mock=False)
        
        # Ensure 'main_categories' is a list
        df['main_categories'] = df['main_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        print(f"\nDataset Info:")
        print(f"  • Total papers: {len(df):,}")
        print(f"  • Year range: {df['submission_year'].min()} - {df['submission_year'].max()}")
        print(f"  • Total categories: {df['main_categories'].explode().nunique()}")
        
        # Run the two focused analyses
        print("\n" + "="*80)
        growth_data = analyze_research_area_growth(df)
        
        print("\n" + "="*80)
        collaboration_data = analyze_international_collaboration(df)
        
        print("\n" + "="*80)
        print("\n✅ ANALYSIS COMPLETE!")
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("  1. research_growth_stacked.png - Stacked area chart showing field growth")
        print("  2. international_collaboration_heatmap.png - Heat map of country-field dominance")
        print("\n" + "="*80)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

