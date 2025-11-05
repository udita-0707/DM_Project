import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import ast
from collections import Counter

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'analysis_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_research_growth(df):
    """
    Analyzes the growth of research areas over time.
    """
    print("\n--- 1. Research Area Growth Analysis ---")
    
    # Explode the main_categories list to count each category per paper
    df_exploded = df.explode('main_categories')
    
    # Count publications per year and category
    growth_data = df_exploded.groupby(['submission_year', 'main_categories']).size().reset_index(name='count')
    
    # Calculate total publications per year for normalization (optional)
    total_per_year = growth_data.groupby('submission_year')['count'].sum().reset_index(name='total')
    growth_data = pd.merge(growth_data, total_per_year, on='submission_year')
    growth_data['proportion'] = growth_data['count'] / growth_data['total']
    
    print("Top 5 categories by total publications:")
    print(growth_data.groupby('main_categories')['count'].sum().nlargest(5))
    
    # Save the data for visualization
    growth_data.to_csv(os.path.join(OUTPUT_DIR, 'research_growth_data.csv'), index=False)
    
    # Get top 7 categories by total publications
    top_categories = growth_data.groupby('main_categories')['count'].sum().nlargest(7).index.tolist()
    growth_data_top = growth_data[growth_data['main_categories'].isin(top_categories)]
    
    # Create a clear, professional visualization - Top 7 Categories
    plt.figure(figsize=(14, 8))
    
    # Use a better color palette
    colors = sns.color_palette("husl", len(top_categories))
    
    for i, category in enumerate(top_categories):
        cat_data = growth_data_top[growth_data_top['main_categories'] == category]
        plt.plot(cat_data['submission_year'], cat_data['count'], 
                marker='o', linewidth=2.5, markersize=6, 
                label=category, color=colors[i])
    
    plt.title('Research Area Growth Over Time (Top 7 Categories)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Submission Year', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Publications', fontsize=13, fontweight='bold')
    plt.legend(title='Category', title_fontsize=12, fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'research_growth_trend_top7.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a bar chart for total publications by category
    plt.figure(figsize=(12, 7))
    category_totals = growth_data.groupby('main_categories')['count'].sum().nlargest(15).sort_values(ascending=True)
    bars = plt.barh(category_totals.index, category_totals.values, color=sns.color_palette("viridis", len(category_totals)))
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(category_totals.items()):
        plt.text(val + 50, i, f'{val:,}', va='center', fontsize=10, fontweight='bold')
    
    plt.title('Total Publications by Research Area (Top 15)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Total Publications', fontsize=13, fontweight='bold')
    plt.ylabel('Research Area', fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'research_area_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis data and plots saved to {OUTPUT_DIR}")
    print(f"  - research_growth_trend_top7.png (line chart)")
    print(f"  - research_area_distribution.png (bar chart)")

def analyze_collaboration(df):
    """
    Analyzes collaboration patterns (using co-authorship as a proxy).
    NOTE: Full analysis requires author affiliation/country data, which is not in the mock dataset.
    """
    print("\n--- 2. Collaboration Analysis (Co-authorship) ---")
    
    # Convert authors string to list
    # Check if it's already a list representation or a comma-separated string
    def parse_authors(x):
        if pd.isna(x):
            return []
        if isinstance(x, str):
            # Try parsing as Python list first (for mock data)
            if x.strip().startswith('['):
                try:
                    return ast.literal_eval(x)
                except:
                    pass
            # Otherwise, split by comma (for processed CSV data)
            return [author.strip() for author in x.split(',')]
        return x
    
    df['authors_list'] = df['authors'].apply(parse_authors)
    
    # Count co-authorship
    df['num_authors'] = df['authors_list'].apply(len)
    
    print("Distribution of co-authorship:")
    print(df['num_authors'].value_counts().sort_index())
    
    # Create collaboration visualization
    plt.figure(figsize=(14, 6))
    
    # Limit to papers with 1-20 authors for clarity
    collab_data = df[df['num_authors'] <= 20]['num_authors'].value_counts().sort_index()
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(collab_data.index, collab_data.values, color=sns.color_palette("coolwarm", len(collab_data)))
    plt.title('Distribution of Authors per Paper (1-20 authors)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Authors', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Papers', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Pie chart for single vs multi-author papers
    plt.subplot(1, 2, 2)
    single_author = (df['num_authors'] == 1).sum()
    multi_author = (df['num_authors'] > 1).sum()
    
    sizes = [single_author, multi_author]
    labels = [f'Single Author\n({single_author:,} papers)', f'Multi-Author\n({multi_author:,} papers)']
    colors_pie = ['#ff9999', '#66b3ff']
    explode = (0.05, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    plt.title('Single vs Multi-Author Papers', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'collaboration_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Collaboration visualization saved to {OUTPUT_DIR}/collaboration_analysis.png")
    
    # NOTE: To address the "international collaboration" and "country domination" questions,
    # the full dataset must be used to extract author affiliations and map them to countries.
    # A heat map of (Field of Study) x (Country) would then be generated.
    
    print("\nNOTE: International collaboration analysis requires author affiliation data (country).")
    print("This mock analysis only shows the number of co-authors per paper.")

def analyze_interdisciplinarity(df):
    """
    Analyzes the relationship between interdisciplinarity and citation count.
    NOTE: Requires citation data, which is not in the mock dataset.
    """
    print("\n--- 3. Interdisciplinarity and Citation Analysis ---")
    
    # Count the number of main categories per paper
    df['discipline_count'] = df['main_categories'].apply(len)
    
    print("Distribution of discipline count:")
    print(df['discipline_count'].value_counts().sort_index())
    
    # Create interdisciplinarity visualization
    plt.figure(figsize=(12, 7))
    
    discipline_dist = df['discipline_count'].value_counts().sort_index()
    
    # Bar plot
    bars = plt.bar(discipline_dist.index, discipline_dist.values, 
                   color=sns.color_palette("plasma", len(discipline_dist)),
                   edgecolor='black', linewidth=1.2)
    
    plt.title('Interdisciplinarity Distribution\n(Number of Disciplines per Paper)', 
              fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Number of Disciplines', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Papers', fontsize=13, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Add percentage labels
    total_papers = discipline_dist.sum()
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_papers) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{percentage:.1f}%', ha='center', va='center', 
                fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'interdisciplinarity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Interdisciplinarity visualization saved to {OUTPUT_DIR}/interdisciplinarity_analysis.png")
    
    # NOTE: To address the question "Are interdisciplinary papers cited more frequently?",
    # we need a 'citation_count' column. The analysis would involve:
    # 1. Calculating the average citation count for papers with 1, 2, 3, ... disciplines.
    # 2. Performing a statistical test (e.g., t-test or ANOVA) to compare the means.
    
    print("\nNOTE: Citation analysis requires 'citation_count' data.")
    print("This mock analysis only shows the number of disciplines per paper.")

def analyze_emerging_keywords(df):
    """
    Identifies emerging keywords based on frequency and recent growth.
    """
    print("\n--- 5. Emerging Keywords Analysis ---")
    
    # Combine titles and abstracts for keyword extraction
    text_data = (df['title'] + ' ' + df['abstract']).str.lower().str.replace(r'[^a-z\s]', '', regex=True).str.split()
    
    # Simple word frequency count (requires more sophisticated NLP for real analysis)
    all_words = [word for sublist in text_data.dropna() for word in sublist if len(word) > 3]
    word_counts = Counter(all_words)
    
    print("Top 20 most frequent words (excluding common stop words):")
    # In a real scenario, a proper stop word list would be used
    common_words = {'this', 'paper', 'analysis', 'using', 'which', 'from', 'with', 'study', 'research', 'novel', 'technique', 'approach', 'that', 'these', 'have', 'their', 'also', 'such', 'both'}
    filtered_counts = {word: count for word, count in word_counts.items() if word not in common_words}
    
    top_keywords = Counter(filtered_counts).most_common(20)
    print(top_keywords)
    
    # Create keyword visualization
    plt.figure(figsize=(14, 8))
    
    keywords = [k[0] for k in top_keywords]
    counts = [k[1] for k in top_keywords]
    
    # Horizontal bar chart for better readability
    colors = sns.color_palette("rocket", len(keywords))
    bars = plt.barh(range(len(keywords)), counts, color=colors, edgecolor='black', linewidth=1.2)
    
    plt.yticks(range(len(keywords)), keywords, fontsize=11, fontweight='bold')
    plt.xlabel('Frequency', fontsize=13, fontweight='bold')
    plt.title('Top 20 Most Frequent Keywords in Abstracts and Titles', fontsize=15, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count + 50, i, f'{count:,}', va='center', fontsize=10, fontweight='bold')
    
    plt.gca().invert_yaxis()  # Highest at top
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'keyword_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Keyword visualization saved to {OUTPUT_DIR}/keyword_analysis.png")
    
    # For "emerging" keywords, one would compare the frequency of a keyword in the last 2 years
    # vs. the previous 5 years, and look for keywords with a high growth rate.
    
    print("\nNOTE: Real emerging keyword analysis requires a large corpus and time-series comparison.")

def main():
    try:
        # Load data, using mock=False to use the processed sample
        df = load_processed_data(mock=False)
        
        # Ensure 'main_categories' is a list of strings
        df['main_categories'] = df['main_categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Run analyses
        analyze_research_growth(df)
        analyze_collaboration(df)
        analyze_interdisciplinarity(df)
        analyze_emerging_keywords(df)
        
        print("\nExploratory analysis complete. Results saved to data/processed/analysis_results.")
        print("NOTE: Full analysis requires the complete arXiv dataset and Semantic Scholar citation data.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

