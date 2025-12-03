"""
Flask Application for Results Page

Web application for displaying analysis results with:
- Interactive word cloud generation from paper titles and abstracts
- Comprehensive Phase-1, Phase-2, and Phase-3 analysis results
- Filtering by year range for word cloud generation
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import sys
import json
import base64
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

app = Flask(__name__)

# Configure static files for analysis results
ANALYSIS_RESULTS_DIR = os.path.join(project_root, 'data', 'processed', 'analysis_results')

# Load data once when the app starts
# NOTE: Using raw arxiv_sample_10k.json file directly to avoid loading 2.2M dataset
RAW_SAMPLE_PATH = os.path.join(project_root, 'data', 'raw', 'arxiv_sample_10k.json')

def get_submission_year(versions):
    """
    Extract submission year from paper versions data.
    
    Args:
        versions: List of version dictionaries with 'created' timestamps
    
    Returns:
        int: Year of first submission, or None if parsing fails
    """
    if versions:
        # Get the first version's creation date (earliest submission)
        date_str = versions[0]['created']  # Format: 'Mon, 20 Oct 2008 11:36:39 GMT'
        try:
            # Try parsing the full date-time format
            return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z').year
        except ValueError:
            try:
                # Fallback: parse date without time (in case format differs)
                date_parts = ' '.join(date_str.split()[:4])
                return datetime.strptime(date_parts, '%a, %d %b %Y').year
            except ValueError:
                return None
    return None

def load_wordcloud_sample():
    """
    Load 10K sample for word cloud directly from raw arxiv_sample_10k.json file.
    This avoids loading the full 2.2M dataset on every worker startup.
    """
    # Load directly from raw JSON file
    if not os.path.exists(RAW_SAMPLE_PATH):
        print(f"⚠ Warning: Raw sample file not found at {RAW_SAMPLE_PATH}")
        print("⚠ Word cloud functionality will be limited.")
        return pd.DataFrame({'title': [], 'abstract': [], 'text_content': [], 'main_categories': [], 'submission_year': []})
    
    print(f"✓ Loading raw 10K sample from: {RAW_SAMPLE_PATH}")
    
    try:
        records = []
        with open(RAW_SAMPLE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    # Extract necessary fields
                    records.append({
                        'id': record.get('id', ''),
                        'title': record.get('title', ''),
                        'abstract': record.get('abstract', ''),
                        'authors': record.get('authors', ''),
                        'categories': record.get('categories', ''),
                        'versions': record.get('versions', [])
                    })
                except json.JSONDecodeError:
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Process categories to get main_categories
        df['main_categories'] = df['categories'].apply(
            lambda x: [c.split('.')[0] for c in x.split(' ')] if isinstance(x, str) else []
        )
        
        # Extract submission year from versions
        df['submission_year'] = df['versions'].apply(get_submission_year)
        
        # Create text_content for word cloud
        df['text_content'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
        
        print(f"✓ Loaded {len(df):,} papers for word cloud")
        return df
        
    except Exception as e:
        print(f"⚠ Error loading raw sample file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({'title': [], 'abstract': [], 'text_content': [], 'main_categories': [], 'submission_year': []})

# Note: Data is now loaded on-demand in the wordcloud_endpoint() function
# to avoid loading it continuously when the app starts

def generate_word_cloud(text, filtered_df=None):
    """
    Generate word cloud visualization from text with statistics.
    
    Creates a comprehensive word cloud visualization including:
    - Main word cloud image
    - Top 20 keywords bar chart
    - Statistics panel with paper counts and category info
    
    Args:
        text: Combined text from titles and abstracts
        filtered_df: Optional DataFrame for statistics calculation
    
    Returns:
        tuple: (base64_encoded_image, word_frequencies_dict)
    """
    if not text:
        return "", {}
        
    # Comprehensive stop words list
    stop_words = set([
        'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'on', 'with', 'this', 
        'paper', 'analysis', 'using', 'which', 'from', 'study', 'research', 'novel', 
        'technique', 'approach', 'we', 'our', 'results', 'show', 'present', 'propose',
        'method', 'methods', 'data', 'model', 'models', 'based', 'different', 'also',
        'can', 'used', 'use', 'one', 'two', 'new', 'time', 'more', 'these', 'their',
        'than', 'when', 'where', 'what', 'how', 'why', 'are', 'was', 'were', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'being', 'been', 'become', 'becomes', 'becoming'
    ])
    
    # Generate word cloud with better settings
    wordcloud = WordCloud(
        width=1000, 
        height=500, 
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10,
        max_font_size=100,
        collocations=True
    ).generate(text)
    
    # Get word frequencies for statistics
    word_freq = wordcloud.words_
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Create enhanced visualization with subplots
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35, figure=fig)
    
    # Main word cloud (larger, left side)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(wordcloud, interpolation='bilinear')
    ax1.axis("off")
    ax1.set_title('Word Cloud Visualization', fontsize=14, fontweight='bold', pad=20)
    
    # Top 20 words bar chart (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if top_words:
        words, freqs = zip(*top_words)
        colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
        bars = ax2.barh(range(len(words)), freqs, color=colors)
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels(words, fontsize=9)
        ax2.set_xlabel('Relative Frequency', fontsize=10, fontweight='bold')
        ax2.set_title('Top 20 Keywords', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        # Add value labels
        for i, freq in enumerate(freqs):
            ax2.text(freq + 0.01, i, f'{freq:.2f}', va='center', fontsize=8)
    
    # Statistics panel (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Calculate statistics if filtered_df is provided
    stats_text = "WORD CLOUD STATISTICS\n\n"
    if filtered_df is not None and len(filtered_df) > 0:
        stats_text += f"Papers Analyzed: {len(filtered_df):,}\n"
        stats_text += f"Total Words: {len(text.split()):,}\n"
        stats_text += f"Unique Words: {len(set(text.lower().split())):,}\n"
        stats_text += f"Top Categories:\n"
        try:
            top_cats = filtered_df['main_categories'].explode().value_counts().head(3)
            for i, (cat, count) in enumerate(top_cats.items(), 1):
                stats_text += f"  {i}. {cat}: {count}\n"
        except:
            pass
        stats_text += f"\nYear Range:\n"
        try:
            if 'submission_year' in filtered_df.columns:
                stats_text += f"  {filtered_df['submission_year'].min()}-{filtered_df['submission_year'].max()}\n"
        except:
            pass
    else:
        stats_text += f"Total Words: {len(text.split()):,}\n"
        stats_text += f"Unique Words: {len(set(text.lower().split())):,}\n"
        stats_text += f"Top Keywords: {len(top_words)}\n"
    
    ax3.text(0.1, 0.5, stats_text, fontsize=10, fontweight='bold',
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.35, wspace=0.35)
    
    # Convert to PNG image in memory
    img = BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img.seek(0)
    
    # Encode to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    
    # Return image and statistics
    stats_dict = {
        'top_words': dict(top_words[:10]),  # Top 10 for JSON
        'total_words': len(text.split()),
        'unique_words': len(set(text.lower().split())),
        'paper_count': len(filtered_df) if filtered_df is not None else 0
    }
    
    return f"data:image/png;base64,{img_base64}", stats_dict

@app.route('/')
def index():
    """
    Redirect root to results page.
    
    Returns:
        Redirect: Redirects to /results page
    """
    from flask import redirect
    return redirect('/results')


@app.route('/wordcloud', methods=['POST'])
def wordcloud_endpoint():
    """
    Generate word cloud from filtered papers based on keyword and year range.
    
    NOTE: This endpoint loads the raw arxiv_sample_10k.json file on-demand
    only when the user clicks the "Generate Word Cloud" button, avoiding
    continuous memory usage when the app is idle.
    
    Accepts POST request with:
    - word: Keyword to search in titles/abstracts
    - year_min: Minimum submission year (optional)
    - year_max: Maximum submission year (optional)
    
    Returns:
        JSON: Base64-encoded word cloud image, paper count, statistics, and metadata
    """
    try:
        w1 = request.form.get('word', '').strip().lower()
        year_min = request.form.get('year_min', '').strip()
        year_max = request.form.get('year_max', '').strip()
        
        if not w1:
            return jsonify({'error': 'Please enter a keyword.'}), 400

        # Load the dataset on-demand only when user requests word cloud
        print(f"Loading word cloud data for keyword: {w1}")
        df = load_wordcloud_sample()
        
        if df.empty:
            return jsonify({'error': 'Word cloud data could not be loaded. Please try again later.'}), 500

        # Filter papers where w1 is present in the title or abstract
        # Handle multi-word phrases: if keyword has spaces, search as phrase; otherwise use word boundaries
        if ' ' in w1:
            # Multi-word phrase: search for exact phrase
            search_pattern = w1
        else:
            # Single word: use word boundaries
            search_pattern = r'\b' + w1 + r'\b'
        
        filtered_df = df[df['text_content'].str.contains(search_pattern, case=False, na=False, regex=(' ' not in w1))].copy()
        
        # Apply year filters if provided
        if year_min:
            try:
                year_min = int(year_min)
                if 'submission_year' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['submission_year'] >= year_min]
            except (ValueError, KeyError, TypeError):
                pass
        
        if year_max:
            try:
                year_max = int(year_max)
                if 'submission_year' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['submission_year'] <= year_max]
            except (ValueError, KeyError, TypeError):
                pass
        
        if filtered_df.empty:
            # Provide helpful error message
            error_msg = f'No papers found containing "{w1}"'
            if year_min or year_max:
                year_range = []
                if year_min:
                    year_range.append(f'from {year_min}')
                if year_max:
                    year_range.append(f'to {year_max}')
                error_msg += f' in the year range {", ".join(year_range)}'
            error_msg += '. Try: removing year filters, using a different keyword, or checking spelling.'
            return jsonify({'error': error_msg}), 404

        # Combine all text content from the filtered papers
        combined_text = " ".join(filtered_df['text_content'].tolist())
        
        # Generate enhanced word cloud with statistics
        img_data, stats = generate_word_cloud(combined_text, filtered_df)
        
        # Get additional statistics
        categories = {}
        year_range = {}
        try:
            categories = filtered_df['main_categories'].explode().value_counts().head(5).to_dict()
        except Exception:
            pass
        
        try:
            if 'submission_year' in filtered_df.columns and not filtered_df['submission_year'].isna().all():
                year_range = {
                    'min': int(filtered_df['submission_year'].min()),
                    'max': int(filtered_df['submission_year'].max())
                }
        except Exception:
            pass
        
        return jsonify({
            'image': img_data, 
            'count': len(filtered_df),
            'stats': stats,
            'categories': categories,
            'year_range': year_range
        })
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/results')
def results():
    """
    Render comprehensive analysis results page.
    
    Returns:
        HTML: Rendered results.html template with all analysis visualizations
    """
    return render_template('results.html')

@app.route('/static/analysis_results/<path:filename>')
def serve_analysis_image(filename):
    """
    Serve analysis result images from the analysis_results directory.
    
    Args:
        filename: Path to the image file relative to analysis_results directory
    
    Returns:
        Image file response
    """
    try:
        # Split the filename to get subdirectory and actual filename
        parts = filename.split('/', 1)
        if len(parts) == 2:
            subdir, actual_filename = parts
            directory = os.path.join(ANALYSIS_RESULTS_DIR, subdir)
        else:
            directory = ANALYSIS_RESULTS_DIR
            actual_filename = filename
        
        # Verify the file exists
        file_path = os.path.join(directory, actual_filename)
        if not os.path.exists(file_path):
            return f"File not found: {filename}", 404
        
        # Determine MIME type based on file extension
        mime_type = 'image/png' if filename.lower().endswith('.png') else 'image/jpeg'
        
        response = send_from_directory(directory, actual_filename, mimetype=mime_type)
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except Exception as e:
        return f"Error serving image: {str(e)}", 500

# Dashboard stats endpoint removed - no longer needed since index.html was deleted

if __name__ == '__main__':
    # For local development, use a standard port
    # In the sandbox, we will use gunicorn to run the app
    print("Results page application ready. Run with: gunicorn -w 4 -b 0.0.0.0:8080 app:app")
    # app.run(debug=True, port=5000)

