import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_processed_data

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'analysis_results', 'predictive_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_citation_proxy(df):
    """Creates a proxy for citation count."""
    current_year = 2025
    df['paper_age'] = current_year - df['submission_year']
    
    df['version_count'] = df['versions'].apply(
        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x) if isinstance(x, list) else 1
    )
    
    # Create composite citation proxy
    df['citation_proxy'] = (
        (df['paper_age'] / df['paper_age'].max()) * 0.6 +
        (df['version_count'] / df['version_count'].max()) * 0.4
    ) * 100
    
    return df

def prepare_features(df):
    """Prepare features for predictive modeling."""
    # Ensure main_categories is a list
    df['main_categories'] = df['main_categories'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Create derived features
    df['num_authors'] = df['authors'].apply(
        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.strip().startswith('[') 
        else len(x.split(',')) if isinstance(x, str) else 0
    )
    df['discipline_count'] = df['main_categories'].apply(len)
    df['title_length'] = df['title'].str.len()
    df['abstract_length'] = df['abstract'].str.len()
    
    # Create citation proxy
    df = create_citation_proxy(df)
    
    # One-hot encode top categories
    df_exploded = df.explode('main_categories')
    top_categories = df_exploded['main_categories'].value_counts().head(10).index.tolist()
    
    for cat in top_categories:
        df[f'category_{cat}'] = df['main_categories'].apply(lambda x: 1 if cat in x else 0)
    
    # Select features
    feature_cols = [
        'submission_year', 'num_authors', 'discipline_count', 
        'title_length', 'abstract_length', 'paper_age', 'version_count'
    ] + [f'category_{cat}' for cat in top_categories]
    
    # Remove rows with missing values in key features
    df_clean = df[feature_cols + ['citation_proxy']].dropna()
    
    X = df_clean[feature_cols]
    y = df_clean['citation_proxy']
    
    return X, y, feature_cols, top_categories

def train_citation_prediction_model():
    """
    Train models to predict citation proxy.
    Uses Linear Regression and Random Forest.
    """
    print("\n" + "="*80)
    print("PREDICTIVE MODEL: Citation Prediction")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    df = load_processed_data(mock=False)
    X, y, feature_cols, top_categories = prepare_features(df)
    
    print(f"  • Total samples: {len(X):,}")
    print(f"  • Features: {len(feature_cols)}")
    print(f"  • Target variable: citation_proxy")
    
    # Split data (temporal split - older papers for training)
    df_temp = df[df['submission_year'].notna()].copy()
    df_temp = df_temp.sort_values('submission_year')
    split_idx = int(len(df_temp) * 0.8)
    
    train_indices = df_temp.index[:split_idx]
    test_indices = df_temp.index[split_idx:]
    
    X_train = X.loc[X.index.isin(train_indices)]
    X_test = X.loc[X.index.isin(test_indices)]
    y_train = y.loc[y.index.isin(train_indices)]
    y_test = y.loc[y.index.isin(test_indices)]
    
    print(f"\nData Split:")
    print(f"  • Training set: {len(X_train):,} samples")
    print(f"  • Test set: {len(X_test):,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test,
            'scaler': scaler if name == 'Linear Regression' else None
        }
        
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features (Random Forest):")
        print(feature_importance.head(10).to_string(index=False))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        colors = sns.color_palette("viridis", len(top_features))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=11)
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.title('Feature Importance: Citation Prediction (Random Forest)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance'] + 0.001, i, f"{row['importance']:.3f}", 
                    va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'citation_prediction_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Feature importance plot saved: citation_prediction_feature_importance.png")
    
    # Prediction vs Actual plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Scatter plot: predicted vs actual
        ax.scatter(y_test, result['y_pred_test'], alpha=0.5, s=20, 
                  color=sns.color_palette("Set2", len(results))[idx])
        
        # Perfect prediction line
        min_val = min(y_test.min(), result['y_pred_test'].min())
        max_val = max(y_test.max(), result['y_pred_test'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Citation Proxy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Citation Proxy', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}\nTest R² = {result["test_r2"]:.4f}, RMSE = {result["test_rmse"]:.4f}', 
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Citation Prediction: Model Performance Comparison', 
                fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'citation_prediction_performance.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Model performance plot saved: citation_prediction_performance.png")
    
    # Save results summary
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train RMSE': [r['train_rmse'] for r in results.values()],
        'Test RMSE': [r['test_rmse'] for r in results.values()],
        'Train MAE': [r['train_mae'] for r in results.values()],
        'Test MAE': [r['test_mae'] for r in results.values()],
        'Train R²': [r['train_r2'] for r in results.values()],
        'Test R²': [r['test_r2'] for r in results.values()]
    })
    
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'citation_prediction_results.csv'), index=False)
    print(f"✓ Results summary saved: citation_prediction_results.csv")
    
    return results, feature_importance if 'Random Forest' in results else None

def forecast_research_growth():
    """
    Forecast future research growth for top categories using simple time-series methods.
    """
    print("\n" + "="*80)
    print("PREDICTIVE MODEL: Research Growth Forecasting")
    print("="*80)
    
    # Load data
    df = load_processed_data(mock=False)
    df['main_categories'] = df['main_categories'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Get top 5 categories
    df_exploded = df.explode('main_categories')
    top_categories = df_exploded['main_categories'].value_counts().head(5).index.tolist()
    
    print(f"\nForecasting growth for top 5 categories: {top_categories}")
    
    # Prepare time series data
    growth_data = df_exploded.groupby(['submission_year', 'main_categories']).size().reset_index(name='count')
    growth_data = growth_data[growth_data['main_categories'].isin(top_categories)]
    
    # Forecast next 3 years (2026-2028)
    forecast_years = [2026, 2027, 2028]
    forecasts = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, category in enumerate(top_categories):
        cat_data = growth_data[growth_data['main_categories'] == category].sort_values('submission_year')
        
        if len(cat_data) < 5:
            continue
        
        years = cat_data['submission_year'].values
        counts = cat_data['count'].values
        
        # Simple linear trend forecast
        z = np.polyfit(years, counts, 1)
        p = np.poly1d(z)
        
        # Forecast
        forecast_counts = [p(year) for year in forecast_years]
        forecast_counts = [max(0, int(c)) for c in forecast_counts]  # Ensure non-negative
        
        forecasts[category] = dict(zip(forecast_years, forecast_counts))
        
        # Plot
        ax = axes[idx]
        
        # Historical data
        ax.plot(years, counts, 'o-', linewidth=2, markersize=6, 
               label='Historical', color=sns.color_palette("husl", 5)[idx])
        
        # Forecast
        all_years = np.concatenate([years, forecast_years])
        all_counts = np.concatenate([counts, forecast_counts])
        ax.plot(forecast_years, forecast_counts, 's--', linewidth=2, markersize=8,
               label='Forecast', color='red', alpha=0.7)
        
        ax.axvline(x=2025, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Current Year')
        
        ax.set_title(f'{category.upper()}\nForecast: {forecast_counts[0]}, {forecast_counts[1]}, {forecast_counts[2]}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Publications', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Remove extra subplot
    if len(top_categories) < len(axes):
        axes[-1].remove()
    
    plt.suptitle('Research Growth Forecasting (2026-2028)\nLinear Trend Projection', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'research_growth_forecast.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Forecast visualization saved: research_growth_forecast.png")
    
    # Print forecasts
    print("\nGrowth Forecasts (2026-2028):")
    forecast_df = pd.DataFrame(forecasts).T
    forecast_df.columns = [f'{y}' for y in forecast_years]
    print(forecast_df.to_string())
    
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'research_growth_forecast.csv'), index=True)
    print(f"\n✓ Forecast data saved: research_growth_forecast.csv")
    
    return forecasts

def classify_emerging_keywords():
    """
    Classify keywords as 'emerging' based on growth rate.
    """
    print("\n" + "="*80)
    print("PREDICTIVE MODEL: Emerging Keyword Classification")
    print("="*80)
    
    # Load data
    df = load_processed_data(mock=False)
    
    # Combine titles and abstracts
    df['text_content'] = df['title'] + ' ' + df['abstract']
    
    # Extract keywords by time period
    recent_years = df[df['submission_year'] >= 2020]
    older_years = df[df['submission_year'] < 2020]
    
    if len(recent_years) == 0 or len(older_years) == 0:
        print("Insufficient data for time-based comparison.")
        return
    
    from collections import Counter
    import re
    
    def extract_keywords(text_series):
        all_words = []
        for text in text_series.dropna():
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            all_words.extend(words)
        return Counter(all_words)
    
    # Common stop words
    stop_words = {'this', 'paper', 'analysis', 'using', 'which', 'from', 'with', 'study', 
                 'research', 'novel', 'technique', 'approach', 'that', 'these', 'have', 
                 'their', 'also', 'such', 'both', 'results', 'method', 'data', 'model',
                 'based', 'different', 'however', 'therefore', 'between', 'within'}
    
    recent_keywords = extract_keywords(recent_years['text_content'])
    older_keywords = extract_keywords(older_years['text_content'])
    
    # Calculate growth rates
    keyword_growth = []
    
    # Get keywords that appear in both periods
    common_keywords = set(recent_keywords.keys()) & set(older_keywords.keys())
    
    for keyword in common_keywords:
        if keyword in stop_words:
            continue
        
        recent_count = recent_keywords[keyword]
        older_count = older_keywords[keyword]
        
        # Normalize by number of papers
        recent_freq = recent_count / len(recent_years)
        older_freq = older_count / len(older_years)
        
        if older_freq > 0:
            growth_rate = ((recent_freq - older_freq) / older_freq) * 100
        else:
            growth_rate = 1000 if recent_freq > 0 else 0
        
        keyword_growth.append({
            'keyword': keyword,
            'recent_freq': recent_freq,
            'older_freq': older_freq,
            'growth_rate': growth_rate
        })
    
    growth_df = pd.DataFrame(keyword_growth)
    growth_df = growth_df.sort_values('growth_rate', ascending=False)
    
    # Classify as emerging (top 10% by growth rate)
    threshold = growth_df['growth_rate'].quantile(0.9)
    growth_df['is_emerging'] = growth_df['growth_rate'] >= threshold
    
    emerging_keywords = growth_df[growth_df['is_emerging']].head(20)
    
    print(f"\nTop 20 Emerging Keywords (Growth Rate >= {threshold:.2f}%):")
    print(emerging_keywords[['keyword', 'growth_rate', 'recent_freq', 'older_freq']].to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(14, 8))
    top_emerging = emerging_keywords.head(15)
    
    colors = ['red' if gr >= threshold else 'blue' for gr in top_emerging['growth_rate']]
    bars = plt.barh(range(len(top_emerging)), top_emerging['growth_rate'], color=colors, alpha=0.7)
    
    plt.yticks(range(len(top_emerging)), top_emerging['keyword'], fontsize=11, fontweight='bold')
    plt.xlabel('Growth Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Top 15 Emerging Keywords\n(Comparing 2020+ vs Pre-2020)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold ({threshold:.1f}%)', alpha=0.7)
    plt.legend()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(top_emerging.iterrows()):
        plt.text(row['growth_rate'] + 5, i, f"{row['growth_rate']:.1f}%", 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'emerging_keywords_classification.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Emerging keywords visualization saved: emerging_keywords_classification.png")
    
    growth_df.to_csv(os.path.join(OUTPUT_DIR, 'emerging_keywords_analysis.csv'), index=False)
    print(f"✓ Emerging keywords data saved: emerging_keywords_analysis.csv")
    
    return growth_df

def main():
    """Main function to run all predictive models."""
    try:
        print("\n" + "="*80)
        print("PREDICTIVE MODELING ANALYSIS")
        print("="*80)
        
        # Run all models
        citation_results, feature_importance = train_citation_prediction_model()
        growth_forecasts = forecast_research_growth()
        emerging_keywords = classify_emerging_keywords()
        
        print("\n" + "="*80)
        print("✅ PREDICTIVE MODELING COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to: {OUTPUT_DIR}")
        print("\nGenerated Files:")
        print("  1. citation_prediction_performance.png - Model comparison")
        print("  2. citation_prediction_feature_importance.png - Feature importance")
        print("  3. citation_prediction_results.csv - Model metrics")
        print("  4. research_growth_forecast.png - Growth forecasts")
        print("  5. research_growth_forecast.csv - Forecast data")
        print("  6. emerging_keywords_classification.png - Emerging keywords")
        print("  7. emerging_keywords_analysis.csv - Keyword analysis")
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

