"""
Visualization functions for persona analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from datetime import datetime
import os
import re

def ensure_output_dir(output_dir: str) -> None:
    """Ensure output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def create_heatmap(data: pd.DataFrame, output_dir: str, timestamp: str):
    """Create heatmap visualization."""
    ensure_output_dir(output_dir)
    plt.figure(figsize=(12, 8))
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numeric Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{timestamp}.png'))
    plt.close()

def create_sentiment_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """Create sentiment analysis visualization."""
    ensure_output_dir(output_dir)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['sentiment_score'], bins=20)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'sentiment_analysis_{timestamp}.png'))
    plt.close()

def create_impact_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """Create impact analysis visualization."""
    ensure_output_dir(output_dir)
    impact_cols = ['impact_on_housing', 'impact_on_transport', 'impact_on_community']
    # Select only existing impact columns from the flattened DataFrame
    existing_impact_cols = [col for col in impact_cols if col in data.columns]
    
    if not existing_impact_cols:
        print(f"Skipping impact_analysis_{timestamp}.png: No impact columns ({', '.join(impact_cols)}) found in the DataFrame.")
        # plt.close() # No figure to close if we skip
        return

    plt.figure(figsize=(12, 6))
    data[existing_impact_cols].boxplot()
    plt.title('Impact Analysis Across Different Areas')
    plt.ylabel('Impact Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'impact_analysis_{timestamp}.png'))
    plt.close()

def normalize_text(text):
    """Normalize text for grouping: lowercase, strip, remove punctuation."""
    if not isinstance(text, str):
        return text
    return re.sub(r'[^a-z0-9 ]', '', text.lower().strip())

def create_theme_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """Create theme analysis visualization with canonical themes."""
    ensure_output_dir(output_dir)
    
    # Count canonical theme occurrences
    theme_counts = {}
    for themes in data['canonical_themes']:
        if isinstance(themes, list):
            for theme_dict in themes:
                if isinstance(theme_dict, dict) and 'canonical' in theme_dict:
                    canonical = theme_dict['canonical']
                    theme_counts[canonical] = theme_counts.get(canonical, 0) + 1
    
    if not theme_counts:
        print(f"Skipping theme_analysis_{timestamp}.png: No canonical themes found.")
        return
    
    # Create bar plot of canonical themes
    plt.figure(figsize=(12, 6))
    themes = list(theme_counts.keys())
    counts = list(theme_counts.values())
    
    # Sort by count in descending order
    sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    themes = [themes[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    plt.bar(themes, counts)
    plt.title('Frequency of Canonical Themes')
    plt.xlabel('Canonical Theme')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'theme_analysis_{timestamp}.png'))
    plt.close()
    
    # Create a second plot showing the mapping of original themes to canonical themes
    theme_mapping = {}
    for themes in data['canonical_themes']:
        if isinstance(themes, list):
            for theme_dict in themes:
                if isinstance(theme_dict, dict) and 'canonical' in theme_dict and 'original' in theme_dict:
                    canonical = theme_dict['canonical']
                    original = theme_dict['original']
                    if canonical not in theme_mapping:
                        theme_mapping[canonical] = set()
                    theme_mapping[canonical].add(original)
    
    # Create a text file with the theme mapping
    mapping_file = os.path.join(output_dir, f'theme_mapping_{timestamp}.txt')
    with open(mapping_file, 'w') as f:
        f.write("Canonical Theme Mapping:\n")
        f.write("=" * 50 + "\n\n")
        for canonical, originals in sorted(theme_mapping.items()):
            f.write(f"Canonical Theme: {canonical}\n")
            f.write("-" * 30 + "\n")
            for original in sorted(originals):
                f.write(f"- {original}\n")
            f.write("\n")
    
    print(f"Theme mapping saved to: {mapping_file}")

def create_location_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """Create location analysis visualization."""
    ensure_output_dir(output_dir)
    plt.figure(figsize=(10, 6))
    # Extract location from nested persona_details
    location_counts = data['persona_details'].apply(lambda x: x['location']).value_counts()
    location_counts.plot(kind='bar')
    plt.title('Distribution of Personas by Location')
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'location_analysis_{timestamp}.png'))
    plt.close()

def create_occupation_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """Create occupation analysis visualization."""
    ensure_output_dir(output_dir)
    plt.figure(figsize=(12, 6))
    # Extract occupation from nested persona_details
    occupation_counts = data['persona_details'].apply(lambda x: x['occupation']).value_counts()
    occupation_counts.plot(kind='bar')
    plt.title('Distribution of Personas by Occupation')
    plt.xlabel('Occupation')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'occupation_analysis_{timestamp}.png'))
    plt.close()

def create_correlation_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """Create correlation analysis visualization."""
    plt.figure(figsize=(12, 8))
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Analysis of Numeric Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_analysis_{timestamp}.png'))
    plt.close()

def create_concern_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """Create concern analysis visualization with normalization for grouping."""
    ensure_output_dir(output_dir)
    concern_counts = {}
    if 'key_concerns' in data.columns:
        for concerns_list in data['key_concerns'].dropna():
            if isinstance(concerns_list, list):
                for concern in concerns_list:
                    norm_concern = normalize_text(concern)
                    concern_counts[norm_concern] = concern_counts.get(norm_concern, 0) + 1
    else:
        print(f"Skipping concern_analysis_{timestamp}.png: 'key_concerns' column not found.")
        return
    
    if concern_counts: # Check if there are any concerns to plot
        plt.figure(figsize=(12, 6))
        concerns = list(concern_counts.keys())
        counts = list(concern_counts.values())
        plt.bar(concerns, counts)
        plt.title('Frequency of Key Concerns (Normalized)')
        plt.xlabel('Concern')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'concern_analysis_{timestamp}.png'))
        plt.close()
    else:
        print(f"No key concerns data to plot for concern_analysis_{timestamp}.png")

def generate_analysis_report(data: pd.DataFrame, output_dir: str, timestamp: str) -> None:
    """Generate comprehensive analysis report."""
    ensure_output_dir(output_dir)
    
    # Create all visualizations
    create_heatmap(data, output_dir, timestamp)
    create_sentiment_analysis(data, output_dir, timestamp)
    create_impact_analysis(data, output_dir, timestamp)
    create_theme_analysis(data, output_dir, timestamp)
    create_location_analysis(data, output_dir, timestamp)
    create_occupation_analysis(data, output_dir, timestamp)
    create_correlation_analysis(data, output_dir, timestamp)
    create_concern_analysis(data, output_dir, timestamp)
    
    # Calculate Average Impact Scores from direct columns
    avg_impact_scores = {}
    impact_score_cols_map = {
        'impact_on_housing': 'Housing',
        'impact_on_transport': 'Transport',
        'impact_on_community': 'Community'
    }
    for col_name, area_name in impact_score_cols_map.items():
        if col_name in data.columns and pd.api.types.is_numeric_dtype(data[col_name]) and not data[col_name].dropna().empty:
            avg_impact_scores[area_name] = data[col_name].dropna().mean()
        else:
            avg_impact_scores[area_name] = 0
            print(f"[Warning] Could not calculate mean for impact score: {area_name}. Column '{col_name}' missing, not numeric, or all NaN.")

    # Calculate Most Common Concerns from direct column
    most_common_concerns = {}
    if 'key_concerns' in data.columns:
        # Ensure items in 'key_concerns' are lists, then explode, then count
        # Filter out non-list items or handle them appropriately if necessary
        valid_concerns_lists = data['key_concerns'].dropna().apply(lambda x: x if isinstance(x, list) else [])
        if not valid_concerns_lists.empty:
            all_concerns_series = valid_concerns_lists.explode().dropna()
            if not all_concerns_series.empty:
                most_common_concerns = all_concerns_series.value_counts().head(5).to_dict()

    # Generate summary statistics
    summary_stats = {
        'Total Personas': len(data),
        'Average Sentiment Score': data['sentiment_score'].dropna().mean() if 'sentiment_score' in data and pd.api.types.is_numeric_dtype(data['sentiment_score']) and not data['sentiment_score'].dropna().empty else 0,
        'Average Impact Scores': avg_impact_scores,
        'Most Common Themes': data['key_themes'].explode().value_counts().head(5).to_dict() if 'key_themes' in data and data['key_themes'].apply(lambda x: isinstance(x, list)).any() and not data['key_themes'].explode().empty else {},
        'Most Common Concerns': most_common_concerns
    }
    
    # Save summary statistics to a text file
    with open(os.path.join(output_dir, f'analysis_summary_{timestamp}.txt'), 'w') as f:
        f.write('Analysis Summary Report\n')
        f.write('=====================\n\n')
        f.write(f'Total Personas: {summary_stats["Total Personas"]}\n')
        f.write(f'Average Sentiment Score: {summary_stats["Average Sentiment Score"]:.2f}\n\n')
        f.write('Average Impact Scores:\n')
        for area, score in summary_stats['Average Impact Scores'].items():
            f.write(f'  {area}: {score:.2f}\n')
        f.write('\nMost Common Themes:\n')
        for theme, count in summary_stats['Most Common Themes'].items():
            f.write(f'  {theme}: {count}\n')
        f.write('\nMost Common Concerns:\n')
        for concern, count in summary_stats['Most Common Concerns'].items():
            f.write(f'  {concern}: {count}\n') 