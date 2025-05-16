"""
View component for theme analysis.
"""
import streamlit as st
import plotly.express as px
import pandas as pd

def render_theme_analysis(controller) -> None:
    """Render the theme analysis interface."""
    st.header("Theme Analysis")
    
    if not controller.current_responses:
        st.warning("No responses available for analysis. Please generate personas first.")
        return
    
    # Get theme analysis
    analysis_results = controller.analyze_themes()
    if not analysis_results:
        st.error("Failed to analyze themes")
        return
    
    # Display theme distribution
    st.subheader("Theme Distribution")
    if 'theme_counts' in analysis_results:
        df = pd.DataFrame(analysis_results['theme_counts'])
        fig = px.bar(
            df,
            x='theme',
            y='count',
            title='Distribution of Canonical Themes',
            labels={'theme': 'Theme', 'count': 'Count'}
        )
        st.plotly_chart(fig)
    
    # Display theme mapping
    st.subheader("Theme Mapping")
    if 'theme_mapping' in analysis_results:
        st.write("Mapping between original and canonical themes:")
        for original, canonical in analysis_results['theme_mapping'].items():
            st.write(f"- **{original}** → {canonical}")
    
    # Display sentiment analysis
    st.subheader("Sentiment Analysis")
    if 'sentiment_by_theme' in analysis_results:
        df = pd.DataFrame(analysis_results['sentiment_by_theme'])
        fig = px.box(
            df,
            x='theme',
            y='sentiment',
            title='Sentiment Distribution by Theme',
            labels={'theme': 'Theme', 'sentiment': 'Sentiment Score'}
        )
        st.plotly_chart(fig)
    
    # Display impact analysis
    st.subheader("Impact Analysis")
    if 'impact_by_theme' in analysis_results:
        impact_df = pd.DataFrame(analysis_results['impact_by_theme'])
        for impact_type in ['housing', 'transport', 'community']:
            fig = px.box(
                impact_df,
                x='theme',
                y=f'impact_{impact_type}',
                title=f'Impact on {impact_type.title()} by Theme',
                labels={'theme': 'Theme', f'impact_{impact_type}': f'Impact Score'}
            )
            st.plotly_chart(fig) 