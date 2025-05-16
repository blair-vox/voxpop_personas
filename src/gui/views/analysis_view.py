"""
View component for general data analysis.
"""
import streamlit as st
import plotly.express as px
import pandas as pd

def render_analysis(controller) -> None:
    """Render the general analysis interface."""
    st.header("Data Analysis")
    
    if not controller.current_responses:
        st.warning("No responses available for analysis. Please generate personas first.")
        return
    
    # Convert responses to DataFrame
    data = []
    for response in controller.current_responses:
        data.append({
            'age': response.persona_details['age'],
            'gender': response.persona_details['gender'],
            'location': response.persona_details['location'],
            'income': response.persona_details['income'],
            'support_level': response.survey_response.support_level,
            'impact_housing': response.survey_response.impact_on_housing,
            'impact_transport': response.survey_response.impact_on_transport,
            'impact_community': response.survey_response.impact_on_community,
            'sentiment': response.sentiment_score
        })
    df = pd.DataFrame(data)
    
    # Demographics analysis
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        age_counts = df['age'].value_counts()
        fig = px.pie(
            values=age_counts.values,
            names=age_counts.index,
            title='Age Distribution'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Gender distribution
        gender_counts = df['gender'].value_counts()
        fig = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title='Gender Distribution'
        )
        st.plotly_chart(fig)
    
    # Location analysis
    st.subheader("Location Analysis")
    location_counts = df['location'].value_counts()
    fig = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        title='Distribution by Location',
        labels={'x': 'Location', 'y': 'Count'}
    )
    st.plotly_chart(fig)
    
    # Support and Impact Analysis
    st.subheader("Support and Impact Analysis")
    
    # Support level distribution
    support_counts = df['support_level'].value_counts().sort_index()
    fig = px.bar(
        x=support_counts.index,
        y=support_counts.values,
        title='Support Level Distribution',
        labels={'x': 'Support Level', 'y': 'Count'}
    )
    st.plotly_chart(fig)
    
    # Impact scores
    impact_cols = ['impact_housing', 'impact_transport', 'impact_community']
    impact_df = df[impact_cols].melt(
        var_name='Impact Type',
        value_name='Score'
    )
    fig = px.box(
        impact_df,
        x='Impact Type',
        y='Score',
        title='Impact Score Distribution'
    )
    st.plotly_chart(fig)
    
    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    fig = px.histogram(
        df,
        x='sentiment',
        title='Sentiment Score Distribution',
        labels={'sentiment': 'Sentiment Score'}
    )
    st.plotly_chart(fig)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    numeric_cols = ['support_level', 'impact_housing', 'impact_transport', 'impact_community', 'sentiment']
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        title='Correlation Matrix',
        labels=dict(x='Variable', y='Variable', color='Correlation')
    )
    st.plotly_chart(fig) 