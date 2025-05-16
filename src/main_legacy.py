"""
Main script for processing persona data and interacting with OpenAI API.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import csv

# Load environment variables
load_dotenv()

# --- Mapping Dictionaries ---
# Tenure
tenure = {
    1: "Own outright",
    2: "Own, paying off mortgage",
    3: "Rent from private landlord or real estate agent",
    4: "Rent from public housing authority",
    5: "Other (boarding, living at home, etc.)",
    999: "Item skipped"
}
# Job Type
job_type = {
    1: 'Upper managerial',
    2: 'Middle managerial',
    3: 'Lower managerial',
    4: 'Supervisory',
    5: 'Non-supervisory',
    999: 'Item skipped'
}
# Job Tenure
job_tenure = {
    1: 'Self-employed',
    2: 'Employee in private company or business',
    3: 'Employee of Federal / State / Local Government',
    4: 'Employee in family business or farm',
    999: 'Item skipped'
}
# Education Level
edu_level = {
    1: "Bachelor degree or higher",
    2: "Advanced diploma or diploma",
    3: "Certificate III/IV",
    4: "Year 12 or equivalent",
    5: "Year 11 or below",
    6: "No formal education",
    999: "Item skipped"
}
# Marital Status
marital_status = {
    1: "Never married",
    2: "Married",
    3: "Widowed",
    4: "Divorced/Separated",
    5: "De facto",
    999: "Item skipped"
}
# Partner Activity
partner_activity = {
    1: "Working full-time",
    2: "Working part-time",
    3: "Caring responsibilities",
    4: "Retired",
    5: "Student",
    6: "Unemployed",
    999: "Item skipped"
}
# Household Size
household_size = {
    1: "1 person",
    2: "2 people",
    3: "3 people",
    4: "4 people",
    5: "5 or more",
    999: "Item skipped"
}
# Family Payments
family_payments = {
    1: "Yes",
    2: "No",
    999: "Item skipped"
}
# Child Care Benefit
child_care_benefit = {
    1: "Yes",
    2: "No",
    999: "Item skipped"
}
# Investment Properties
investment_properties = {
    1: "Yes",
    2: "No",
    999: "Item skipped"
}
# Transport Infrastructure
transport_infrastructure = {
    1: "Much more than now",
    2: "Somewhat more than now",
    3: "The same as now",
    4: "Somewhat less than now",
    5: "Much less than now",
    999: "Item skipped"
}
# Political Leaning
political_leaning = {
    1: "Liberal",
    2: "Labor",
    3: "National Party",
    4: "Greens",
    **{i: "Other party" for i in range(5, 98)},
    999: "Item skipped"
}
# Trust in Government
trust_gov = {
    1: "Usually look after themselves",
    2: "Sometimes look after themselves",
    3: "Sometimes can be trusted to do the right thing",
    4: "Usually can be trusted to do the right thing",
    999: "Item skipped"
}
# Issues and Engagement Dictionaries (as in merge.py)
issues_dict = {
    "D1_1": "Taxation",
    "D1_2": "Immigration",
    "D1_3": "Education",
    "D1_4": "The environment",
    "D1_6": "Health and Medicare",
    "D1_7": "Refugees and asylum seekers",
    "D1_8": "Global warming",
    "D1_10": "Management of the economy",
    "D1_11": "The COVID-19 pandemic",
    "D1_12": "The cost of living",
    "D1_13": "National security"
}
engagement_dict = {
    "A1": "Interest in politics",
    "A2_1": "Attention to newspapers",
    "A2_2": "Attention to television",
    "A2_3": "Attention to radio",
    "A2_4": "Attention to internet",
    "A4_1": "Discussed politics in person",
    "A4_2": "Discussed politics online",
    "A4_3": "Persuaded others to vote",
    "A4_4": "Showed support for a party",
    "A4_5": "Attended political meetings",
    "A4_6": "Contributed money",
    "A7_1": "No candidate contact",
    "A7_2": "Contact by telephone",
    "A7_3": "Contact by mail",
    "A7_4": "Contact face-to-face",
    "A7_5": "Contact by SMS",
    "A7_6": "Contact by email",
    "A7_7": "Contact via social network"
}

def get_mapping_dict(variable_name):
    mapping_dict = {
        # Engagement Dictionary
        'A1': {1: 'A good deal', 2: 'Some', 3: 'Not much', 4: 'None', 999: 'Item skipped'},
        'A2_1': {1: 'A good deal', 2: 'Some', 3: 'Not much', 4: 'None at all', 999: 'Item skipped'},
        'A2_2': {1: 'A good deal', 2: 'Some', 3: 'Not much', 4: 'None at all', 999: 'Item skipped'},
        'A2_3': {1: 'A good deal', 2: 'Some', 3: 'Not much', 4: 'None at all', 999: 'Item skipped'},
        'A2_4': {1: 'A good deal', 2: 'Some', 3: 'Not much', 4: 'None at all', 999: 'Item skipped'},
        'A4_1': {1: 'Frequently', 2: 'Occasionally', 3: 'Rarely', 4: 'Not at all', 999: 'Item skipped'},
        'A4_2': {1: 'Frequently', 2: 'Occasionally', 3: 'Rarely', 4: 'Not at all', 999: 'Item skipped'},
        'A4_3': {1: 'Frequently', 2: 'Occasionally', 3: 'Rarely', 4: 'Not at all', 999: 'Item skipped'},
        'A4_4': {1: 'Frequently', 2: 'Occasionally', 3: 'Rarely', 4: 'Not at all', 999: 'Item skipped'},
        'A4_5': {1: 'Frequently', 2: 'Occasionally', 3: 'Rarely', 4: 'Not at all', 999: 'Item skipped'},
        'A4_6': {1: 'Frequently', 2: 'Occasionally', 3: 'Rarely', 4: 'Not at all', 999: 'Item skipped'},
        # Issues Dictionary
        'D1_1': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_2': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_3': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_4': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_6': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_7': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_8': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_10': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_11': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_12': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
        'D1_13': {1: 'Extremely important', 2: 'Quite important', 3: 'Not very important', 999: 'Item skipped'},
    }
    return mapping_dict.get(variable_name, {})

class SurveyResponse(BaseModel):
    """Model for structured survey responses."""
    support_level: int  # 1-5 scale
    impact_on_housing: int  # 1-5 scale
    impact_on_transport: int  # 1-5 scale
    impact_on_community: int  # 1-5 scale
    key_concerns: List[str]
    suggested_improvements: List[str]

class PersonaResponse(BaseModel):
    """Model for complete persona response including both narrative and survey data."""
    persona_details: Dict[str, Any]  # Changed from Dict[str, str] to Dict[str, Any] to handle lists
    narrative_response: str
    survey_response: SurveyResponse
    timestamp: str
    sentiment_score: Optional[float] = None
    key_themes: Optional[List[str]] = None

class Persona(BaseModel):
    """Pydantic model for persona data."""
    name: str
    age: str
    gender: str
    location: str
    income: str
    tenure: str
    job_tenure: str
    occupation: str
    education: str
    transport: str
    marital_status: str
    partner_activity: str
    household_size: str
    family_payments: str
    child_care_benefit: str
    investment_properties: str
    transport_infrastructure: str
    political_leaning: str
    trust: str
    issues: List[str]
    engagement: str

import csv
import numpy as np
import pandas as pd

def read_abs_data(file_path: str) -> pd.DataFrame:
    columns = [
        "SA2 (UR)",
        "AGE5P Age in Five Year Groups",
        "SEXP Sex",
        "FPIP Parent Indicator",
        "MSTP Registered Marital Status",
        "1-digit level ANCP Ancestry Multi Response",
        "INCP Total Personal Income (weekly)",
        "Negative income",
        "Nil income",
        "$1-$149 ($1-$7,799)",
        "$150-$299 ($7,800-$15,599)",
        "$300-$399 ($15,600-$20,799)",
        "$400-$499 ($20,800-$25,999)",
        "$500-$649 ($26,000-$33,799)",
        "$650-$799 ($33,800-$41,599)",
        "$800-$999 ($41,600-$51,999)",
        "$1,000-$1,249 ($52,000-$64,999)",
        "$1,250-$1,499 ($65,000-$77,999)",
        "$1,500-$1,749 ($78,000-$90,999)",
        "$1,750-$1,999 ($91,000-$103,999)",
        "$2,000-$2,999 ($104,000-$155,999)",
        "$3,000-$3,499 ($156,000-$181,999)",
        "$3,500 or more ($182,000 or more)",
        "Not stated",
        "Not applicable",
        "Total"
    ]
    df = pd.read_csv(
        file_path,
        skiprows=10,
        header=None,
        names=columns,
        encoding='utf-8',
        skipinitialspace=True,
        on_bad_lines='warn',
        quoting=csv.QUOTE_MINIMAL,
        sep=',',
        engine='python'
    )
    for col in columns[:6]:
        df[col] = df[col].replace('', np.nan).ffill()
    for col in columns[6:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(how='all', subset=columns[6:])
    print("Successfully loaded ABS dataset.")
    return df

def map_age_to_abs_age(age, abs_to_aes_age):
    for abs_cat, (min_age, max_age) in abs_to_aes_age.items():
        if pd.notnull(age) and min_age <= age <= max_age:
            return abs_cat
    return None

abs_to_aes_age = {
    "0-4 years" : (0, 4),
    "5-9 years" : (5, 9),
    "10-14 years" : (10, 14),
    "15-19 years" : (15, 19),
    "20-24 years": (20, 24),
    "25-29 years": (25, 29),
    "30-34 years": (30, 34),
    "35-39 years": (35, 39),
    "40-44 years": (40, 44),
    "45-49 years": (45, 49),
    "50-54 years": (50, 54),
    "55-59 years": (55, 59),
    "60-64 years": (60, 64),
    "65-69 years": (65, 69),
    "70-74 years": (70, 74),
    "75-79 years": (75, 79),
    "80-84 years": (80, 84),
    "85-89 years": (85, 89),
    "90-95 years": (90, 95),
    "96-99 years": (96, 99),
    "100 years and over": (100, 120)
}

abs_to_aes_sex = {
    "Male" : 1,
    "Female" : 2,
}

abs_to_aes_marital = {
    "Never married" : 1,
    "Married" : 2,
    "Separated" : 4,
    "Widowed" : 3,
    "Divorced" : 4
}

def map_income_to_abs_income(aes_income, abs_to_aes_income):
    for abs_cat, aes_val in abs_to_aes_income.items():
        if isinstance(aes_val, tuple):
            if pd.notnull(aes_income) and aes_val[0] <= aes_income <= aes_val[1]:
                return abs_cat
        elif aes_income == aes_val:
            return abs_cat
    return None

abs_to_aes_income = {
    'Negative income' : 1,
    'Nil income' : 1, 
    '$1-$149 ($1-$7,799)' : 1, 
    '$150-$299 ($7,800-$15,599)' : 1,
    '$300-$399 ($15,600-$20,799)' : 3,
    '$400-$499 ($20,800-$25,999)' : 3,
    '$500-$649 ($26,000-$33,799)' : 5,
    '$650-$799 ($33,800-$41,599)' : 7,
    '$800-$999 ($41,600-$51,999)' : 9,
    '$1,000-$1,249 ($52,000-$64,999)' : 9,
    '$1,250-$1,499 ($65,000-$77,999)' : 11,
    '$1,500-$1,749 ($78,000-$90,999)' : 13,
    '$1,750-$1,999 ($91,000-$103,999)' : 13,
    '$2,000-$2,999 ($104,000-$155,999)' : (15, 17),
    '$3,000-$3,499 ($156,000-$181,999)' : 19,
    '$3,500 or more ($182,000 or more)' : (21, 25),
    'Not stated' : 999,
    'Not applicable' : 999
}

def sample_personas_major_city(
    merged, N,
    strata_cols=["ABS_AGE_CATEGORY", "ABS_SEX", "ABS_MARITAL", "ABS_INCOME"],
    major_city_col="J5",
    major_city_value=5
):
    """
    Sample N personas from the merged frame, ensuring each persona is from a major city (J5 == 5)
    and excluding any personas with 'Total' location.
    """
    strata_weights = merged.groupby(strata_cols)['weight'].first().reset_index()
    strata_probs = strata_weights['weight'] / strata_weights['weight'].sum()
    personas = []
    attempts = 0
    max_attempts = N * 10
    while len(personas) < N and attempts < max_attempts:
        stratum = strata_weights.sample(
            n=1,
            weights=strata_probs
        ).iloc[0]
        matches = merged
        for col in strata_cols:
            matches = matches[matches[col] == stratum[col]]
        matches = matches[matches[major_city_col] == major_city_value]
        # Exclude matches with 'Total' location
        matches = matches[matches['SA2 (UR)'] != 'Total']
        if not matches.empty:
            persona = matches.sample(n=1)
            personas.append(persona)
        attempts += 1
    if len(personas) < N:
        print(f"Warning: Only {len(personas)} personas could be sampled with {major_city_col} == {major_city_value} after {max_attempts} attempts.")
    if personas:
        personas_df = pd.concat(personas, ignore_index=True)
    else:
        personas_df = pd.DataFrame()
    return personas_df

def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of text using TextBlob.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Sentiment score between -1 and 1
    """
    return TextBlob(text).sentiment.polarity

def extract_key_themes(text: str, client: OpenAI) -> List[str]:
    """
    Extract key themes from text using OpenAI.
    
    Args:
        text (str): Text to analyze
        client (OpenAI): OpenAI client instance
        
    Returns:
        List[str]: List of key themes
    """
    prompt = f"""Extract the 3-5 most important themes or topics from the following text. 
    Return them as a comma-separated list of short phrases:

    {text}"""
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts key themes from text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    themes = response.choices[0].message.content.strip().split(',')
    return [theme.strip() for theme in themes]

def create_heatmap(data: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Create heatmap of support levels by demographic groups.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
    """
    # Create pivot table for support levels
    pivot_data = data.pivot_table(
        values='support_level',
        index='political_leaning',
        columns='age',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    # Create heatmap with explicit colorbar
    sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=3, 
                cbar_kws={'label': 'Support Level\n(1=Strongly Oppose, 3=Neutral, 5=Strongly Support)'})
    plt.title('Support Level Heatmap by Age and Political Leaning')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'support_heatmap_{timestamp}.png'))
    plt.close()

def create_sentiment_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Create sentiment analysis visualizations.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
    """
    # Convert sentiment scores to -100 to +100 scale
    data['sentiment_percentage'] = (data['sentiment_score'] * 100).round(0)
    
    # Extract numeric age from age range (e.g., "45-49 years" -> 45)
    data['age_numeric'] = data['age'].str.extract(r'(\d+)').astype(int)
    
    # Sentiment by political leaning
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='political_leaning', y='sentiment_percentage', data=data)
    plt.title('Sentiment Distribution by Political Leaning')
    plt.ylabel('Sentiment (-100% to +100%)\n(-100%=Very Negative, 0%=Neutral, +100%=Very Positive)')
    plt.ylim(-100, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sentiment_by_politics_{timestamp}.png'))
    plt.close()
    
    # Sentiment by age group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='age', y='sentiment_percentage', data=data.sort_values('age_numeric'))
    plt.title('Sentiment Distribution by Age')
    plt.ylabel('Sentiment (-100% to +100%)\n(-100%=Very Negative, 0%=Neutral, +100%=Very Positive)')
    plt.ylim(-100, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sentiment_by_age_{timestamp}.png'))
    plt.close()

def create_impact_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Create impact analysis visualizations.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
    """
    # Impact scores by demographic
    impact_cols = ['impact_on_housing', 'impact_on_transport', 'impact_on_community']
    
    plt.figure(figsize=(12, 6))
    data[impact_cols].mean().plot(kind='bar')
    plt.title('Average Impact Scores by Category')
    plt.ylabel('Average Score (1-5)\n(1=Very Negative, 3=Neutral, 5=Very Positive)')
    plt.xticks(rotation=45)
    plt.ylim(0, 5)  # Set y-axis limits to match the scale
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'impact_scores_{timestamp}.png'))
    plt.close()

def create_theme_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Create theme analysis visualizations.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
    """
    # Count theme occurrences
    all_themes = []
    for themes in data['key_themes']:
        # Split the string by commas and strip any extra whitespace
        theme_list = [theme.strip() for theme in themes.split(',')]
        all_themes.extend(theme_list)
    
    theme_counts = pd.Series(all_themes).value_counts()
    
    # Plot top themes
    plt.figure(figsize=(12, 6))
    theme_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Key Themes')
    plt.ylabel('Number of Mentions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top_themes_{timestamp}.png'))
    plt.close()

def create_location_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Create analysis of responses by location.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
    """
    # Support level by location
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='location', y='support_level', data=data)
    plt.title('Support Level Distribution by Location')
    plt.ylabel('Support Level (1-5)\n(1=Strongly Oppose, 3=Neutral, 5=Strongly Support)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'support_by_location_{timestamp}.png'))
    plt.close()
    
    # Sentiment by location
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='location', y='sentiment_percentage', data=data)
    plt.title('Sentiment Distribution by Location')
    plt.ylabel('Sentiment (-100% to +100%)\n(-100%=Very Negative, 0%=Neutral, +100%=Very Positive)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sentiment_by_location_{timestamp}.png'))
    plt.close()

def create_occupation_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Create analysis of responses by occupation.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
    """
    # Support level by occupation
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='occupation', y='support_level', data=data)
    plt.title('Support Level Distribution by Occupation')
    plt.ylabel('Support Level (1-5)\n(1=Strongly Oppose, 3=Neutral, 5=Strongly Support)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'support_by_occupation_{timestamp}.png'))
    plt.close()
    
    # Sentiment by occupation
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='occupation', y='sentiment_percentage', data=data)
    plt.title('Sentiment Distribution by Occupation')
    plt.ylabel('Sentiment (-100% to +100%)\n(-100%=Very Negative, 0%=Neutral, +100%=Very Positive)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sentiment_by_occupation_{timestamp}.png'))
    plt.close()

def create_correlation_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Create correlation analysis between different metrics.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
    """
    # Create correlation matrix
    numeric_cols = ['support_level', 'impact_on_housing', 'impact_on_transport', 
                   'impact_on_community', 'sentiment_percentage']
    corr_matrix = data[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Response Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'correlation_matrix_{timestamp}.png'))
    plt.close()

def create_concern_analysis(data: pd.DataFrame, output_dir: str, timestamp: str):
    """
    Create analysis of key concerns by demographic groups.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
        timestamp (str): Timestamp for file naming
    """
    # Extract all concerns, handling NaN values
    all_concerns = []
    for concerns in data['key_concerns']:
        if pd.notna(concerns):  # Check if the value is not NaN
            concerns_list = concerns.split(', ')
            all_concerns.extend(concerns_list)
    
    # Get top 5 concerns
    top_concerns = pd.Series(all_concerns).value_counts().head(5).index
    
    # Create subplots for each top concern
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, concern in enumerate(top_concerns):
        # Calculate percentage of each group mentioning this concern
        concern_data = []
        for group in ['political_leaning', 'location', 'occupation']:
            group_concerns = data.groupby(group)['key_concerns'].apply(
                lambda x: sum(pd.notna(x) and concern in ' '.join(x.split(', ')) for x in x) / len(x) * 100
            )
            concern_data.append(group_concerns)
        
        # Plot
        ax = axes[i]
        concern_data[0].plot(kind='bar', ax=ax)
        ax.set_title(f'% Mentioning: {concern}')
        ax.set_ylabel('Percentage')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'concern_analysis_{timestamp}.png'))
    plt.close()

def generate_analysis_report(data: pd.DataFrame, output_dir: str, timestamp: str) -> None:
    """
    Generate a comprehensive analysis report of the responses.
    
    Args:
        data (pd.DataFrame): DataFrame containing all responses
        output_dir (str): Directory to save the report
        timestamp (str): Timestamp for file naming
    """
    report_path = os.path.join(output_dir, f'analysis_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("Bondi Parking Policy Proposal - Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall support statistics
        f.write("Overall Support Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Support Level: {data['support_level'].mean():.2f}\n")
        f.write(f"Support Distribution:\n{data['support_level'].value_counts().sort_index()}\n\n")
        
        # Analysis by political leaning
        f.write("Analysis by Political Leaning:\n")
        f.write("-" * 30 + "\n")
        group_stats = data.groupby('political_leaning', observed=True)['support_level'].mean()
        for group, mean_support in group_stats.items():
            f.write(f"{group}: {mean_support:.2f}\n")
        f.write("\n")
        
        # Analysis by age group
        f.write("Analysis by Age Group:\n")
        f.write("-" * 30 + "\n")
        # Extract the first number from age ranges (e.g., "45-49 years" -> 45)
        data['age_numeric'] = data['age'].str.extract(r'(\d+)').astype(float)
        data['age_group'] = pd.cut(data['age_numeric'], 
                                  bins=[0, 30, 50, 70, 100],
                                  labels=['18-30', '31-50', '51-70', '70+'])
        age_stats = data.groupby('age_group', observed=True)['support_level'].mean()
        for group, mean_support in age_stats.items():
            f.write(f"{group}: {mean_support:.2f}\n")
        f.write("\n")
        
        # Key themes and concerns
        f.write("Key Themes and Concerns:\n")
        f.write("-" * 30 + "\n")
        all_themes = [theme for themes in data['key_themes'] for theme in themes]
        theme_counts = pd.Series(all_themes).value_counts()
        f.write("Most Common Themes:\n")
        for theme, count in theme_counts.head(5).items():
            f.write(f"- {theme}: {count} mentions\n")
        f.write("\n")
        
        # Impact analysis
        f.write("Impact Analysis:\n")
        f.write("-" * 30 + "\n")
        impact_cols = ['impact_on_housing', 'impact_on_transport', 'impact_on_community']
        for col in impact_cols:
            f.write(f"\n{col.replace('_', ' ').title()}:\n")
            impact_stats = data[col].value_counts().sort_index()
            for level, count in impact_stats.items():
                f.write(f"Level {level}: {count} responses\n")
        
        # Sentiment analysis
        f.write("\nSentiment Analysis:\n")
        f.write("-" * 30 + "\n")
        # Convert sentiment scores to -100 to +100 scale
        sentiment_percentage = (data['sentiment_score'] * 100).round(0)
        f.write(f"Average Sentiment: {sentiment_percentage.mean():.1f}% (negative to positive)\n")
        f.write(f"Sentiment Distribution:\n")
        sentiment_stats = pd.cut(sentiment_percentage, 
                               bins=[-100, -50, 0, 50, 100],
                               labels=['Very Negative (-100% to -50%)', 'Negative (-49% to 0%)', 'Positive (1% to 50%)', 'Very Positive (51% to 100%)'])
        for sentiment, count in sentiment_stats.value_counts().items():
            f.write(f"{sentiment}: {count} responses\n")

def load_personas(file_path: str) -> List[Persona]:
    """
    Load and parse persona data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing persona data
        
    Returns:
        List[Persona]: List of Persona objects
    """
    df = pd.read_csv(file_path)
    personas = []
    
    for _, row in df.iterrows():
        persona = Persona(
            name=row['name'],
            age=str(row['age']),  # Convert age to string
            gender=row['gender'],
            location=row['location'],
            income=row['income'],
            tenure=str(row['tenure']),
            job_tenure=str(row['job_tenure']),
            occupation=row['occupation'],
            education=row['education'],
            transport=row['transport'],
            marital_status=str(row['marital_status']),
            partner_activity=str(row['partner_activity']),
            household_size=str(row['household_size']),
            family_payments=str(row['family_payments']),
            child_care_benefit=str(row['child_care_benefit']),
            investment_properties=str(row['investment_properties']),
            transport_infrastructure=str(row['transport_infrastructure']),
            political_leaning=str(row['political_leaning']),
            trust=str(row['trust']),
            issues=row['issues'].split(', ') if isinstance(row['issues'], str) else [],
            engagement=str(row['engagement'])
        )
        personas.append(persona)
    
    return personas

def parse_survey_response(response_text: str) -> SurveyResponse:
    """
    Parse the structured survey response from the LLM output.
    
    Args:
        response_text (str): The survey response section from the LLM output
        
    Returns:
        SurveyResponse: Parsed survey response
    """
    # Extract numbers and lists from the response
    lines = response_text.strip().split('\n')
    survey_data = {}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            # Map the keys to match our model
            key_mapping = {
                'support_level': 'support_level',
                'impact on housing': 'impact_on_housing',
                'impact on transport': 'impact_on_transport',
                'impact on community': 'impact_on_community',
                'key concerns': 'key_concerns',
                'suggested improvements': 'suggested_improvements'
            }
            
            # Get the correct key name
            model_key = key_mapping.get(key, key)
            
            if model_key in ['support_level', 'impact_on_housing', 'impact_on_transport', 'impact_on_community']:
                # Extract the first number from the line
                try:
                    survey_data[model_key] = int(''.join(filter(str.isdigit, value)))
                except ValueError:
                    survey_data[model_key] = 0
            elif model_key in ['key_concerns', 'suggested_improvements']:
                # Split the comma-separated list and clean up
                items = [item.strip() for item in value.split(',')]
                survey_data[model_key] = items
    
    # Ensure all required fields are present
    required_fields = {
        'support_level': 0,
        'impact_on_housing': 0,
        'impact_on_transport': 0,
        'impact_on_community': 0,
        'key_concerns': [],
        'suggested_improvements': []
    }
    
    # Fill in any missing fields with defaults
    for field, default in required_fields.items():
        if field not in survey_data:
            survey_data[field] = default
    
    return SurveyResponse(**survey_data)

def generate_persona_response(persona: Persona, client: OpenAI) -> PersonaResponse:
    """
    Generate a persona's response to the local proposal using OpenAI API.
    
    Args:
        persona (Persona): Persona object containing demographic information
        client (OpenAI): OpenAI client instance
        
    Returns:
        PersonaResponse: Complete response including both narrative and survey data
    """
    # Build persona details string, excluding Unknown and Item skipped values
    persona_details = []
    for field, value in [
        ("Name", persona.name),
        ("Age", persona.age),
        ("Gender", persona.gender),
        ("Location", f"{persona.location}, NSW"),
        ("Income", persona.income),
        ("Tenure", persona.tenure),
        ("Job Tenure", persona.job_tenure),
        ("Occupation", persona.occupation),
        ("Education", persona.education),
        ("Transport Mode", persona.transport),
        ("Marital Status", persona.marital_status),
        ("Partner Activity", persona.partner_activity),
        ("Household Size", persona.household_size),
        ("Family Payments", persona.family_payments),
        ("Child Care Benefit", persona.child_care_benefit),
        ("Investment Properties", persona.investment_properties),
        ("Transport Infrastructure", persona.transport_infrastructure),
        ("Political Leaning", persona.political_leaning),
        ("Trust in Government", persona.trust),
        ("Key Issues", ', '.join(persona.issues)),
        ("Engagement Level", persona.engagement)
    ]:
        if value and value not in ["Unknown", "Item skipped"]:
            persona_details.append(f"- {field}: {value}")

    prompt = f"""You are simulating the response of a fictional but demographically grounded persona for use in a synthetic civic focus group. This persona is based on Australian Census data and local electoral trends. 
    You should use the language of the persona your are simulating. 
    Consider the tone and language of the persona you are simulating. 
    Consider the issues that the persona you are simulating is concerned about. 
    What are their life circumstances? How would it impact their current lifestyle?
    If they have family, how would this policy impact them?

Persona Details:
{chr(10).join(persona_details)}

You have been asked to react to the following **local proposal**:

> "Waverley Council is considering a policy that would remove minimum parking requirements for new apartment developments in Bondi. This means developers could build fewer or no car spaces if they believe it suits the residents' needs."

IMPORTANT: Based on your Australian demographic profile, you should take a strong position on this issue. 
Consider how your background might lead you to your views, you are free to be as moderate or extreme as you like:

- If you're a car-dependent commuter, you might strongly oppose this policy
- If you're a young renter who doesn't own a car, you might strongly support it
- If you're concerned about housing affordability, you might see this as a crucial step
- If you're worried about parking in your neighborhood, you might see this as a major threat
- If you're environmentally conscious, you might view this as essential for sustainability
- If you're a property owner, you might be concerned about impacts on property values
- If you have investment properties, you might be concerned about property values
- If you have children and receive family payments, you might be concerned about housing affordability
- If you're in a larger household, you might be more concerned about parking availability
- If you're retired, you might be more concerned about community impact

BACKGROUND:
	Liberal Party: A center-right party advocating for free-market policies, individual liberties, and limited government intervention.
	Labor Party: A center-left party focused on social justice, workers' rights, and government involvement in healthcare and education.
    National Party: A conservative, rural-focused party promoting agricultural interests and regional development.
	Greens: A progressive party emphasizing environmental protection, social equality, and climate action.
	One Nation: A right-wing nationalist party advocating for stricter immigration controls and Australian sovereignty.

Please provide:

1. A short narrative response (2-3 sentences) that reflects:
   - A clear, strong position on the policy (either strongly support or strongly oppose)
   - Why — in your own words, as someone with this background
   - What specific impacts you're most concerned about

2. A structured survey response with the following:
   - Support Level (1-5, where 1 is strongly oppose and 5 is strongly support)
   - Impact on Housing Affordability (1-5, where 1 is very negative and 5 is very positive)
   - Impact on Transport (1-5, where 1 is very negative and 5 is very positive)
   - Impact on Community (1-5, where 1 is very negative and 5 is very positive)
   - Key Concerns (comma-separated list)
   - Suggested Improvements (comma-separated list)

Where possible, reflect your opinion using tone and language that matches your demographic and issue profile. Don't be afraid to take strong positions based on your background.

**Relevant local context for grounding:**
- Reddit post (r/sydney, 2023): "There are too many ghost garages in Bondi. We need more housing, not car spots."
- Grattan Institute: 68% of renters under 35 support relaxed parking minimums in high-transit areas.
- AEC data: Bondi Junction booths saw 30%+ Green vote among young renters in 2022.
- Local resident group: "Parking is already a nightmare. This will make it worse."
- Property developer: "Parking minimums add $50,000+ to each apartment's cost."

Format your response as follows:

NARRATIVE RESPONSE:
[Your narrative response here]

SURVEY RESPONSE:
Support Level: [1-5]
Impact on Housing: [1-5]
Impact on Transport: [1-5]
Impact on Community: [1-5]
Key Concerns: [comma-separated list]
Suggested Improvements: [comma-separated list]"""

    # Print the prompt for the first persona only
    if persona.name == "Persona_0":
        print("\n=== First Prompt Sent to OpenAI ===")
        print(prompt)
        print("=== End of Prompt ===\n")

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that simulates realistic persona responses based on demographic data. You should encourage strong, diverse viewpoints based on demographic factors."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7  # Add some variation to responses
    )
    response_text = response.choices[0].message.content
    # Split the response into narrative and survey parts
    parts = response_text.split('SURVEY RESPONSE:')
    narrative = parts[0].replace('NARRATIVE RESPONSE:', '').strip()
    survey = parts[1].strip() if len(parts) > 1 else ""
    # Parse the survey response
    survey_response = parse_survey_response(survey)
    # Analyze sentiment and extract themes
    sentiment_score = analyze_sentiment(narrative)
    key_themes = extract_key_themes(narrative, client)
    # Return all persona details in persona_details
    return PersonaResponse(
        persona_details={
            "name": persona.name,
            "age": persona.age,
            "gender": getattr(persona, 'gender', ''),
            "location": persona.location,
            "income": getattr(persona, 'income', ''),
            "tenure": getattr(persona, 'tenure', ''),
            "job_tenure": getattr(persona, 'job_tenure', ''),
            "occupation": getattr(persona, 'occupation', ''),
            "education": getattr(persona, 'education', ''),
            "transport": getattr(persona, 'transport', ''),
            "marital_status": getattr(persona, 'marital_status', ''),
            "partner_activity": getattr(persona, 'partner_activity', ''),
            "household_size": getattr(persona, 'household_size', ''),
            "family_payments": getattr(persona, 'family_payments', ''),
            "child_care_benefit": getattr(persona, 'child_care_benefit', ''),
            "investment_properties": getattr(persona, 'investment_properties', ''),
            "transport_infrastructure": getattr(persona, 'transport_infrastructure', ''),
            "political_leaning": getattr(persona, 'political_leaning', ''),
            "trust": getattr(persona, 'trust', ''),
            "issues": persona.issues,
            "engagement": getattr(persona, 'engagement', '')
        },
        narrative_response=narrative,
        survey_response=survey_response,
        timestamp=datetime.now().isoformat(),
        sentiment_score=sentiment_score,
        key_themes=key_themes
    )

def save_responses(responses: List[PersonaResponse], output_dir: str = "output"):
    """
    Save the responses to a single JSON file and generate analysis.
    
    Args:
        responses (List[PersonaResponse]): List of persona responses
        output_dir (str): Directory to save the responses
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filepath = os.path.join(output_dir, f"responses_{timestamp}.json")
    with open(json_filepath, 'w') as f:
        json.dump([response.model_dump() for response in responses], f, indent=2)
    # Create DataFrame for analysis
    summary_data = []
    for response in responses:
        persona = response.persona_details
        # Add all persona fields if present
        summary_data.append({
            "name": persona.get("name", ""),
            "age": persona.get("age", ""),
            "gender": persona.get("gender", ""),
            "location": persona.get("location", ""),
            "income": persona.get("income", ""),
            "tenure": persona.get("tenure", ""),
            "job_tenure": persona.get("job_tenure", ""),
            "occupation": persona.get("occupation", ""),
            "education": persona.get("education", ""),
            "transport": persona.get("transport", ""),
            "marital_status": persona.get("marital_status", ""),
            "partner_activity": persona.get("partner_activity", ""),
            "household_size": persona.get("household_size", ""),
            "family_payments": persona.get("family_payments", ""),
            "child_care_benefit": persona.get("child_care_benefit", ""),
            "investment_properties": persona.get("investment_properties", ""),
            "transport_infrastructure": persona.get("transport_infrastructure", ""),
            "political_leaning": persona.get("political_leaning", ""),
            "trust": persona.get("trust", ""),
            "issues": ', '.join(persona.get("issues", [])),
            "engagement": persona.get("engagement", ""),
            "support_level": response.survey_response.support_level,
            "impact_on_housing": response.survey_response.impact_on_housing,
            "impact_on_transport": response.survey_response.impact_on_transport,
            "impact_on_community": response.survey_response.impact_on_community,
            "key_concerns": ", ".join(response.survey_response.key_concerns),
            "suggested_improvements": ", ".join(response.survey_response.suggested_improvements),
            "sentiment_score": response.sentiment_score,
            "key_themes": ", ".join(response.key_themes) if response.key_themes else ""
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, f"responses_summary_{timestamp}.csv"), index=False)
    # Save personas as well
    persona_objects = []
    for response in responses:
        # Try to reconstruct Persona from persona_details
        persona_dict = response.persona_details.copy()
        persona_dict['issues'] = persona_dict.get('issues', [])
        persona_objects.append(Persona(**persona_dict))
    save_personas_csv(persona_objects, output_dir, timestamp)
    # Generate visualizations and analysis
    create_heatmap(summary_df, output_dir, timestamp)
    create_sentiment_analysis(summary_df, output_dir, timestamp)
    create_impact_analysis(summary_df, output_dir, timestamp)
    create_theme_analysis(summary_df, output_dir, timestamp)
    create_location_analysis(summary_df, output_dir, timestamp)
    create_occupation_analysis(summary_df, output_dir, timestamp)
    create_correlation_analysis(summary_df, output_dir, timestamp)
    create_concern_analysis(summary_df, output_dir, timestamp)
    generate_analysis_report(summary_df, output_dir, timestamp)
    return timestamp

def regenerate_plots(data_file: str, output_dir: str = "output"):
    """
    Regenerate plots from an existing data file.
    
    Args:
        data_file (str): Path to the CSV data file
        output_dir (str): Directory to save the plots
    """
    # Extract timestamp from filename
    timestamp = data_file.split('_')[-1].replace('.csv', '')
    
    # Ensure timestamp is in the correct format
    if len(timestamp) == 6:  # If timestamp is in YYMMDD format
        timestamp = f"20{timestamp}"  # Add century prefix
    
    # Load data
    data = pd.read_csv(data_file)
    
    # Convert key_themes from string to list
    data['key_themes'] = data['key_themes'].apply(eval)
    
    # Generate visualizations
    create_heatmap(data, output_dir, timestamp)
    create_sentiment_analysis(data, output_dir, timestamp)
    create_impact_analysis(data, output_dir, timestamp)
    create_theme_analysis(data, output_dir, timestamp)
    create_location_analysis(data, output_dir, timestamp)
    create_occupation_analysis(data, output_dir, timestamp)
    create_correlation_analysis(data, output_dir, timestamp)
    create_concern_analysis(data, output_dir, timestamp)
    generate_analysis_report(data, output_dir, timestamp)
    
    print(f"\nPlots regenerated with timestamp: {timestamp}")

def generate_poststrat_frame(synthetic_personas: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a post-stratification frame from synthetic personas.

    Parameters:
        synthetic_personas (DataFrame): Synthetic personas data.

    Returns:
        DataFrame: Post-stratification frame with demographic weights.
    """
    # Group by key demographic variables and calculate weighted counts
    poststrat_frame = (synthetic_personas.groupby(["AGE", "H1", "STATE", "J6", "B1"])
                      .agg(weighted_count=("weight_final", "sum"))
                      .reset_index())
    # Normalize weights to sum to 1
    poststrat_frame["weight"] = poststrat_frame["weighted_count"] / poststrat_frame["weighted_count"].sum()
    return poststrat_frame

def generate_synthetic_personas(aes_file: str, abs_file: str, num_personas: int = 10, sample_size: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic personas by merging AES and ABS toy datasets.

    Parameters:
        aes_file (str): Path to the AES toy data CSV file.
        abs_file (str): Path to the ABS toy data CSV file.
        num_personas (int): Number of synthetic personas to generate.
        sample_size (int): Number of rows to sample from AES data.

    Returns:
        tuple: (synthetic_personas DataFrame, poststrat_frame DataFrame)
    """
    # Load AES and ABS data
    aes_data = pd.read_csv(aes_file)
    abs_data = pd.read_csv(abs_file)

    # Sample AES data to reduce size
    toy_data = aes_data.sample(n=sample_size, random_state=42)

    # Select relevant columns
    selected_cols = ["AGE", "H1", "STATE", "J6", "B9_1", "B1", "weight_final"]
    toy_data = toy_data[selected_cols]

    # Merge AES and ABS data on common demographic variables
    merged_data = pd.merge(toy_data, abs_data,
                           left_on=["STATE", "AGE", "H1", "J6"],
                           right_on=["State", "Age_Group", "Gender", "Income_Level"],
                           how="inner")

    # Drop redundant columns after merging
    merged_data = merged_data.drop(columns=["State", "Age_Group", "Gender", "Income_Level"])

    # Calculate weighted population for sampling
    merged_data["Weighted_Pop"] = merged_data["weight_final"] * merged_data["Population"]

    # Generate synthetic personas using weighted sampling
    synthetic_personas = merged_data.sample(n=num_personas, weights="Weighted_Pop", replace=True)

    # Generate post-stratification frame
    poststrat_frame = generate_poststrat_frame(synthetic_personas)

    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Save the synthetic personas and post-stratification frame
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    personas_file = os.path.join(output_dir, f"synthetic_personas_{timestamp}.csv")
    poststrat_file = os.path.join(output_dir, f"poststrat_frame_{timestamp}.csv")
    
    synthetic_personas.to_csv(personas_file, index=False)
    poststrat_frame.to_csv(poststrat_file, index=False)

    # Visualize the distribution of political affiliation
    plt.figure(figsize=(10, 6))
    sns.countplot(data=synthetic_personas, x="B1")
    plt.title("Distribution of Political Affiliation in Synthetic Personas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f"political_distribution_{timestamp}.png")
    plt.savefig(plot_file)
    plt.close()

    # Create and save demographic distribution plots
    plt.figure(figsize=(15, 10))
    
    # Age distribution
    plt.subplot(2, 2, 1)
    sns.countplot(data=synthetic_personas, x="AGE")
    plt.title("Age Distribution")
    plt.xticks(rotation=45)
    
    # Gender distribution
    plt.subplot(2, 2, 2)
    sns.countplot(data=synthetic_personas, x="H1")
    plt.title("Gender Distribution")
    plt.xticks(rotation=45)
    
    # State distribution
    plt.subplot(2, 2, 3)
    sns.countplot(data=synthetic_personas, x="STATE")
    plt.title("State Distribution")
    plt.xticks(rotation=45)
    
    # Income distribution
    plt.subplot(2, 2, 4)
    sns.countplot(data=synthetic_personas, x="J6")
    plt.title("Income Distribution")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    demo_plot_file = os.path.join(output_dir, f"demographic_distribution_{timestamp}.png")
    plt.savefig(demo_plot_file)
    plt.close()

    print(f"\nGenerated {num_personas} synthetic personas")
    print(f"Personas data saved to: {personas_file}")
    print(f"Post-stratification frame saved to: {poststrat_file}")
    print(f"Political distribution plot saved to: {plot_file}")
    print(f"Demographic distribution plot saved to: {demo_plot_file}")

    return synthetic_personas, poststrat_frame

def create_toy_datasets(aes_file: str = "toy_aes_data.csv", abs_file: str = "toy_abs_data.csv", 
                       aes_input: str = "data/aes22_unrestricted_v3.csv", abs_input: str = "data/Personas_wide.csv", 
                       num_rows: int = 100) -> tuple[str, str]:
    """
    Create toy AES and ABS datasets by sampling from the large input datasets.

    Parameters:
        aes_file (str): Path to save the toy AES data CSV file.
        abs_file (str): Path to save the toy ABS data CSV file.
        aes_input (str): Path to the large AES data file.
        abs_input (str): Path to the large ABS data file.
        num_rows (int): Number of rows to sample from each dataset.

    Returns:
        tuple: (path to toy AES file, path to toy ABS file)
    """
    # Check if the input files exist
    if not os.path.exists(aes_input):
        print("Error: AES input file not found.")
        return None, None

    # Read the AES data file
    try:
        # Read AES data with proper column handling and error handling
        aes_data = pd.read_csv(aes_input, 
                             skipinitialspace=True,
                             low_memory=False,  # Handle mixed types
                             quoting=csv.QUOTE_MINIMAL,  # Handle quoted fields
                             on_bad_lines='warn')  # Warn about problematic lines
        
        # Clean column names
        aes_data.columns = aes_data.columns.str.strip()
        
        print("Successfully loaded AES dataset.")
        print(f"AES columns: {aes_data.columns.tolist()}")
        
    except Exception as e:
        print(f"Error reading AES file: {e}")
        print("\nTrying alternative reading method...")
        try:
            # Try reading with different parameters
            aes_data = pd.read_csv(aes_input, 
                                 skipinitialspace=True,
                                 low_memory=False,
                                 encoding='utf-8',
                                 engine='python')  # Use python engine which is more flexible
            
            print("Successfully loaded AES dataset using alternative method.")
            print(f"AES columns: {aes_data.columns.tolist()}")
            
        except Exception as e2:
            print(f"Error with alternative reading method: {e2}")
            return None, None

    # Verify required AES columns exist
    required_aes_cols = ["AGE", "H1", "STATE", "J6", "B1", "weight_final"]
    missing_aes = [col for col in required_aes_cols if col not in aes_data.columns]
    if missing_aes:
        print(f"Missing required columns in AES data: {missing_aes}")
        return None, None

    # Sample the AES data
    aes_sample = aes_data.sample(n=min(num_rows, len(aes_data)), random_state=42)

    # Create a simplified ABS dataset with demographic categories
    abs_sample = pd.DataFrame({
        'State': ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'],
        'Age_Group': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'Gender': ['Male', 'Female'],
        'Income_Level': ['Low', 'Medium', 'High'],
        'Population': [1000] * 8  # Equal weights for simplicity
    })

    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Add timestamp to filenames to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aes_file = os.path.join(data_dir, aes_file.replace(".csv", f"_{timestamp}.csv"))
    abs_file = os.path.join(data_dir, abs_file.replace(".csv", f"_{timestamp}.csv"))

    # Save the sampled datasets
    aes_sample.to_csv(aes_file, index=False)
    abs_sample.to_csv(abs_file, index=False)
    print(f"Toy AES and ABS datasets created and saved as:")
    print(f"AES: {aes_file}")
    print(f"ABS: {abs_file}")

    return aes_file, abs_file

def draw_weighted_personas(poststrat_frame: pd.DataFrame, num_personas: int = 10) -> pd.DataFrame:
    """
    Draw weighted personas from the post-stratification frame.
    
    Args:
        poststrat_frame (pd.DataFrame): Post-stratification frame with weights
        num_personas (int): Number of personas to draw
        
    Returns:
        pd.DataFrame: Drawn personas with demographic attributes
    """
    # Draw personas using the weights
    drawn_personas = poststrat_frame.sample(
        n=num_personas,
        weights='weight',
        replace=True,
        random_state=42
    )
    
    return drawn_personas

def validate_demographic_distribution(personas: pd.DataFrame) -> None:
    """
    Create visualizations to validate the demographic distribution of drawn personas.
    
    Args:
        personas (pd.DataFrame): Drawn personas data
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a figure with subplots for each demographic variable
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot distributions
    variables = ['AGE', 'H1', 'STATE', 'J6']
    titles = ['Age Distribution', 'Gender Distribution', 'State Distribution', 'Income Distribution']
    
    for i, (var, title) in enumerate(zip(variables, titles)):
        sns.countplot(data=personas, x=var, ax=axes[i])
        axes[i].set_title(title)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'demographic_validation_{timestamp}.png'))
    plt.close()
    
    # Print summary statistics
    print("\nDemographic Distribution Summary:")
    for var in variables:
        print(f"\n{var} Distribution:")
        print(personas[var].value_counts(normalize=True).round(3))

def generate_aes_persona_response(persona: pd.Series, client: OpenAI) -> PersonaResponse:
    """
    Generate a persona's response using AES/ABS data format.
    
    Args:
        persona (pd.Series): Persona data from AES/ABS format
        client (OpenAI): OpenAI client instance
        
    Returns:
        PersonaResponse: Complete response including both narrative and survey data
    """
    # Extract all possible fields, defaulting to 'Unknown' or '' if missing
    persona_details = {
        "name": f"Persona_{persona.name}",
        "age": str(persona.get('AGE', 'Unknown')),
        "gender": str(persona.get('H1', 'Unknown')),
        "location": str(persona.get('STATE', 'Unknown')),
        "income": str(persona.get('J6', 'Unknown')),
        "tenure": str(persona.get('J1', 'Unknown')),
        "job_tenure": str(persona.get('G5_E', 'Unknown')),
        "occupation": str(persona.get('G5_D', 'Unknown')),
        "education": str(persona.get('G3', 'Unknown')),
        "transport": str(persona.get('transport', 'Unknown')),
        "marital_status": str(persona.get('H8', 'Unknown')),
        "partner_activity": str(persona.get('G7_1', 'Unknown')),
        "household_size": str(persona.get('G7_2', 'Unknown')),
        "family_payments": str(persona.get('G7_3', 'Unknown')),
        "child_care_benefit": str(persona.get('G7_4', 'Unknown')),
        "investment_properties": str(persona.get('G7_5', 'Unknown')),
        "transport_infrastructure": str(persona.get('G7_6', 'Unknown')),
        "political_leaning": str(persona.get('B1', 'Unknown')),
        "trust": str(persona.get('C6', 'Unknown')),
        "issues": str(persona.get('issues', '')),
        "engagement": str(persona.get('engagement', ''))
    }
    prompt = f"""You are simulating the response of a fictional but demographically grounded persona for use in a synthetic civic focus group. This persona is based on Australian Electoral Study (AES) and ABS Census data.\n\nPersona Details:\n- Name: {persona_details['name']}\n- Age: {persona_details['age']}\n- Gender: {persona_details['gender']}\n- Location: {persona_details['location']}\n- Income: {persona_details['income']}\n- Tenure: {persona_details['tenure']}\n- Job Tenure: {persona_details['job_tenure']}\n- Occupation: {persona_details['occupation']}\n- Education: {persona_details['education']}\n- Transport: {persona_details['transport']}\n- Marital Status: {persona_details['marital_status']}\n- Partner Activity: {persona_details['partner_activity']}\n- Household Size: {persona_details['household_size']}\n- Family Payments: {persona_details['family_payments']}\n- Child Care Benefit: {persona_details['child_care_benefit']}\n- Investment Properties: {persona_details['investment_properties']}\n- Transport Infrastructure: {persona_details['transport_infrastructure']}\n- Political Leaning: {persona_details['political_leaning']}\n- Trust in Government: {persona_details['trust']}\n- Key Issues: {persona_details['issues']}\n- Engagement Level: {persona_details['engagement']}\n\nYou have been asked to react to the following **local proposal**:\n\n> \"Waverley Council is considering a policy that would remove minimum parking requirements for new apartment developments in Bondi. This means developers could build fewer or no car spaces if they believe it suits the residents' needs.\"\n\nIMPORTANT: Based on your demographic profile, you should take a strong position on this issue. Consider how your background might lead you to have extreme views:\n\n- If you're a car-dependent commuter, you might strongly oppose this policy\n- If you're a young renter who doesn't own a car, you might strongly support it\n- If you're concerned about housing affordability, you might see this as a crucial step\n- If you're worried about parking in your neighborhood, you might see this as a major threat\n- If you're environmentally conscious, you might view this as essential for sustainability\n- If you're a property owner, you might be concerned about impacts on property values\n- If you have investment properties, you might be concerned about property values\n- If you have children and receive family payments, you might be concerned about housing affordability\n- If you're in a larger household, you might be more concerned about parking availability\n- If you're retired, you might be more concerned about community impact\n\nPlease provide:\n\n1. A short narrative response (2-3 sentences) that reflects:\n   - A clear, strong position on the policy (either strongly support or strongly oppose)\n   - Why — in your own words, as someone with this background\n   - What specific impacts you're most concerned about\n\n2. A structured survey response with the following:\n   - Support Level (1-5, where 1 is strongly oppose and 5 is strongly support)\n   - Impact on Housing Affordability (1-5, where 1 is very negative and 5 is very positive)\n   - Impact on Transport (1-5, where 1 is very negative and 5 is very positive)\n   - Impact on Community (1-5, where 1 is very negative and 5 is very positive)\n   - Key Concerns (comma-separated list)\n   - Suggested Improvements (comma-separated list)\n\nFormat your response as follows:\n\nNARRATIVE RESPONSE:\n[Your narrative response here]\n\nSURVEY RESPONSE:\nSupport Level: [1-5]\nImpact on Housing: [1-5]\nImpact on Transport: [1-5]\nImpact on Community: [1-5]\nKey Concerns: [comma-separated list]\nSuggested Improvements: [comma-separated list]"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that simulates realistic persona responses based on demographic data. You should encourage strong, diverse viewpoints based on demographic factors."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    response_text = response.choices[0].message.content
    # Split the response into narrative and survey parts
    parts = response_text.split('SURVEY RESPONSE:')
    narrative = parts[0].replace('NARRATIVE RESPONSE:', '').strip()
    survey = parts[1].strip() if len(parts) > 1 else ""
    # Parse the survey response
    survey_response = parse_survey_response(survey)
    # Analyze sentiment and extract themes
    sentiment_score = analyze_sentiment(narrative)
    key_themes = extract_key_themes(narrative, client)
    return PersonaResponse(
        persona_details=persona_details,
        narrative_response=narrative,
        survey_response=survey_response,
        timestamp=datetime.now().isoformat(),
        sentiment_score=sentiment_score,
        key_themes=key_themes
    )

def row_to_persona(row, personas_df=None):
    def safe_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    return Persona(
        name=f"Persona_{row.name}",
        age=row.get('ABS_AGE_CATEGORY', 'Unknown'),
        gender=row.get('ABS_SEX', 'Unknown'),
        location=row.get('SA2 (UR)', 'Unknown'),
        income=row.get('ABS_INCOME', 'Unknown'),
        tenure=tenure.get(safe_int(row.get('J1')), 'Unknown'),
        job_tenure=job_tenure.get(safe_int(row.get('G5_E')), 'Unknown'),
        occupation=job_type.get(safe_int(row.get('G5_D')), 'Unknown'),
        education=edu_level.get(safe_int(row.get('G3')), 'Unknown'),
        marital_status=marital_status.get(safe_int(row.get('H8')), 'Unknown'),
        partner_activity=partner_activity.get(safe_int(row.get('I1')), 'Unknown'),
        household_size=household_size.get(safe_int(row.get('W1')), 'Unknown'),
        family_payments=family_payments.get(safe_int(row.get('J8_1')), 'Unknown'),
        child_care_benefit=child_care_benefit.get(safe_int(row.get('J8_2')), 'Unknown'),
        investment_properties=investment_properties.get(safe_int(row.get('J2')), 'Unknown'),
        transport_infrastructure=transport_infrastructure.get(safe_int(row.get('D8_9')), 'Unknown'),
        political_leaning=political_leaning.get(safe_int(row.get('B9_1')), 'Unknown'),
        trust=trust_gov.get(safe_int(row.get('C6')), 'Unknown'),
        issues=construct_issues_list(row, personas_df) if personas_df is not None else [],
        engagement=construct_engagement_string(row, personas_df) if personas_df is not None else "Unknown"
    )

# --- Engagement and Issues String Construction Functions ---
def construct_engagement_string(row, personas_df):
    engagement_str = "I engage in politics in the following way: "
    parts = []
    for key, description in engagement_dict.items():
        answer_value = personas_df.at[row.name, key] if key in personas_df.columns else 999
        answer_text = get_mapping_dict(key).get(answer_value, "Unknown")
        if answer_text not in ["Item skipped", "Unknown"]:
            parts.append(f"{description}, {answer_text}")
    if parts:
        engagement_str += "; ".join(parts)
    else:
        engagement_str = "No significant engagement reported."
    return engagement_str

def construct_issues_list(row, personas_df):
    parts = []
    for key, description in issues_dict.items():
        answer_value = personas_df.at[row.name, key] if key in personas_df.columns else 999
        answer_text = get_mapping_dict(key).get(answer_value, "Unknown")
        if answer_text not in ["Item skipped", "Unknown"]:
            parts.append(f"{description}, {answer_text}")
    return parts


def construct_issues_string(row, personas_df):
    issues_str = "How I feel about the following Issues include: "
    parts = []
    for key, description in issues_dict.items():
        answer_value = personas_df.at[row.name, key] if key in personas_df.columns else 999
        answer_text = get_mapping_dict(key).get(answer_value, "Unknown")
        if answer_text not in ["Item skipped", "Unknown"]:
            parts.append(f"{description}, {answer_text}")
    if parts:
        issues_str += "; ".join(parts)
    else:
        issues_str = "No significant issues reported."
    return issues_str
# --- End Engagement/Issues Functions ---

def persona_to_dict(persona):
    # Helper to convert Persona object to dict for CSV
    return {
        'name': persona.name,
        'age': persona.age,
        'gender': getattr(persona, 'gender', ''),
        'location': persona.location,
        'income': getattr(persona, 'income', ''),
        'tenure': getattr(persona, 'tenure', ''),
        'job_tenure': getattr(persona, 'job_tenure', ''),
        'occupation': getattr(persona, 'occupation', ''),
        'education': getattr(persona, 'education', ''),
        'transport': getattr(persona, 'transport', ''),
        'marital_status': getattr(persona, 'marital_status', ''),
        'partner_activity': getattr(persona, 'partner_activity', ''),
        'household_size': getattr(persona, 'household_size', ''),
        'family_payments': getattr(persona, 'family_payments', ''),
        'child_care_benefit': getattr(persona, 'child_care_benefit', ''),
        'investment_properties': getattr(persona, 'investment_properties', ''),
        'transport_infrastructure': getattr(persona, 'transport_infrastructure', ''),
        'political_leaning': getattr(persona, 'political_leaning', ''),
        'trust': getattr(persona, 'trust', ''),
        'issues': ', '.join(getattr(persona, 'issues', [])),
        'engagement': getattr(persona, 'engagement', '')
    }

# --- Update persona construction with missing field check ---
def construct_persona_with_check(row, personas_df=None):
    def safe_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def get_value(mapping_dict, key, default='Unknown'):
        value = mapping_dict.get(safe_int(row.get(key)), default)
        return value if value != 'Item skipped' else default

    persona = Persona(
        name=f"Persona_{row.name}",
        age=row.get('ABS_AGE_CATEGORY', 'Unknown'),
        gender=row.get('ABS_SEX', 'Unknown'),
        location=row.get('SA2 (UR)', 'Unknown'),
        income=row.get('ABS_INCOME', 'Unknown'),
        tenure=get_value(tenure, 'J1'),
        job_tenure=get_value(job_tenure, 'G5_E'),
        occupation=get_value(job_type, 'G5_D'),
        education=get_value(edu_level, 'G3'),
        transport=random_car_ownership(),  # Use random_car_ownership for transport
        marital_status=get_value(marital_status, 'H8'),
        partner_activity=get_value(partner_activity, 'I1'),
        household_size=get_value(household_size, 'W1'),
        family_payments=get_value(family_payments, 'J8_1'),
        child_care_benefit=get_value(child_care_benefit, 'J8_2'),
        investment_properties=get_value(investment_properties, 'J2'),
        transport_infrastructure=get_value(transport_infrastructure, 'D8_9'),
        political_leaning=get_value(political_leaning, 'B9_1'),
        trust=get_value(trust_gov, 'C6'),
        issues=construct_issues_list(row, personas_df) if personas_df is not None else [],
        engagement=construct_engagement_string(row, personas_df) if personas_df is not None else "Unknown"
    )
    # Check for missing fields
    missing = []
    for field in ['name','age','gender','location','income','tenure','job_tenure','occupation','education','transport','marital_status','partner_activity','household_size','family_payments','child_care_benefit','investment_properties','transport_infrastructure','political_leaning','trust','issues','engagement']:
        val = getattr(persona, field, None)
        if val in [None, '', 'Unknown', [], ['Unknown']]:
            missing.append(field)
    if missing:
        print(f"[WARNING] Persona {persona.name} has missing fields: {', '.join(missing)}")
    return persona

def random_car_ownership():
    # Updated probabilities based on the table
    probabilities = [0.179, 0.456, 0.236, 0.068, 0.061]  # Waverley percentages
    
    # Car ownership categories
    car_ownership = ["No motor vehicles", "1 motor vehicle", "2 motor vehicles", 
                     "3 or more motor vehicles", "Not stated"]
    
    # Randomly choose based on probabilities
    return np.random.choice(car_ownership, p=probabilities)

def save_personas_csv(personas: List[Persona], output_dir: str, timestamp: str) -> None:
    """Save personas to a CSV file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert personas to list of dictionaries
    persona_dicts = [persona_to_dict(persona) for persona in personas]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(persona_dicts)
    output_file = os.path.join(output_dir, f'personas_{timestamp}.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved personas to {output_file}")

def main(num_personas):
    """Main function to process personas and generate responses."""
    print("\nVoxPop Personas Analysis Menu:")
    print("1. Run full ABS/AES harmonization and persona generation pipeline")
    print("2. Run full analysis with new responses")
    print("3. Regenerate plots from existing data")
    print("4. Generate synthetic personas from AES/ABS data")
    print("5. Create toy datasets from large AES/ABS data")
    print("6. Test AES/ABS persona generation")
    print("7. Exit")
    
    choice = input("\nEnter your choice (1-7): ")
    
    if choice == "1":
        print("\nRunning full ABS/AES harmonization and persona generation pipeline...\n")
        # 1. Read and harmonize data
        abs_data = read_abs_data("data/Personas_wide.csv")
        aes_data = pd.read_csv("data/aes22_unrestricted_v3.csv", low_memory=False)
        
        # 2. Harmonize AES data
        print("Harmonizing AES data...")
        aes_data['ABS_AGE_CATEGORY'] = aes_data['AGE'].apply(lambda x: map_age_to_abs_age(x, abs_to_aes_age))
        aes_code_to_abs_sex = {v: k for k, v in abs_to_aes_sex.items()}
        aes_data['ABS_SEX'] = aes_data['H1'].map(aes_code_to_abs_sex)
        aes_data['ABS_MARITAL'] = aes_data['H8'].map({v: k for k, v in abs_to_aes_marital.items()})
        aes_data['ABS_INCOME'] = aes_data['J6'].apply(lambda x: map_income_to_abs_income(x, abs_to_aes_income))
        
        # Check unique values after mapping
        print("Unique AGE values in AES:", aes_data['AGE'].unique())
        print("Unique H1 (sex) values in AES:", aes_data['H1'].unique())
        print("Unique H8 (marital) values in AES:", aes_data['H8'].unique())
        print("Unique J6 (income) values in AES:", aes_data['J6'].unique())

        print("Unique ABS_AGE_CATEGORY after mapping:", aes_data['ABS_AGE_CATEGORY'].unique())
        print("Unique ABS_SEX after mapping:", aes_data['ABS_SEX'].unique())
        print("Unique ABS_MARITAL after mapping:", aes_data['ABS_MARITAL'].unique())
        print("Unique ABS_INCOME after mapping:", aes_data['ABS_INCOME'].unique())
        
        # Drop rows with missing values in key columns
        aes_data_harmonized = aes_data.dropna(subset=['ABS_AGE_CATEGORY', 'ABS_SEX', 'ABS_MARITAL', 'ABS_INCOME'])
        print(f"After harmonization: {len(aes_data_harmonized)} rows in AES data")
        print("Columns after harmonization:", aes_data_harmonized.columns.tolist())
        
        # After harmonization
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        aes_harmonized_path = os.path.join(output_dir, f"aes_data_harmonized_{timestamp}.csv")
        aes_data_harmonized.to_csv(aes_harmonized_path, index=False)
        print(f"Harmonized AES data saved to: {aes_harmonized_path}")
        
        # 3. Prepare ABS data for merge
        print("Preparing ABS data for merge...")
        # Define the income columns as in ABS
        income_cols = [
            "Negative income",
            "Nil income",
            "$1-$149 ($1-$7,799)",
            "$150-$299 ($7,800-$15,599)",
            "$300-$399 ($15,600-$20,799)",
            "$400-$499 ($20,800-$25,999)",
            "$500-$649 ($26,000-$33,799)",
            "$650-$799 ($33,800-$41,599)",
            "$800-$999 ($41,600-$51,999)",
            "$1,000-$1,249 ($52,000-$64,999)",
            "$1,250-$1,499 ($65,000-$77,999)",
            "$1,500-$1,749 ($78,000-$90,999)",
            "$1,750-$1,999 ($91,000-$103,999)",
            "$2,000-$2,999 ($104,000-$155,999)",
            "$3,000-$3,499 ($156,000-$181,999)",
            "$3,500 or more ($182,000 or more)",
            "Not stated",
            "Not applicable"
        ]
        # Filter out rows where all income columns sum to zero or NaN
        income_sums = abs_data[income_cols].sum(axis=1)
        filtered_abs_data = abs_data[income_sums > 0.0].copy()
        # Melt the ABS data to long format
        melted = filtered_abs_data.melt(
            id_vars=[
                "SA2 (UR)",
                "AGE5P Age in Five Year Groups",
                "SEXP Sex",
                "FPIP Parent Indicator",
                "MSTP Registered Marital Status",
                "1-digit level ANCP Ancestry Multi Response"
            ],
            value_vars=income_cols,
            var_name="Income Level",
            value_name="weight"
        )
        # Keep only rows with positive weights
        poststrat_frame = melted[melted['weight'] > 0].copy()
        print(poststrat_frame.head())
        print(f"Number of strata (demographics x income): {len(poststrat_frame)}")
        
        # 4. Merge datasets
        print("Merging datasets...")
        poststrat_for_merge = poststrat_frame.rename(columns={
            "AGE5P Age in Five Year Groups": "ABS_AGE_CATEGORY",
            "SEXP Sex": "ABS_SEX",
            "MSTP Registered Marital Status": "ABS_MARITAL",
            "Income Level": "ABS_INCOME"
        })

        merged = pd.merge(
            poststrat_for_merge,
            aes_data_harmonized,
            on=["ABS_AGE_CATEGORY", "ABS_SEX", "ABS_MARITAL", "ABS_INCOME"],
            how="inner"
        )

        # Fix weight columns
        if 'weight_x' in merged.columns:
            merged = merged.rename(columns={'weight_x': 'weight'})
        if 'weight_y' in merged.columns:
            merged = merged.drop(columns=['weight_y'])

        print(f"Merge successful. Result has {len(merged)} rows")
        print("Columns after merge:", merged.columns.tolist())
        
        # After merging
        merged_path = os.path.join(output_dir, f"merged_data_{timestamp}.csv")
        merged.to_csv(merged_path, index=False)
        print(f"Merged data saved to: {merged_path}")
        
        # 5. Sample personas (major city filter ON by default)
        N = num_personas  # For quick test
        print(f"\nSampling {N} personas...")
        personas_df = sample_personas_major_city(merged, N)
        print(f"Successfully sampled {len(personas_df)} personas")
        print("Columns in personas_df:", personas_df.columns.tolist())
        
        # 6. Construct Persona objects
        print("\nConstructing Persona objects...")
        persona_objects = [construct_persona_with_check(row, personas_df) for _, row in personas_df.iterrows()]
        
        # 7. Print results
        print("\nGenerated Personas:")
        for persona in persona_objects:
            print(f"\n{persona}")
        print("\nPipeline complete.\n")

        # --- NEW: Generate LLM responses, print, and save analysis (was Step 2) ---
        proceed = input("\nGenerate LLM responses and analysis for these personas? (y/n): ")
        if proceed.lower() == 'y':
            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            responses = []
            for persona in persona_objects:
                response = generate_persona_response(persona, client)
                responses.append(response)
                print(f"\nPersona Profile:")
                print(f"Name: {response.persona_details['name']}")
                print(f"Age: {response.persona_details['age']}")
                print(f"Location: {response.persona_details['location']}")
                print(f"Occupation: {response.persona_details['occupation']}")
                print(f"Political Leaning: {response.persona_details['political_leaning']}")
                print("\nNarrative Response:")
                print(response.narrative_response)
                print("\nSurvey Response:")
                print(f"Support Level: {response.survey_response.support_level}")
                print(f"Impact on Housing: {response.survey_response.impact_on_housing}")
                print(f"Impact on Transport: {response.survey_response.impact_on_transport}")
                print(f"Impact on Community: {response.survey_response.impact_on_community}")
                print(f"Key Concerns: {', '.join(response.survey_response.key_concerns)}")
                print(f"Suggested Improvements: {', '.join(response.survey_response.suggested_improvements)}")
                print("\nAnalysis:")
                print(f"Sentiment Score: {response.sentiment_score:.2f}")
                print(f"Key Themes: {', '.join(response.key_themes)}")
                print("=" * 80)
            # Save responses and generate analysis/plots
            timestamp = save_responses(responses)
            print(f"\nAnalysis complete. Files saved with timestamp: {timestamp}")
    elif choice == "2":
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load personas
        data_path = Path("data/voxpopai_200_personas_skewed.csv")
        personas = load_personas(str(data_path))
        
        # Generate responses for each persona
        print("\nGenerating persona responses to the Bondi parking policy proposal...\n")
        print("=" * 80)
        
        responses = []
        for persona in personas:
            response = generate_persona_response(persona, client)
            responses.append(response)
            
            print(f"\nPersona Profile:")
            print(f"Name: {response.persona_details['name']}")
            print(f"Age: {response.persona_details['age']}")
            print(f"Location: {response.persona_details['location']}")
            print(f"Occupation: {response.persona_details['occupation']}")
            print(f"Political Leaning: {response.persona_details['political_leaning']}")
            
            print("\nNarrative Response:")
            print(response.narrative_response)
            
            print("\nSurvey Response:")
            print(f"Support Level: {response.survey_response.support_level}")
            print(f"Impact on Housing: {response.survey_response.impact_on_housing}")
            print(f"Impact on Transport: {response.survey_response.impact_on_transport}")
            print(f"Impact on Community: {response.survey_response.impact_on_community}")
            print(f"Key Concerns: {', '.join(response.survey_response.key_concerns)}")
            print(f"Suggested Improvements: {', '.join(response.survey_response.suggested_improvements)}")
            
            print("\nAnalysis:")
            print(f"Sentiment Score: {response.sentiment_score:.2f}")
            print(f"Key Themes: {', '.join(response.key_themes)}")
            
            print("=" * 80)
        
        # Save responses and generate analysis
        timestamp = save_responses(responses)
        print(f"\nAnalysis complete. Files saved with timestamp: {timestamp}")
        
    elif choice == "3":
        # List available data files
        data_files = [f for f in os.listdir("output") if f.startswith("responses_summary_") and f.endswith(".csv")]
        if not data_files:
            print("\nNo data files found in output directory.")
            return
            
        print("\nAvailable data files:")
        for i, file in enumerate(data_files, 1):
            print(f"{i}. {file}")
        
        file_choice = input("\nEnter the number of the file to use (or 'q' to quit): ")
        if file_choice.lower() == 'q':
            return
            
        try:
            file_index = int(file_choice) - 1
            selected_file = os.path.join("output", data_files[file_index])
            print('Selected file: ', file_index, selected_file)
            regenerate_plots(selected_file)
        except (ValueError, IndexError):
            print("\nInvalid selection. Please try again.")
            
    elif choice == "4":
        # Get input file paths
        aes_file = input("\nEnter path to AES data file: ")
        abs_file = input("Enter path to ABS data file: ")
        
        # Get number of personas to generate
        try:
            num_personas = int(input("Enter number of personas to generate (default 10): ") or "10")
            sample_size = int(input("Enter sample size for AES data (default 100): ") or "100")
        except ValueError:
            print("\nInvalid input. Using default values.")
            num_personas = 10
            sample_size = 100
        
        # Generate synthetic personas
        synthetic_personas, poststrat_frame = generate_synthetic_personas(aes_file, abs_file, num_personas, sample_size)
            
    elif choice == "5":
        # Create toy datasets with default paths
        aes_file, abs_file = create_toy_datasets()
        if aes_file and abs_file:
            print("\nToy datasets created successfully. You can now use these files with option 4.")
            
    elif choice == "6":
        # Get input file paths
        poststrat_file = input("\nEnter path to post-stratification frame: ")
        
        # Get number of personas to generate
        try:
            num_personas = int(input("Enter number of personas to generate (default 10): ") or "10")
        except ValueError:
            print("\nInvalid input. Using default value of 10.")
            num_personas = 10
        
        # Load post-stratification frame
        poststrat_frame = pd.read_csv(poststrat_file)
        
        # Draw weighted personas
        personas = draw_weighted_personas(poststrat_frame, num_personas)
        
        # Validate demographic distribution
        validate_demographic_distribution(personas)
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Generate responses for each persona
        print("\nGenerating persona responses to the Bondi parking policy proposal...\n")
        print("=" * 80)
        
        responses = []
        for _, persona in personas.iterrows():
            response = generate_aes_persona_response(persona, client)
            responses.append(response)
            
            print(f"\nPersona Profile:")
            print(f"Name: {response.persona_details['name']}")
            print(f"Age: {response.persona_details['age']}")
            print(f"Location: {response.persona_details['location']}")
            print(f"Political Leaning: {response.persona_details['political_leaning']}")
            
            print("\nNarrative Response:")
            print(response.narrative_response)
            
            print("\nSurvey Response:")
            print(f"Support Level: {response.survey_response.support_level}")
            print(f"Impact on Housing: {response.survey_response.impact_on_housing}")
            print(f"Impact on Transport: {response.survey_response.impact_on_transport}")
            print(f"Impact on Community: {response.survey_response.impact_on_community}")
            print(f"Key Concerns: {', '.join(response.survey_response.key_concerns)}")
            print(f"Suggested Improvements: {', '.join(response.survey_response.suggested_improvements)}")
            
            print("\nAnalysis:")
            print(f"Sentiment Score: {response.sentiment_score:.2f}")
            print(f"Key Themes: {', '.join(response.key_themes)}")
            
            print("=" * 80)
        
        # Save responses and generate analysis
        timestamp = save_responses(responses)
        print(f"\nAnalysis complete. Files saved with timestamp: {timestamp}")
            
    elif choice == "7":
        print("\nExiting program.")
        return
        
    else:
        print("\nInvalid choice. Please try again.")
        return

if __name__ == "__main__":
    main() 