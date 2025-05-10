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
    persona_details: Dict[str, str]
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
    occupation: str
    education: str
    transport: str
    political_leaning: str
    trust: str
    issues: List[str]
    engagement: str

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
    
    # Convert age to numeric for proper sorting
    data['age'] = pd.to_numeric(data['age'])
    
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
    sns.boxplot(x='age', y='sentiment_percentage', data=data.sort_values('age'))
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
        all_themes.extend(themes)
    
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
        data['age_group'] = pd.cut(data['age'].astype(int), 
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
            tenure=row['tenure'],
            occupation=row['occupation'],
            education=row['education'],
            transport=row['transport'],
            political_leaning=row['political_leaning'],
            trust=row['trust'],
            issues=row['issues'].split(', ') if isinstance(row['issues'], str) else [],
            engagement=row['engagement']
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
    prompt = f"""You are simulating the response of a fictional but demographically grounded persona for use in a synthetic civic focus group. This persona is based on Australian Census data and local electoral trends.

Persona Details:
- Name: {persona.name}
- Age: {persona.age}  
- Gender: {persona.gender}  
- Location: {persona.location}, NSW  
- Income: {persona.income}  
- Tenure: {persona.tenure}  
- Occupation: {persona.occupation}  
- Education: {persona.education}  
- Transport Mode: {persona.transport}  
- Political Leaning: {persona.political_leaning}  
- Trust in Government: {persona.trust}  
- Key Issues: {', '.join(persona.issues)}  
- Engagement Level: {persona.engagement}

You have been asked to react to the following **local proposal**:

> "Waverley Council is considering a policy that would remove minimum parking requirements for new apartment developments in Bondi. This means developers could build fewer or no car spaces if they believe it suits the residents' needs."

IMPORTANT: Based on your demographic profile, you should take a strong position on this issue. Consider how your background might lead you to have extreme views:

- If you're a car-dependent commuter, you might strongly oppose this policy
- If you're a young renter who doesn't own a car, you might strongly support it
- If you're concerned about housing affordability, you might see this as a crucial step
- If you're worried about parking in your neighborhood, you might see this as a major threat
- If you're environmentally conscious, you might view this as essential for sustainability
- If you're a property owner, you might be concerned about impacts on property values

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
    
    return PersonaResponse(
        persona_details={
            "name": persona.name,
            "age": persona.age,
            "location": persona.location,
            "occupation": persona.occupation,
            "political_leaning": persona.political_leaning
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
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for all files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save all responses in a single JSON file
    json_filepath = os.path.join(output_dir, f"responses_{timestamp}.json")
    
    with open(json_filepath, 'w') as f:
        json.dump([response.model_dump() for response in responses], f, indent=2)
    
    # Create DataFrame for analysis
    summary_data = []
    for response in responses:
        summary_data.append({
            "name": response.persona_details["name"],
            "age": response.persona_details["age"],
            "location": response.persona_details["location"],
            "occupation": response.persona_details["occupation"],
            "political_leaning": response.persona_details["political_leaning"],
            "support_level": response.survey_response.support_level,
            "impact_on_housing": response.survey_response.impact_on_housing,
            "impact_on_transport": response.survey_response.impact_on_transport,
            "impact_on_community": response.survey_response.impact_on_community,
            "key_concerns": ", ".join(response.survey_response.key_concerns),
            "suggested_improvements": ", ".join(response.survey_response.suggested_improvements),
            "sentiment_score": response.sentiment_score,
            "key_themes": response.key_themes
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, f"responses_summary_{timestamp}.csv"), index=False)
    
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
    selected_cols = ["AGE", "H1", "STATE", "J6", "A1", "B9_1", "B1", "weight_final"]
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
    # Map AES/ABS variables to our persona format
    persona_details = {
        "name": f"Persona_{persona.name}",  # Using index as identifier
        "age": str(persona['AGE']),
        "location": persona['STATE'],
        "occupation": "Not specified",  # We might need to map this from other variables
        "political_leaning": persona['B1']  # Political party preference
    }
    
    prompt = f"""You are simulating the response of a fictional but demographically grounded persona for use in a synthetic civic focus group. This persona is based on Australian Electoral Study (AES) and ABS Census data.

Persona Details:
- Age: {persona['AGE']}
- Gender: {persona['H1']}
- Location: {persona['STATE']}
- Income Level: {persona['J6']}
- Political Preference: {persona['B1']}

You have been asked to react to the following **local proposal**:

> "Waverley Council is considering a policy that would remove minimum parking requirements for new apartment developments in Bondi. This means developers could build fewer or no car spaces if they believe it suits the residents' needs."

IMPORTANT: Based on your demographic profile, you should take a strong position on this issue. Consider how your background might lead you to have extreme views:

- If you're a car-dependent commuter, you might strongly oppose this policy
- If you're a young renter who doesn't own a car, you might strongly support it
- If you're concerned about housing affordability, you might see this as a crucial step
- If you're worried about parking in your neighborhood, you might see this as a major threat
- If you're environmentally conscious, you might view this as essential for sustainability
- If you're a property owner, you might be concerned about impacts on property values

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

def main():
    """Main function to process personas and generate responses."""
    print("\nVoxPop Personas Analysis Menu:")
    print("1. Run full analysis with new responses")
    print("2. Regenerate plots from existing data")
    print("3. Generate synthetic personas from AES/ABS data")
    print("4. Create toy datasets from large AES/ABS data")
    print("5. Test AES/ABS persona generation")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == "1":
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
        
    elif choice == "2":
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
            regenerate_plots(selected_file)
        except (ValueError, IndexError):
            print("\nInvalid selection. Please try again.")
            
    elif choice == "3":
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
            
    elif choice == "4":
        # Create toy datasets with default paths
        aes_file, abs_file = create_toy_datasets()
        if aes_file and abs_file:
            print("\nToy datasets created successfully. You can now use these files with option 3.")
            
    elif choice == "5":
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
            
    elif choice == "6":
        print("\nExiting program.")
        return
        
    else:
        print("\nInvalid choice. Please try again.")
        return

if __name__ == "__main__":
    main() 