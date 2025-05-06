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

def create_heatmap(data: pd.DataFrame, output_dir: str):
    """
    Create heatmap of support levels by demographic groups.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
    """
    # Create pivot table for support levels
    pivot_data = data.pivot_table(
        values='support_level',
        index='political_leaning',
        columns='age',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=3)
    plt.title('Support Level Heatmap by Age and Political Leaning')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'support_heatmap.png'))
    plt.close()

def create_sentiment_analysis(data: pd.DataFrame, output_dir: str):
    """
    Create sentiment analysis visualizations.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
    """
    # Convert sentiment scores to a more intuitive scale (0-100)
    data['sentiment_percentage'] = ((data['sentiment_score'] + 1) / 2 * 100).round(0)
    
    # Sentiment by political leaning
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='political_leaning', y='sentiment_percentage', data=data)
    plt.title('Sentiment Distribution by Political Leaning')
    plt.ylabel('Sentiment (% Positive)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_politics.png'))
    plt.close()
    
    # Sentiment by age group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='age', y='sentiment_percentage', data=data)
    plt.title('Sentiment Distribution by Age')
    plt.ylabel('Sentiment (% Positive)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_age.png'))
    plt.close()

def create_impact_analysis(data: pd.DataFrame, output_dir: str):
    """
    Create impact analysis visualizations.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
    """
    # Impact scores by demographic
    impact_cols = ['impact_on_housing', 'impact_on_transport', 'impact_on_community']
    
    plt.figure(figsize=(12, 6))
    data[impact_cols].mean().plot(kind='bar')
    plt.title('Average Impact Scores by Category')
    plt.ylabel('Average Score (1-5)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'impact_scores.png'))
    plt.close()

def create_theme_analysis(data: pd.DataFrame, output_dir: str):
    """
    Create theme analysis visualizations.
    
    Args:
        data (pd.DataFrame): Response data
        output_dir (str): Output directory
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
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_themes.png'))
    plt.close()

def generate_analysis_report(data: pd.DataFrame, output_dir: str) -> None:
    """
    Generate a comprehensive analysis report of the responses.
    
    Args:
        data (pd.DataFrame): DataFrame containing all responses
        output_dir (str): Directory to save the report
    """
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
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
        group_stats = data.groupby('political_leaning')['support_level'].mean()
        for group, mean_support in group_stats.items():
            f.write(f"{group}: {mean_support:.2f}\n")
        f.write("\n")
        
        # Analysis by age group
        f.write("Analysis by Age Group:\n")
        f.write("-" * 30 + "\n")
        data['age_group'] = pd.cut(data['age'].astype(int), 
                                  bins=[0, 30, 50, 70, 100],
                                  labels=['18-30', '31-50', '51-70', '70+'])
        age_stats = data.groupby('age_group')['support_level'].mean()
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
        # Convert sentiment scores to percentage
        sentiment_percentage = ((data['sentiment_score'] + 1) / 2 * 100).round(0)
        f.write(f"Average Sentiment: {sentiment_percentage.mean():.1f}% Positive\n")
        f.write(f"Sentiment Distribution:\n")
        sentiment_stats = pd.cut(sentiment_percentage, 
                               bins=[0, 20, 40, 60, 80, 100],
                               labels=['Very Negative (0-20%)', 'Negative (21-40%)', 'Neutral (41-60%)', 'Positive (61-80%)', 'Very Positive (81-100%)'])
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

Please provide:

1. A short narrative response (2-3 sentences) that reflects:
   - Whether you support or oppose the policy
   - Why — in your own words, as someone with this background
   - (Optional) What would make you change your mind

2. A structured survey response with the following:
   - Support Level (1-5, where 1 is strongly oppose and 5 is strongly support)
   - Impact on Housing Affordability (1-5, where 1 is very negative and 5 is very positive)
   - Impact on Transport (1-5, where 1 is very negative and 5 is very positive)
   - Impact on Community (1-5, where 1 is very negative and 5 is very positive)
   - Key Concerns (comma-separated list)
   - Suggested Improvements (comma-separated list)

Where possible, reflect your opinion using tone and language that matches your demographic and issue profile.

**Relevant local context for grounding:**
- Reddit post (r/sydney, 2023): "There are too many ghost garages in Bondi. We need more housing, not car spots."
- Grattan Institute: 68% of renters under 35 support relaxed parking minimums in high-transit areas.
- AEC data: Bondi Junction booths saw 30%+ Green vote among young renters in 2022.

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
            {"role": "system", "content": "You are a helpful assistant that simulates realistic persona responses based on demographic data."},
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
    
    # Save all responses in a single JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
    create_heatmap(summary_df, output_dir)
    create_sentiment_analysis(summary_df, output_dir)
    create_impact_analysis(summary_df, output_dir)
    create_theme_analysis(summary_df, output_dir)
    generate_analysis_report(summary_df, output_dir)

def main():
    """Main function to process personas and generate responses."""
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load personas
    data_path = Path("data/voxpopai_clean_personas.csv")
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
    save_responses(responses)

if __name__ == "__main__":
    main() 