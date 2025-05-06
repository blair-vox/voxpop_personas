"""
Main script for processing persona data and interacting with OpenAI API.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class Persona(BaseModel):
    """Pydantic model for persona data."""
    name: str
    description: str
    characteristics: List[str]
    goals: List[str]
    pain_points: List[str]

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
            description=row['description'],
            characteristics=row['characteristics'].split(',') if isinstance(row['characteristics'], str) else [],
            goals=row['goals'].split(',') if isinstance(row['goals'], str) else [],
            pain_points=row['pain_points'].split(',') if isinstance(row['pain_points'], str) else []
        )
        personas.append(persona)
    
    return personas

def generate_insights(persona: Persona, client: OpenAI) -> Dict[str, Any]:
    """
    Generate insights for a persona using OpenAI API.
    
    Args:
        persona (Persona): Persona object containing user information
        client (OpenAI): OpenAI client instance
        
    Returns:
        Dict[str, Any]: Generated insights
    """
    prompt = f"""
    Analyze the following persona and provide insights:
    
    Name: {persona.name}
    Description: {persona.description}
    Characteristics: {', '.join(persona.characteristics)}
    Goals: {', '.join(persona.goals)}
    Pain Points: {', '.join(persona.pain_points)}
    
    Please provide:
    1. Key insights about this persona
    2. Potential solutions to their pain points
    3. Recommendations for engaging with this persona
    """
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes user personas and provides actionable insights."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "persona_name": persona.name,
        "insights": response.choices[0].message.content
    }

def main():
    """Main function to process personas and generate insights."""
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load personas
    data_path = Path("data/voxpopai_clean_personas.csv")
    personas = load_personas(str(data_path))
    
    # Generate insights for each persona
    for persona in personas:
        insights = generate_insights(persona, client)
        print(f"\nInsights for {insights['persona_name']}:")
        print(insights['insights'])
        print("-" * 80)

if __name__ == "__main__":
    main() 