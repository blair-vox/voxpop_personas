"""
Controller for handling persona generation and analysis in the GUI.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from openai import OpenAI
import ast
import streamlit as st

from models.persona import Persona, PersonaResponse
from generation.persona_gen import (
    generate_persona_response,
    save_responses_with_canonical_themes
)
from analysis.visualization import create_theme_analysis
from pipeline.persona_pipeline import run_new_pipeline

class PersonaController:
    def __init__(self):
        self.current_responses: List[PersonaResponse] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.client = OpenAI()
    
    def generate_personas(self, num_personas: int, random_state: Optional[int] = None, custom_prompt: Optional[str] = None, question: Optional[str] = None) -> List[PersonaResponse]:
        """
        Generate a specified number of personas using the main pipeline.
        
        Args:
            num_personas (int): Number of personas to generate
            random_state (Optional[int]): Random state for reproducibility. If None, uses current time.
            custom_prompt (Optional[str]): Custom prompt to use for persona generation
            question (Optional[str]): The local proposal/question to include in the prompt
        
        Returns:
            List[PersonaResponse]: List of generated persona responses
        """
        responses = run_new_pipeline(num_personas, random_state=random_state, custom_prompt=custom_prompt, question=question)
        self.current_responses = responses
        return responses
    
    def analyze_themes(self) -> Dict[str, Any]:
        """Analyze themes from current responses (in-memory only, no saving)."""
        if not self.current_responses:
            print("[DEBUG] No current_responses in analyze_themes.")
            return {}
        
        # Convert responses to DataFrame format
        data = []
        for i, response in enumerate(self.current_responses):
            print(f"[DEBUG] Response {i} canonical_themes type:", type(response.canonical_themes))
            print(f"[DEBUG] Response {i} canonical_themes value:", response.canonical_themes)
            print(f"[DEBUG] Response {i} sentiment_score:", response.sentiment_score)
            data.append({
                'persona_details': response.persona_details,
                'narrative_response': response.narrative_response,
                'timestamp': response.timestamp,
                'sentiment_score': response.sentiment_score,
                'key_themes': response.key_themes,
                'canonical_themes': response.canonical_themes,
                'support_level': response.survey_response.support_level,
                'impact_on_housing': response.survey_response.impact_on_housing,
                'impact_on_transport': response.survey_response.impact_on_transport,
                'impact_on_community': response.survey_response.impact_on_community,
                'key_concerns': response.survey_response.key_concerns,
                'suggested_improvements': response.survey_response.suggested_improvements
            })
        print("[DEBUG] Number of responses:", len(self.current_responses))
        df = pd.DataFrame(data)

        # Theme counts (flatten canonical themes)
        canonical_theme_list = []
        for ct in df['canonical_themes']:
            if isinstance(ct, dict):
                canonical_theme_list.extend(list(ct.keys()))
            elif isinstance(ct, list):
                canonical_theme_list.extend([str(x) for x in ct])
        print("[DEBUG] canonical_theme_list:", canonical_theme_list)
        theme_counts = pd.Series(canonical_theme_list).value_counts().reset_index()
        theme_counts.columns = ['theme', 'count']

        # Theme mapping (original to canonical)
        theme_mapping = {}
        for idx, row in df.iterrows():
            if isinstance(row['key_themes'], list) and isinstance(row['canonical_themes'], dict):
                for orig, canon in zip(row['key_themes'], row['canonical_themes'].keys()):
                    theme_mapping[orig] = canon

        # Aggregate sentiment and impact by theme
        sentiment_by_theme = []
        impact_by_theme = []
        for idx, row in df.iterrows():
            canonical_theme_list = row['canonical_themes']
            # Parse stringified dicts if needed
            if canonical_theme_list and isinstance(canonical_theme_list[0], str):
                try:
                    canonical_theme_list = [ast.literal_eval(x) for x in canonical_theme_list]
                except Exception as e:
                    print(f"[DEBUG] Error parsing canonical_theme_list at row {idx}: {e}")
                    continue
            for theme_dict in canonical_theme_list:
                if isinstance(theme_dict, dict):
                    sentiment_by_theme.append({'theme': theme_dict.get('canonical', 'Unknown'), 'sentiment': row['sentiment_score']})
                    impact_by_theme.append({'theme': theme_dict.get('canonical', 'Unknown'), 'impact': row['impact_on_housing']})
        print("[DEBUG] sentiment_by_theme:", sentiment_by_theme)
        print("[DEBUG] impact_by_theme:", impact_by_theme)

        return {
            'theme_counts': theme_counts,
            'theme_mapping': theme_mapping,
            'sentiment_by_theme': sentiment_by_theme,
            'impact_by_theme': impact_by_theme
        }
    
    def save_responses(self, output_dir: str = "output") -> str:
        """Save current responses to file."""
        if not self.current_responses:
            return ""
        
        return save_responses_with_canonical_themes(
            self.current_responses,
            self.client,
            output_dir=output_dir
        ) 