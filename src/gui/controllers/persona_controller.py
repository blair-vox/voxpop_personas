"""
Controller for handling persona generation and analysis in the GUI.
"""
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from models.persona import Persona, PersonaResponse
from generation.persona_gen import (
    generate_persona_response,
    save_responses_with_canonical_themes
)
from analysis.visualization import create_theme_analysis

class PersonaController:
    def __init__(self):
        self.current_responses: List[PersonaResponse] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_personas(self, num_personas: int) -> List[PersonaResponse]:
        """Generate a specified number of personas."""
        # TODO: Implement persona generation
        pass
    
    def analyze_themes(self) -> Dict[str, Any]:
        """Analyze themes from current responses."""
        if not self.current_responses:
            return {}
        
        # Convert responses to DataFrame format
        data = []
        for response in self.current_responses:
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
        
        df = pd.DataFrame(data)
        return create_theme_analysis(df)
    
    def save_responses(self, output_dir: str = "output") -> str:
        """Save current responses to file."""
        if not self.current_responses:
            return ""
        
        return save_responses_with_canonical_themes(
            self.current_responses,
            output_dir=output_dir
        ) 