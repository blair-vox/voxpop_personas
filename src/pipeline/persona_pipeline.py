"""
Pipeline for generating personas and running the full workflow.
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import csv
import ast
from openai import OpenAI
from typing import List, Optional

from models.persona import PersonaResponse, Persona
from data.data_processor import (
    sample_personas_major_city,
    construct_persona_with_check,
    map_age_to_abs_age,
    map_income_to_abs_income
)
from generation.persona_gen import (
    generate_persona_response,
    save_responses_with_canonical_themes,
    save_personas_csv
)
from analysis.visualization import (
    create_heatmap,
    create_sentiment_analysis,
    create_impact_analysis,
    create_theme_analysis,
    create_location_analysis,
    create_occupation_analysis,
    create_correlation_analysis,
    create_concern_analysis,
    generate_analysis_report
)
from config.mappings import abs_to_aes_age, abs_to_aes_income, abs_to_aes_sex, abs_to_aes_marital
from utils.persona_utils import generate_synthetic_personas, draw_weighted_personas, validate_demographic_distribution

def run_new_pipeline(num_personas: int, random_state: Optional[int] = None, custom_prompt: Optional[str] = None, question: Optional[str] = None) -> List[PersonaResponse]:
    """
    Run the complete persona generation pipeline.
    
    Args:
        num_personas (int): Number of personas to generate
        random_state (Optional[int]): Random state for reproducibility
        custom_prompt (Optional[str]): Custom prompt to use for persona generation
        question (Optional[str]): The local proposal/question to include in the prompt
    
    Returns:
        List[PersonaResponse]: List of generated persona responses
    """
    # Initialize OpenAI client
    client = OpenAI()
    
    # Set AES and ABS file paths
    aes_file = 'data/aes22_unrestricted_v3.csv'
    abs_file = 'data/Personas_wide.csv'
    
    # Generate synthetic personas
    synthetic_personas_df, _ = generate_synthetic_personas(aes_file, abs_file, num_personas, random_state=random_state)
    
    # Generate responses for each persona
    responses = []
    for idx, row_data in synthetic_personas_df.iterrows():
        print(f"[DEBUG] Row {idx} data:", row_data.to_dict())
        # Use the robust persona construction function
        current_persona = construct_persona_with_check(row_data, synthetic_personas_df)
        print(f"[DEBUG] Constructed Persona object for row {idx}: {current_persona}")
        # Use custom prompt and question if provided, otherwise use default
        response = generate_persona_response(current_persona, client, custom_prompt=custom_prompt, question=question)
        responses.append(response)
    
    return responses 