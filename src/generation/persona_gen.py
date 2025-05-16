"""
Persona generation functions.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI
from textblob import TextBlob
import re

from models.persona import Persona, PersonaResponse, SurveyResponse
from config.prompts.persona_prompts import SYSTEM_PROMPT, PERSONA_TEMPLATE, THEME_EXTRACTION_PROMPT

def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of text."""
    return TextBlob(text).sentiment.polarity

def extract_key_themes(text: str, client: OpenAI) -> List[str]:
    """Extract key themes from text."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": THEME_EXTRACTION_PROMPT},
            {"role": "user", "content": text}
        ]
    )
    themes = response.choices[0].message.content.split(',')
    return [theme.strip() for theme in themes]

# --- Canonical Theme Grouping ---
def flatten_theme_mapping(theme_mapping: Dict) -> List[str]:
    """Flatten nested theme mapping into a list of valid categories."""
    categories = []
    for key, value in theme_mapping.items():
        if isinstance(value, dict):
            # Handle nested categories
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    categories.extend(subvalue)
                else:
                    categories.append(subvalue)
        elif isinstance(value, list):
            categories.extend(value)
        else:
            categories.append(value)
    return [cat.replace('_', ' ').lower() for cat in categories]

def get_canonical_theme_mapping(
    unique_themes: List[str],
    method: str = "sbert",  # Changed default back to sbert since BART will be in map_themes_to_canonical
    client: Optional[OpenAI] = None
) -> Dict[str, List[str]]:
    """
    Group themes into natural categories using either sentence-BERT ('sbert') or OpenAI ('openai').
    BART classification is handled separately in map_themes_to_canonical.
    """
    print(f"\nUsing {method.upper()} for theme grouping...")
    
    if not unique_themes:
        print("No themes to group.")
        return {}

    if method == "sbert":
        print("Initializing sentence-BERT for theme clustering...")
        from sklearn.cluster import AgglomerativeClustering
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # 1. Embed themes
        print("Generating embeddings for themes...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(unique_themes)

        # 2. Cluster embeddings
        n_clusters = min(len(unique_themes), 10)
        print(f"Clustering {len(unique_themes)} themes into {n_clusters} clusters...")
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings)

        # 3. Group themes by cluster
        clusters = {}
        for label, theme in zip(labels, unique_themes):
            clusters.setdefault(label, []).append(theme)

        # 4. Pick canonical label for each cluster
        canonical_mapping = {}
        for themes in clusters.values():
            canonical = min(themes, key=len)
            canonical_mapping[canonical] = themes

        print(f"Clustering complete. Found {len(canonical_mapping)} clusters.")
        return canonical_mapping

    elif method == "openai":
        if client is None:
            print("Error: OpenAI client must be provided for method='openai'")
            raise ValueError("OpenAI client must be provided for method='openai'.")

        print("Using OpenAI for theme grouping...")
        import json as pyjson
        import ast

        prompt = (
            "Here is a list of themes from survey responses:\n"
            f"{pyjson.dumps(unique_themes)}\n"
            "Please group these themes into natural categories. For each category, provide:\n"
            "- The canonical category name\n"
            "- The list of original themes that belong to it\n"
            "Return the result as a JSON dictionary where each key is the canonical category and the value is a list of original themes."
        )
        print("Sending themes to OpenAI for grouping...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for data analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        try:
            mapping = pyjson.loads(content)
            print("Successfully parsed OpenAI response")
        except Exception:
            try:
                mapping = ast.literal_eval(content)
                print("Successfully parsed OpenAI response using ast.literal_eval")
            except Exception:
                print("[ERROR] Could not parse OpenAI canonical theme mapping. Returning empty mapping.")
                mapping = {}
        return mapping

    else:
        print(f"Error: Unknown method '{method}'")
        raise ValueError(f"Unknown method: {method}")

def map_themes_to_canonical(
    key_themes: List[str], 
    canonical_mapping: Dict[str, List[str]],
    method: str = "bart",
    config_path: str = "themes_config.yaml",
    confidence_threshold: float = 0.3  # Threshold for including a category
) -> List[Dict[str, str]]:
    """
    Map themes to canonical categories using either:
    - BART: Classify directly into predefined categories (can assign multiple categories per theme)
    - Other methods: Use the provided canonical_mapping from natural grouping
    """
    if not key_themes:
        return []

    if method == "bart":
        print("\nUsing BART for theme classification...")
        import yaml
        from transformers import pipeline

        # Load theme mapping from config
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Successfully loaded theme configuration from {config_path}")
        except Exception as e:
            print(f"Error loading theme configuration: {e}")
            return [{"canonical": "Other", "original": theme} for theme in key_themes]
        
        # Get valid categories from config
        valid_categories = flatten_theme_mapping(config['theme_mapping'])
        print(f"Found {len(valid_categories)} valid categories in configuration")
        
        # Initialize BART classifier
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=config['ml_models_config']['categories_classifier']['model_name']
            )
            print(f"Initialized BART classifier with model: {config['ml_models_config']['categories_classifier']['model_name']}")
        except Exception as e:
            print(f"Error initializing BART classifier: {e}")
            return [{"canonical": "Other", "original": theme} for theme in key_themes]
        
        # Classify each theme
        print(f"\nClassifying {len(key_themes)} themes...")
        result = []
        for i, theme in enumerate(key_themes, 1):
            if i % 10 == 0:  # Progress update every 10 themes
                print(f"Processed {i}/{len(key_themes)} themes...")
            
            # Get classification scores for all valid categories
            classification = classifier(
                theme,
                valid_categories,
                multi_label=True  # Enable multi-label classification
            )
            
            # Get all categories above the confidence threshold
            high_confidence_categories = [
                (label, score) 
                for label, score in zip(classification['labels'], classification['scores'])
                if score >= confidence_threshold
            ]
            
            if high_confidence_categories:
                # Sort by confidence score
                high_confidence_categories.sort(key=lambda x: x[1], reverse=True)
                
                # Add each high-confidence category as a separate mapping
                for category, confidence in high_confidence_categories:
                    result.append({
                        "canonical": category,
                        "original": theme,
                        "confidence": confidence
                    })
            else:
                # If no categories meet the threshold, use the highest scoring one
                best_category = classification['labels'][0]
                confidence = classification['scores'][0]
                result.append({
                    "canonical": best_category,
                    "original": theme,
                    "confidence": confidence
                })
                print(f"Low confidence ({confidence:.2f}) for theme: '{theme}' -> '{best_category}'")
        
        print(f"\nTheme classification complete.")
        return result

    else:
        # Use the provided canonical_mapping from natural grouping
        result = []
        for theme in key_themes:
            found = False
            for canonical, originals in canonical_mapping.items():
                if theme in originals:
                    result.append({"canonical": canonical, "original": theme})
                    found = True
                    break
            if not found:
                result.append({"canonical": "Other", "original": theme})
        return result

def generate_persona_response(persona: Persona, client: OpenAI) -> PersonaResponse:
    """Generate response for a persona."""
    # Format persona details for the prompt
    persona_details = {
        'name': persona.name,
        'age': persona.age,
        'gender': persona.gender,
        'location': persona.location,
        'income': persona.income,
        'tenure': persona.tenure,
        'job_tenure': persona.job_tenure,
        'occupation': persona.occupation,
        'education': persona.education,
        'transport': persona.transport,
        'marital_status': persona.marital_status,
        'partner_activity': persona.partner_activity,
        'household_size': persona.household_size,
        'family_payments': persona.family_payments,
        'child_care_benefit': persona.child_care_benefit,
        'investment_properties': persona.investment_properties,
        'transport_infrastructure': persona.transport_infrastructure,
        'political_leaning': persona.political_leaning,
        'trust': persona.trust,
        'issues': persona.issues,
        'engagement': persona.engagement
    }
    
    # Generate response using OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PERSONA_TEMPLATE.format(persona_details=json.dumps(persona_details, indent=2))}
        ]
    )
    
    # Parse the response
    response_text = response.choices[0].message.content
    narrative_response = response_text.split('SURVEY RESPONSE:')[0].replace('NARRATIVE RESPONSE:', '').strip()
    survey_response_text = response_text.split('SURVEY RESPONSE:')[1].strip()
    
    # Parse survey response
    survey_lines = survey_response_text.split('\n')
    survey_data = {}
    for line in survey_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            # Normalize "impact_on_housing_affordability" to "impact_on_housing"
            if key == "impact_on_housing_affordability":
                key = "impact_on_housing"
            value = value.strip()
            if key in ['support_level', 'impact_on_housing', 'impact_on_transport', 'impact_on_community']:
                try:
                    survey_data[key] = int(value)
                except ValueError:
                    print(f"[WARNING] Could not parse integer value for {key}: '{value}'. Using 0 as default.")
                    survey_data[key] = 0  # Default for unparseable int
            elif key in ['key_concerns', 'suggested_improvements']:
                survey_data[key] = [item.strip() for item in value.split(',')]
    
    # Expected keys for SurveyResponse
    expected_numeric_keys = ['support_level', 'impact_on_housing', 'impact_on_transport', 'impact_on_community']
    expected_list_keys = ['key_concerns', 'suggested_improvements']

    # Check for missing keys and apply defaults
    for k in expected_numeric_keys:
        if k not in survey_data:
            print(f"[WARNING] Survey data missing key: '{k}'. Using 0 as default.")
            survey_data[k] = 0
    
    for k in expected_list_keys:
        if k not in survey_data:
            print(f"[WARNING] Survey data missing key: '{k}'. Using empty list as default.")
            survey_data[k] = []

    # Create survey response object
    survey_response = SurveyResponse(
        support_level=survey_data.get('support_level', 0),
        impact_on_housing=survey_data.get('impact_on_housing', 0),
        impact_on_transport=survey_data.get('impact_on_transport', 0),
        impact_on_community=survey_data.get('impact_on_community', 0),
        key_concerns=survey_data.get('key_concerns', []),
        suggested_improvements=survey_data.get('suggested_improvements', [])
    )
    
    # Analyze sentiment and extract themes
    sentiment_score = analyze_sentiment(narrative_response)
    key_themes = extract_key_themes(narrative_response, client)
    
    # Create and return persona response
    return PersonaResponse(
        persona_details=persona_details,
        narrative_response=narrative_response,
        survey_response=survey_response,
        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
        sentiment_score=sentiment_score,
        key_themes=key_themes
    )

def generate_aes_persona_response(persona: pd.Series, client: OpenAI) -> PersonaResponse:
    """Generate response for an AES persona."""
    # Convert persona series to Persona object
    persona_obj = Persona(
        name=f"Persona_{persona.name}",
        age=persona.get('ABS_AGE_CATEGORY', 'Unknown'),
        gender=persona.get('ABS_SEX', 'Unknown'),
        location=persona.get('SA2 (UR)', 'Unknown'),
        income=persona.get('ABS_INCOME', 'Unknown'),
        tenure=persona.get('J1', 'Unknown'),
        job_tenure=persona.get('G5_E', 'Unknown'),
        occupation=persona.get('G5_D', 'Unknown'),
        education=persona.get('G3', 'Unknown'),
        transport=persona.get('transport', 'Unknown'),
        marital_status=persona.get('H8', 'Unknown'),
        partner_activity=persona.get('I1', 'Unknown'),
        household_size=persona.get('W1', 'Unknown'),
        family_payments=persona.get('J8_1', 'Unknown'),
        child_care_benefit=persona.get('J8_2', 'Unknown'),
        investment_properties=persona.get('J2', 'Unknown'),
        transport_infrastructure=persona.get('D8_9', 'Unknown'),
        political_leaning=persona.get('B9_1', 'Unknown'),
        trust=persona.get('C6', 'Unknown'),
        issues=persona.get('issues', []),
        engagement=persona.get('engagement', 'Unknown')
    )
    
    # Generate response using the Persona object
    return generate_persona_response(persona_obj, client)

def save_responses_with_canonical_themes(
    responses: List[PersonaResponse], 
    client: OpenAI, 
    output_dir: str = "output",
    grouping_method: str = "sbert",  # Method for natural grouping
    classification_method: str = "bart",  # Method for mapping to categories
    config_path: str = "themes_config.yaml"
):
    """Save responses to a single combined JSON file, including canonical theme mapping."""
    print(f"\nSaving responses with canonical themes...")
    print(f"Grouping method: {grouping_method.upper()}")
    print(f"Classification method: {classification_method.upper()}")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Collect all unique themes
    print("Collecting unique themes from responses...")
    all_themes = set()
    for response in responses:
        if hasattr(response, 'key_themes') and response.key_themes:
            all_themes.update(response.key_themes)
    unique_themes = sorted(list(all_themes))
    print(f"Found {len(unique_themes)} unique themes")
    
    # Get natural groupings if not using BART
    canonical_mapping = {}
    if classification_method != "bart":
        print(f"\nGenerating natural theme groupings...")
        canonical_mapping = get_canonical_theme_mapping(
            unique_themes, 
            method=grouping_method,
            client=client
        )
    
    # Update each response with canonical_themes
    print("\nUpdating responses with canonical themes...")
    for response in responses:
        if hasattr(response, 'key_themes') and response.key_themes:
            response.canonical_themes = map_themes_to_canonical(
                response.key_themes, 
                canonical_mapping,
                method=classification_method,
                config_path=config_path
            )
        else:
            response.canonical_themes = []
    
    # Save combined responses
    print("\nSaving responses to file...")
    combined_data = [response.model_dump() for response in responses]
    combined_file_path = os.path.join(output_dir, f'all_responses_{timestamp}.json')
    with open(combined_file_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"All responses (with canonical themes) saved to: {combined_file_path}")
    
    # Save the mapping for reference
    mapping_file_path = os.path.join(output_dir, f'canonical_theme_mapping_{timestamp}.json')
    with open(mapping_file_path, 'w') as f:
        json.dump(canonical_mapping, f, indent=2)
    print(f"Canonical theme mapping saved to: {mapping_file_path}")
    
    return combined_file_path

def save_personas_csv(personas: List[Persona], output_dir: str, timestamp: str) -> None:
    """Save personas to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    personas_data = [persona.model_dump() for persona in personas]
    df = pd.DataFrame(personas_data)
    df.to_csv(os.path.join(output_dir, f'personas_{timestamp}.csv'), index=False) 