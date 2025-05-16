"""
Main script for processing persona data and interacting with OpenAI API.
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
import csv
import ast
from dotenv import load_dotenv
from openai import OpenAI
import sys

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

# Load environment variables
load_dotenv()

def read_abs_data(file_path: str) -> pd.DataFrame:
    """Read ABS data with proper formatting."""
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

def generate_synthetic_personas(aes_file: str, abs_file: str, num_personas: int = 10, sample_size: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic personas from AES and ABS data."""
    # Read data files with appropriate settings
    aes_data = pd.read_csv(aes_file, low_memory=False)
    abs_data = read_abs_data(abs_file)
    
    # 1. Harmonize AES data
    print("Harmonizing AES data...")
    aes_data['ABS_AGE_CATEGORY'] = aes_data['AGE'].apply(lambda x: map_age_to_abs_age(x, abs_to_aes_age))
    aes_code_to_abs_sex = {v: k for k, v in abs_to_aes_sex.items()}
    aes_data['ABS_SEX'] = aes_data['H1'].map(aes_code_to_abs_sex)
    aes_code_to_abs_marital = {v: k for k, v in abs_to_aes_marital.items()}
    aes_data['ABS_MARITAL'] = aes_data['H8'].map(aes_code_to_abs_marital)
    aes_data['ABS_INCOME'] = aes_data['J6'].apply(lambda x: map_income_to_abs_income(x, abs_to_aes_income))
    
    # Drop rows with missing values in key columns
    aes_data_harmonized = aes_data.dropna(subset=['ABS_AGE_CATEGORY', 'ABS_SEX', 'ABS_MARITAL', 'ABS_INCOME'])
    print(f"After harmonization: {len(aes_data_harmonized)} rows in AES data")
    
    # 2. Prepare ABS data for merge
    print("Preparing ABS data for merge...")
    # Define the income columns as in ABS
    income_cols = [
        "Negative income", "Nil income", "$1-$149 ($1-$7,799)", "$150-$299 ($7,800-$15,599)",
        "$300-$399 ($15,600-$20,799)", "$400-$499 ($20,800-$25,999)", "$500-$649 ($26,000-$33,799)",
        "$650-$799 ($33,800-$41,599)", "$800-$999 ($41,600-$51,999)", "$1,000-$1,249 ($52,000-$64,999)",
        "$1,250-$1,499 ($65,000-$77,999)", "$1,500-$1,749 ($78,000-$90,999)", "$1,750-$1,999 ($91,000-$103,999)",
        "$2,000-$2,999 ($104,000-$155,999)", "$3,000-$3,499 ($156,000-$181,999)", "$3,500 or more ($182,000 or more)",
        "Not stated", "Not applicable"
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
    
    # 3. Merge datasets
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
    
    # 4. Sample personas
    print(f"\nSampling {num_personas} personas...")
    synthetic_personas = sample_personas_major_city(merged, num_personas)
    print(f"Successfully sampled {len(synthetic_personas)} personas")
    
    # 5. Create poststratification frame
    poststrat_frame = pd.DataFrame()
    for _, row in synthetic_personas.iterrows():
        persona = construct_persona_with_check(row, synthetic_personas)
        poststrat_frame = pd.concat([poststrat_frame, pd.DataFrame([persona.model_dump()])], ignore_index=True)
    
    return synthetic_personas, poststrat_frame

def generate_poststrat_frame(synthetic_personas: pd.DataFrame) -> pd.DataFrame:
    """Generate poststratification frame from synthetic personas."""
    poststrat_frame = pd.DataFrame()
    for _, row in synthetic_personas.iterrows():
        persona = construct_persona_with_check(row, synthetic_personas)
        poststrat_frame = pd.concat([poststrat_frame, pd.DataFrame([persona.model_dump()])], ignore_index=True)
    return poststrat_frame

def draw_weighted_personas(poststrat_frame: pd.DataFrame, num_personas: int = 10) -> pd.DataFrame:
    """Draw weighted personas from poststratification frame."""
    if 'weight' in poststrat_frame.columns:
        drawn_personas = poststrat_frame.sample(
            n=num_personas,
            weights='weight',
            replace=True,
            random_state=42
        )
    else:
        drawn_personas = poststrat_frame.sample(
            n=num_personas,
            replace=True,
            random_state=42
        )
    return drawn_personas

def validate_demographic_distribution(personas: pd.DataFrame) -> None:
    """Validate demographic distribution of personas."""
    # Check age distribution
    age_dist = personas['age'].value_counts(normalize=True)
    print("\nAge Distribution:")
    print(age_dist)
    
    # Check gender distribution
    gender_dist = personas['gender'].value_counts(normalize=True)
    print("\nGender Distribution:")
    print(gender_dist)
    
    # Check location distribution
    location_dist = personas['location'].value_counts(normalize=True)
    print("\nLocation Distribution:")
    print(location_dist)
    
    # Check income distribution
    income_dist = personas['income'].value_counts(normalize=True)
    print("\nIncome Distribution:")
    print(income_dist)

def create_toy_datasets(aes_file: str = "toy_aes_data.csv", abs_file: str = "toy_abs_data.csv", 
                       aes_input: str = "data/aes22_unrestricted_v3.csv", abs_input: str = "data/Personas_wide.csv", 
                       num_rows: int = 100) -> tuple[str, str]:
    """Create toy datasets for testing."""
    # Read input files
    aes_data = pd.read_csv(aes_input)
    abs_data = pd.read_csv(abs_input)
    
    # Sample rows
    aes_sample = aes_data.sample(n=num_rows, random_state=42)
    abs_sample = abs_data.sample(n=num_rows, random_state=42)
    
    # Save toy datasets
    aes_sample.to_csv(aes_file, index=False)
    abs_sample.to_csv(abs_file, index=False)
    
    return aes_file, abs_file

def main():
    print("Select which version to run:")
    print("1. New code (modular)")
    print("2. Legacy code (original)")
    choice = input("Enter 1 or 2: ").strip()
    try:
        num_personas = int(input("How many personas to generate? ").strip())
    except ValueError:
        print("Invalid number. Exiting.")
        sys.exit(1)

    if choice == "2":
        # Run legacy code
        import main_legacy
        main_legacy.main(num_personas)
    else:
        # Run new code (current pipeline)
        run_new_pipeline(num_personas)

def run_new_pipeline(num_personas):
    # Central configuration
    client = OpenAI()
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Generate synthetic personas
    synthetic_personas_df, poststrat_frame = generate_synthetic_personas(
        aes_file="data/aes22_unrestricted_v3.csv",
        abs_file="data/Personas_wide.csv",
        num_personas=num_personas
    )

    # Draw weighted personas
    personas_df_for_response = draw_weighted_personas(poststrat_frame, num_personas=num_personas)

    # Validate demographic distribution
    validate_demographic_distribution(personas_df_for_response)

    # Generate responses for each persona
    responses = []
    for _, row_data in personas_df_for_response.iterrows():
        issues_val = row_data.get('issues', [])
        if isinstance(issues_val, str):
            try:
                parsed_issues = ast.literal_eval(issues_val)
                if isinstance(parsed_issues, list):
                    issues_val = parsed_issues
                else:
                    issues_val = [str(parsed_issues)]
            except (ValueError, SyntaxError):
                issues_val = [str(issues_val)]
        elif not isinstance(issues_val, list):
            issues_val = [str(issues_val)]

        current_persona = Persona(
            name=str(row_data.get('name', f"Persona_{row_data.name}")),
            age=str(row_data.get('age', 'Unknown')),
            gender=str(row_data.get('gender', 'Unknown')),
            location=str(row_data.get('location', 'Unknown')),
            income=str(row_data.get('income', 'Unknown')),
            tenure=str(row_data.get('tenure', 'Unknown')),
            job_tenure=str(row_data.get('job_tenure', 'Unknown')),
            occupation=str(row_data.get('occupation', 'Unknown')),
            education=str(row_data.get('education', 'Unknown')),
            transport=str(row_data.get('transport', 'Unknown')),
            marital_status=str(row_data.get('marital_status', 'Unknown')),
            partner_activity=str(row_data.get('partner_activity', 'Unknown')),
            household_size=str(row_data.get('household_size', 'Unknown')),
            family_payments=str(row_data.get('family_payments', 'Unknown')),
            child_care_benefit=str(row_data.get('child_care_benefit', 'Unknown')),
            investment_properties=str(row_data.get('investment_properties', 'Unknown')),
            transport_infrastructure=str(row_data.get('transport_infrastructure', 'Unknown')),
            political_leaning=str(row_data.get('political_leaning', 'Unknown')),
            trust=str(row_data.get('trust', 'Unknown')),
            issues=issues_val,
            engagement=str(row_data.get('engagement', 'Unknown'))
        )
        response = generate_persona_response(current_persona, client)
        responses.append(response)

    # Save responses and generate analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_responses_file_path = save_responses_with_canonical_themes(responses, client, output_dir=output_dir)

    try:
        all_responses_df = pd.read_json(combined_responses_file_path)
        print(f"Successfully loaded combined responses from {combined_responses_file_path} for visualization.")
        if 'survey_response' in all_responses_df.columns and not all_responses_df.empty and isinstance(all_responses_df['survey_response'].iloc[0], dict):
            print("Flattening survey_response data for correlation analysis...")
            survey_data_df = pd.json_normalize(all_responses_df['survey_response'])
            all_responses_df = pd.concat([
                all_responses_df.drop(columns=['survey_response']),
                survey_data_df
            ], axis=1)
            print(f"Columns after flattening: {all_responses_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading or processing combined responses JSON for visualization: {e}")
        all_responses_df = pd.DataFrame([r.model_dump() for r in responses])
        if 'survey_response' in all_responses_df.columns and not all_responses_df.empty and isinstance(all_responses_df['survey_response'].iloc[0], dict):
            try:
                print("Flattening survey_response data in fallback...")
                survey_data_df = pd.json_normalize(all_responses_df['survey_response'])
                all_responses_df = pd.concat([
                    all_responses_df.drop(columns=['survey_response']),
                    survey_data_df
                ], axis=1)
            except Exception as ex_fallback:
                print(f"Error flattening survey_response in fallback: {ex_fallback}")

    if not all_responses_df.empty:
        create_heatmap(all_responses_df, output_dir, timestamp)
        create_sentiment_analysis(all_responses_df, output_dir, timestamp)
        create_impact_analysis(all_responses_df, output_dir, timestamp)
        create_theme_analysis(all_responses_df, output_dir, timestamp)
        create_location_analysis(all_responses_df, output_dir, timestamp)
        create_occupation_analysis(all_responses_df, output_dir, timestamp)
        create_correlation_analysis(all_responses_df, output_dir, timestamp)
        create_concern_analysis(all_responses_df, output_dir, timestamp)
        generate_analysis_report(all_responses_df, output_dir, timestamp)
    else:
        print("Skipping visualizations as the responses DataFrame is empty.")

if __name__ == "__main__":
    main() 