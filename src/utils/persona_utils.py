import os
from datetime import datetime
import pandas as pd
import numpy as np
import csv
from config.mappings import abs_to_aes_age, abs_to_aes_income, abs_to_aes_sex, abs_to_aes_marital
from data.data_processor import (
    sample_personas_major_city,
    construct_persona_with_check,
    map_age_to_abs_age,
    map_income_to_abs_income
)
from typing import Optional

def generate_synthetic_personas(
    aes_file: str, 
    abs_file: str, 
    num_personas: int = 10, 
    sample_size: int = 100,
    random_state: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic personas from AES and ABS data.
    
    Args:
        aes_file (str): Path to AES data file
        abs_file (str): Path to ABS data file
        num_personas (int): Number of personas to generate
        sample_size (int): Sample size for initial sampling
        random_state (Optional[int]): Random state for reproducibility. If None, uses current time.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple of (synthetic personas, poststratification frame)
    """
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
    income_cols = [
        "Negative income", "Nil income", "$1-$149 ($1-$7,799)", "$150-$299 ($7,800-$15,599)",
        "$300-$399 ($15,600-$20,799)", "$400-$499 ($20,800-$25,999)", "$500-$649 ($26,000-$33,799)",
        "$650-$799 ($33,800-$41,599)", "$800-$999 ($41,600-$51,999)", "$1,000-$1,249 ($52,000-$64,999)",
        "$1,250-$1,499 ($65,000-$77,999)", "$1,500-$1,749 ($78,000-$90,999)", "$1,750-$1,999 ($91,000-$103,999)",
        "$2,000-$2,999 ($104,000-$155,999)", "$3,000-$3,499 ($156,000-$181,999)", "$3,500 or more ($182,000 or more)",
        "Not stated", "Not applicable"
    ]
    income_sums = abs_data[income_cols].sum(axis=1)
    filtered_abs_data = abs_data[income_sums > 0.0].copy()
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
    if 'weight_x' in merged.columns:
        merged = merged.rename(columns={'weight_x': 'weight'})
    if 'weight_y' in merged.columns:
        merged = merged.drop(columns=['weight_y'])
    print(f"Merge successful. Result has {len(merged)} rows")
    
    # 4. Sample personas
    print(f"\nSampling {num_personas} personas...")
    synthetic_personas = sample_personas_major_city(merged, num_personas, random_state=random_state)
    print(f"Successfully sampled {len(synthetic_personas)} personas")
    
    # 5. Create poststratification frame
    poststrat_frame = pd.DataFrame()
    for _, row in synthetic_personas.iterrows():
        print("[DEBUG] Row keys:", row.keys())
        print("[DEBUG] Row sample:", row)
        persona = construct_persona_with_check(row, synthetic_personas)
        poststrat_frame = pd.concat([poststrat_frame, pd.DataFrame([persona.model_dump()])], ignore_index=True)
    return synthetic_personas, poststrat_frame

def draw_weighted_personas(poststrat_frame: pd.DataFrame, num_personas: int = 10, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Draw weighted personas from poststratification frame.
    
    Args:
        poststrat_frame (pd.DataFrame): DataFrame containing persona data
        num_personas (int): Number of personas to sample
        random_state (Optional[int]): Random state for reproducibility. If None, uses current time.
    
    Returns:
        pd.DataFrame: Sampled personas
    """
    if 'weight' in poststrat_frame.columns:
        drawn_personas = poststrat_frame.sample(
            n=num_personas,
            weights='weight',
            replace=True,
            random_state=random_state
        )
    else:
        drawn_personas = poststrat_frame.sample(
            n=num_personas,
            replace=True,
            random_state=random_state
        )
    return drawn_personas

def validate_demographic_distribution(personas: pd.DataFrame) -> None:
    """Validate demographic distribution of personas."""
    age_dist = personas['ABS_AGE_CATEGORY'].value_counts(normalize=True)
    print("\nAge Distribution:")
    print(age_dist)
    gender_dist = personas['ABS_SEX'].value_counts(normalize=True)
    print("\nGender Distribution:")
    print(gender_dist)
    location_dist = personas['SA2 (UR)'].value_counts(normalize=True)
    print("\nLocation Distribution:")
    print(location_dist)
    income_dist = personas['ABS_INCOME'].value_counts(normalize=True)
    print("\nIncome Distribution:")
    print(income_dist)

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