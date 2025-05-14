def read_abs_data(file_path: str) -> pd.DataFrame:
    """
    Read ABS data from their CSV export format.
    """
    # Define the column names in the correct order
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

    # Data starts at line 11 (index 10), so skip the first 10 rows and don't use any header from the file
    df = pd.read_csv(
        file_path,
        skiprows=10,
        header=None,
        names=columns,
        encoding='utf-8',
        skipinitialspace=True,
        # low_memory=False,
        on_bad_lines='warn',
        quoting=csv.QUOTE_MINIMAL,
        sep=',',
        engine='python'
    )

    # Forward fill demographic columns
    for col in columns[:6]:
        df[col] = df[col].replace('', np.nan).ffill()

    # Convert income columns to numeric
    for col in columns[6:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are completely empty
    df = df.dropna(how='all', subset=columns[6:])

    print("Successfully loaded ABS dataset.")
    return df

# Usage
abs_data = read_abs_data("../data/Personas_wide.csv")
display(abs_data.head())

aes_input: str = "../data/aes22_unrestricted_v3.csv"

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
  

# Verify required AES columns exist
required_aes_cols = ["AGE", "H1", "STATE", "J6", "B1", "weight_final"]
missing_aes = [col for col in required_aes_cols if col not in aes_data.columns]
if missing_aes:
    print(f"Missing required columns in AES data: {missing_aes}")

# Calculate the sum of income columns for each row
income_sums = abs_data[income_cols].sum(axis=1)

# Filter out rows where the sum is zero or NaN
filtered_abs_data = abs_data[income_sums > 0.0].copy()

print(f"Filtered ABS data shape: {filtered_abs_data.shape}")
display(filtered_abs_data.head())

# Melt the income columns into a long format
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

# Filter out zero or missing weights
poststrat_frame = melted[melted['weight'] > 0].copy()

print(poststrat_frame.head())
print(f"Number of strata (demographics x income): {len(poststrat_frame)}")

# List of age groups to exclude (all under 18)
exclude_ages = [
    "0-4 years", "5-9 years", "10-14 years", "15-17 years"
]

# Filter the poststrat_frame to only include adults
adult_poststrat_frame = poststrat_frame[
    ~poststrat_frame["AGE5P Age in Five Year Groups"].isin(exclude_ages)
].copy()

print(f"Number of adult strata: {len(adult_poststrat_frame)}")
print(adult_poststrat_frame["AGE5P Age in Five Year Groups"].unique())

N = 1000  # Number of personas to draw

def draw_personas(poststrat_frame, N):

    probabilities = poststrat_frame['weight'] / poststrat_frame['weight'].sum()
    drawn_indices = np.random.choice(
        poststrat_frame.index,
        size=N,
        replace=True,
        p=probabilities
    )
    personas = poststrat_frame.loc[drawn_indices].reset_index(drop=True)

    return personas

personas = draw_personas(poststrat_frame=adult_poststrat_frame, N=1000)
print(personas.head())

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

age_order = {
    "0-4 years",
    "5-9 years",
    "10-14 years",
    "15-19 years",
    "20-24 years",
    "25-29 years",
    "30-34 years",
    "35-39 years",
    "40-44 years",
    "45-49 years",
    "50-54 years",
    "55-59 years",
    "60-64 years",
    "65-69 years",
    "70-74 years",
    "75-79 years",
    "80-84 years",
    "85-89 years",
    "90-95 years",
    "96-99 years",
    "100 years and over" 
}

abs_to_aes_sex = {
    "Male" : 1,
    "Female" : 2,
    
}

def map_age_to_abs_age(age, abs_to_aes_age):
    for abs_cat, (min_age, max_age) in abs_to_aes_age.items():
        if pd.notnull(age) and min_age <= age <= max_age:
            return abs_cat
    return None

aes_data['ABS_AGE_CATEGORY'] = aes_data['AGE'].apply(lambda x: map_age_to_abs_age(x, abs_to_aes_age))

# Invert the mapping: {1: "Male", 2: "Female"}
aes_code_to_abs_sex = {v: k for k, v in abs_to_aes_sex.items()}

# Now map the AES column to ABS labels
aes_data['ABS_SEX'] = aes_data['H1'].map(aes_code_to_abs_sex)

# Invert the mapping for code → label
aes_data['ABS_MARITAL'] = aes_data['H8'].map({v: k for k, v in abs_to_aes_marital.items()})

def map_income_to_abs_income(aes_income, abs_to_aes_income):
    for abs_cat, aes_val in abs_to_aes_income.items():
        if isinstance(aes_val, tuple):
            if pd.notnull(aes_income) and aes_val[0] <= aes_income <= aes_val[1]:
                return abs_cat
        elif aes_income == aes_val:
            return abs_cat
    return None

aes_data['ABS_INCOME'] = aes_data['J6'].apply(lambda x: map_income_to_abs_income(x, abs_to_aes_income))

aes_data_harmonized = aes_data.dropna(subset=['ABS_AGE_CATEGORY', 'ABS_SEX', 'ABS_MARITAL', 'ABS_INCOME'])

poststrat_for_merge = poststrat_frame.rename(columns={
    "AGE5P Age in Five Year Groups": "ABS_AGE_CATEGORY",
    "SEXP Sex": "ABS_SEX",
    "MSTP Registered Marital Status": "ABS_MARITAL",
    "Income Level": "ABS_INCOME"
    # Add more if you harmonized other variables
})

# Optionally, drop the AES weight column if you don't need it
if 'weight_y' in merged.columns:
    merged = merged.drop(columns=['weight_y'])

# For clarity, you can also rename weight_x back to weight
merged = merged.rename(columns={'weight_x': 'weight'})

import numpy as np

# simple sampling
N = 1000  # Number of personas to draw

# Normalize weights
probabilities = merged['weight'] / merged['weight'].sum()

# Draw indices of the merged DataFrame, proportional to ABS weights
drawn_indices = np.random.choice(
    merged.index,
    size=N,
    replace=True,
    p=probabilities
)
sampled = merged.loc[drawn_indices].reset_index(drop=True)

#better sampling
N = 1000

# First, create a DataFrame of unique strata with their weights
strata_cols = ["ABS_AGE_CATEGORY", "ABS_SEX", "ABS_MARITAL", "ABS_INCOME"]  # add more if needed
strata_weights = merged.groupby(strata_cols)['weight'].first().reset_index()

# Draw N strata according to their weights
strata_probs = strata_weights['weight'] / strata_weights['weight'].sum()
drawn_strata = strata_weights.sample(
    n=N,
    replace=True,
    weights=strata_probs
).reset_index(drop=True)

# For each drawn stratum, randomly select one AES respondent from that stratum
personas = []
for _, stratum in drawn_strata.iterrows():
    # Find all AES respondents in this stratum
    matches = merged[
        (merged['ABS_AGE_CATEGORY'] == stratum['ABS_AGE_CATEGORY']) &
        (merged['ABS_SEX'] == stratum['ABS_SEX']) &
        (merged['ABS_MARITAL'] == stratum['ABS_MARITAL']) &
        (merged['ABS_INCOME'] == stratum['ABS_INCOME'])
    ]
    # Randomly select one AES respondent
    persona = matches.sample(n=1)
    personas.append(persona)

# Concatenate all sampled personas into a DataFrame
personas_df = pd.concat(personas, ignore_index=True)

# big city limits sampling
import pandas as pd

def sample_personas_major_city(
    merged, N,
    strata_cols=["ABS_AGE_CATEGORY", "ABS_SEX", "ABS_MARITAL", "ABS_INCOME"],
    major_city_col="J5",
    major_city_value=5
):
    """
    Sample N personas from the merged frame, ensuring each persona is from a major city (J5 == 5).
    
    Parameters:
        merged: DataFrame containing merged ABS and AES data.
        N: Number of personas to sample.
        strata_cols: List of columns defining the strata.
        major_city_col: Column in AES data indicating major city status.
        major_city_value: Value in major_city_col indicating major city (default 5).
    
    Returns:
        personas_df: DataFrame of sampled personas.
    """
    # Get unique strata and their weights
    strata_weights = merged.groupby(strata_cols)['weight'].first().reset_index()
    strata_probs = strata_weights['weight'] / strata_weights['weight'].sum()

    personas = []
    attempts = 0
    max_attempts = N * 10  # Prevent infinite loops

    while len(personas) < N and attempts < max_attempts:
        # Draw one stratum according to weights
        stratum = strata_weights.sample(
            n=1,
            weights=strata_probs
        ).iloc[0]

        # Find all AES respondents in this stratum with J5 == 5
        matches = merged
        for col in strata_cols:
            matches = matches[matches[col] == stratum[col]]
        matches = matches[matches[major_city_col] == major_city_value]

        if not matches.empty:
            persona = matches.sample(n=1)
            personas.append(persona)
        # else: redraw (do nothing, try again)
        attempts += 1

    if len(personas) < N:
        print(f"Warning: Only {len(personas)} personas could be sampled with {major_city_col} == {major_city_value} after {max_attempts} attempts.")

    if personas:
        personas_df = pd.concat(personas, ignore_index=True)
    else:
        personas_df = pd.DataFrame()  # Return empty DataFrame if none found

    return personas_df

N = 1000
personas_df = sample_personas_major_city(merged, N)
print(personas_df.head())

#attempting to build prompt
from pydantic import BaseModel
from typing import List, Dict, Optional

class SurveyResponse(BaseModel):
    support_level: int
    impact_on_housing: int
    impact_on_transport: int
    impact_on_community: int
    key_concerns: List[str]
    suggested_improvements: List[str]

class PersonaResponse(BaseModel):
    persona_details: Dict[str, str]
    narrative_response: str
    survey_response: SurveyResponse
    timestamp: str
    sentiment_score: Optional[float] = None
    key_themes: Optional[List[str]] = None

# If you want to use the Persona class for more detailed personas:
class Persona(BaseModel):
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

    from textblob import TextBlob

def analyze_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def generate_focus_group_response(persona_row):
    # Example: create a prompt for the LLM or just a mock response
    # Here, we'll just create a mock narrative and survey response
    narrative = f"As a {persona_row['ABS_AGE_CATEGORY']} year old {persona_row['ABS_SEX']} with income {persona_row['ABS_INCOME']}, I feel strongly about this issue."
    survey = SurveyResponse(
        support_level=3,
        impact_on_housing=4,
        impact_on_transport=2,
        impact_on_community=3,
        key_concerns=["Parking", "Affordability"],
        suggested_improvements=["More public transport", "Better planning"]
    )
    sentiment_score = analyze_sentiment(narrative)
    return PersonaResponse(
        persona_details=persona_row.to_dict(),
        narrative_response=narrative,
        survey_response=survey,
        timestamp=pd.Timestamp.now().isoformat(),
        sentiment_score=sentiment_score,
        key_themes=None  # Or use your theme extraction if desired
    )

# J1
tenure = {
    1: "Own outright",
    2: "Own, paying off mortgage",
    3: "Rent from private landlord or real estate agent",
    4: "Rent from public housing authority",
    5: "Other (boarding, living at home, etc.)",
    999: "Item skipped"
}
# G5_D
job_type = {
    1:	'Upper managerial',
    2:	'Middle managerial',
    3:	'Lower managerial',
    4:	'Supervisory',
    5:	'Non-supervisory',
    999:	'Item skipped'
}
# G5_E
job_tenure = {
    1:	'Self-employed',
    2:  'Employee in private company or business',
    3:	'Employee of Federal / State / Local Government',
    4:	'Employee in family business or farm',
    999:	'Item skipped'
}
# G3_EDU
uni_level = {
    1:	'University',
    2:	'Non-University',
    999:	'Item skipped'
}
# G3
edu_level = {
    1: "Bachelor degree or higher",
    2: "Advanced diploma or diploma",
    3: "Certificate III/IV",
    4: "Year 12 or equivalent",
    5: "Year 11 or below",
    6: "No formal education",
    999: "Item skipped"
}
#B9_1
political_leaning = {
    1: "Liberal",
    2: "Labor",
    3: "National Party",
    4: "Greens",
    **{i: "Other party (please specify)" for i in range(5, 98)},
    999: "Item skipped"
}
# C6
trust_gov = {
    1: "Usually look after themselves",
    2: "Sometimes look after themselves",
    3: "Sometimes can be trusted to do the right thing",
    4: "Usually can be trusted to do the right thing",
    999: "Item skipped"
}

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

print(get_mapping_dict('A1'))


def row_to_persona(row):
    return Persona(
        name=f"Persona_{row.name}",
        age=row.get('ABS_AGE_CATEGORY', 'Unknown'),
        gender=row.get('ABS_SEX', 'Unknown'),
        location=row.get('SA2 (UR)', 'Unknown'),
        income=row.get('ABS_INCOME', 'Unknown'),
        tenure=tenure.get(row.get('J1', 999), 'Unknown'),
        job_tenure=job_tenure.get(row.get('G5_E', 999), 'Unknown'),
        occupation=job_type.get(row.get('G5_D', 999), 'Unknown'),
        education=edu_level.get(row.get('G3', 999), 'Unknown'),
        political_leaning=political_leaning.get(row.get('B9_1', 999), 'Unknown'), 
        trust = trust_gov.get(row.get('C6', 999), 'Unknown'),
        issues=[],
        engagement="Unknown"
    )

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


engagement_text = construct_engagement_string(personas_df.iloc[0], personas_df)
issues_text = construct_issues_string(personas_df.iloc[0], personas_df)
print(engagement_text)
print(issues_text)

    
# attempt at a prompt
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
- Job Tenure: {persona.job_tenure}
- Occupation: {persona.occupation}  
- Education: {persona.education}  
- Political Leaning: {persona.political_leaning}  
- Trust in Government: {persona.trust}  
- Key Issues: {issues_text}  
- Engagement Level: {engagement_text}

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
    )# Initialize your OpenAI client
