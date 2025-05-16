"""
Mapping dictionaries for various demographic and survey data.
"""

# Tenure
tenure = {
    1: "Own outright",
    2: "Own, paying off mortgage",
    3: "Rent from private landlord or real estate agent",
    4: "Rent from public housing authority",
    5: "Other (boarding, living at home, etc.)",
    999: "Item skipped"
}

# Job Type
job_type = {
    1: 'Upper managerial',
    2: 'Middle managerial',
    3: 'Lower managerial',
    4: 'Supervisory',
    5: 'Non-supervisory',
    999: 'Item skipped'
}

# Job Tenure
job_tenure = {
    1: 'Self-employed',
    2: 'Employee in private company or business',
    3: 'Employee of Federal / State / Local Government',
    4: 'Employee in family business or farm',
    999: 'Item skipped'
}

# Education Level
edu_level = {
    1: "Bachelor degree or higher",
    2: "Advanced diploma or diploma",
    3: "Certificate III/IV",
    4: "Year 12 or equivalent",
    5: "Year 11 or below",
    6: "No formal education",
    999: "Item skipped"
}

# Marital Status
marital_status = {
    1: "Never married",
    2: "Married",
    3: "Widowed",
    4: "Divorced/Separated",
    5: "De facto",
    999: "Item skipped"
}

# Partner Activity
partner_activity = {
    1: "Working full-time",
    2: "Working part-time",
    3: "Caring responsibilities",
    4: "Retired",
    5: "Student",
    6: "Unemployed",
    999: "Item skipped"
}

# Household Size
household_size = {
    1: "1 person",
    2: "2 people",
    3: "3 people",
    4: "4 people",
    5: "5 or more",
    999: "Item skipped"
}

# Family Payments
family_payments = {
    1: "Yes",
    2: "No",
    999: "Item skipped"
}

# Child Care Benefit
child_care_benefit = {
    1: "Yes",
    2: "No",
    999: "Item skipped"
}

# Investment Properties
investment_properties = {
    1: "Yes",
    2: "No",
    999: "Item skipped"
}

# Transport Infrastructure
transport_infrastructure = {
    1: "Much more than now",
    2: "Somewhat more than now",
    3: "The same as now",
    4: "Somewhat less than now",
    5: "Much less than now",
    999: "Item skipped"
}

# Political Leaning
political_leaning = {
    1: "Liberal",
    2: "Labor",
    3: "National Party",
    4: "Greens",
    **{i: "Other party" for i in range(5, 98)},
    999: "Item skipped"
}

# Trust in Government
trust_gov = {
    1: "Usually look after themselves",
    2: "Sometimes look after themselves",
    3: "Sometimes can be trusted to do the right thing",
    4: "Usually can be trusted to do the right thing",
    999: "Item skipped"
}

# Issues and Engagement Dictionaries
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

# Age mapping for ABS categories
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

# Income mapping for ABS categories
abs_to_aes_income = {
    'Negative income': 1,
    'Nil income': 1, 
    '$1-$149 ($1-$7,799)': 1, 
    '$150-$299 ($7,800-$15,599)': 1,
    '$300-$399 ($15,600-$20,799)': 3,
    '$400-$499 ($20,800-$25,999)': 3,
    '$500-$649 ($26,000-$33,799)': 5,
    '$650-$799 ($33,800-$41,599)': 7,
    '$800-$999 ($41,600-$51,999)': 9,
    '$1,000-$1,249 ($52,000-$64,999)': 9,
    '$1,250-$1,499 ($65,000-$77,999)': 11,
    '$1,500-$1,749 ($78,000-$90,999)': 13,
    '$1,750-$1,999 ($91,000-$103,999)': 13,
    '$2,000-$2,999 ($104,000-$155,999)': (15, 17),
    '$3,000-$3,499 ($156,000-$181,999)': 19,
    '$3,500 or more ($182,000 or more)': (21, 25),
    'Not stated': 999,
    'Not applicable': 999
}

# Sex mapping for ABS categories
abs_to_aes_sex = {
    "Male": 1,
    "Female": 2,
}

# Marital status mapping for ABS categories
abs_to_aes_marital = {
    "Never married": 1,
    "Married": 2,
    "Separated": 4,
    "Widowed": 3,
    "Divorced": 4
}

def get_mapping_dict(variable_name):
    """Get the appropriate mapping dictionary for a given variable name."""
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