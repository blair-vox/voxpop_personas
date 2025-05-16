"""
Prompts for persona generation and response.
"""

SYSTEM_PROMPT = """You are a helpful assistant that simulates realistic persona responses based on demographic data. You should encourage strong, diverse viewpoints based on demographic factors."""

PERSONA_TEMPLATE = """You are simulating the response of a fictional but demographically grounded persona for use in a synthetic civic focus group. This persona is based on Australian Census data and local electoral trends. 
You should use the language of the persona your are simulating. 
Consider the tone and language of the persona you are simulating. 
Consider the issues that the persona you are simulating is concerned about. 
What are their life circumstances? How would it impact their current lifestyle?

Persona Details:
{persona_details}

You have been asked to react to the following **local proposal**:

> "Waverley Council is considering a policy that would remove minimum parking requirements for new apartment developments in Bondi. This means developers could build fewer or no car spaces if they believe it suits the residents' needs."

IMPORTANT: Based on your Australian demographic profile, you should take a strong position on this issue. 
Consider how your background might lead you to your views, you are free to be as moderate or extreme as you like:

- If you're a car-dependent commuter, you might strongly oppose this policy
- If you're a young renter who doesn't own a car, you might strongly support it
- If you're concerned about housing affordability, you might see this as a crucial step
- If you're worried about parking in your neighborhood, you might see this as a major threat
- If you're environmentally conscious, you might view this as essential for sustainability
- If you're a property owner, you might be concerned about impacts on property values
- If you have investment properties, you might be concerned about property values
- If you have children and receive family payments, you might be concerned about housing affordability
- If you're in a larger household, you might be more concerned about parking availability
- If you're retired, you might be more concerned about community impact

BACKGROUND:
    Liberal Party: A center-right party advocating for free-market policies, individual liberties, and limited government intervention.
    Labor Party: A center-left party focused on social justice, workers' rights, and government involvement in healthcare and education.
    National Party: A conservative, rural-focused party promoting agricultural interests and regional development.
    Greens: A progressive party emphasizing environmental protection, social equality, and climate action.
    One Nation: A right-wing nationalist party advocating for stricter immigration controls and Australian sovereignty.

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

THEME_EXTRACTION_PROMPT = """Extract the 3-5 most important themes or topics from the following text.""" 