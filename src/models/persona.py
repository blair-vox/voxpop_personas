"""
Models for persona data and responses.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class SurveyResponse(BaseModel):
    """Model for structured survey responses."""
    support_level: int  # 1-5 scale
    impact_on_housing: int  # 1-5 scale
    impact_on_transport: int  # 1-5 scale
    impact_on_community: int  # 1-5 scale
    key_concerns: List[str]
    suggested_improvements: List[str]

class PersonaResponse(BaseModel):
    """Model for complete persona response including both narrative and survey data."""
    persona_details: Dict[str, Any]  # Changed from Dict[str, str] to Dict[str, Any] to handle lists
    narrative_response: str
    survey_response: SurveyResponse
    timestamp: str
    sentiment_score: Optional[float] = None
    key_themes: Optional[List[str]] = None
    canonical_themes: Optional[List[Dict[str, str]]] = None

class Persona(BaseModel):
    """Pydantic model for persona data."""
    name: str
    age: str
    gender: str
    location: str
    income: str
    tenure: str
    job_tenure: str
    occupation: str
    education: str
    transport: str
    marital_status: str
    partner_activity: str
    household_size: str
    family_payments: str
    child_care_benefit: str
    investment_properties: str
    transport_infrastructure: str
    political_leaning: str
    trust: str
    issues: List[str]
    engagement: str 