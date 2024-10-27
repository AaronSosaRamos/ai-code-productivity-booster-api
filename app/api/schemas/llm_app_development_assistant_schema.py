from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ApplicationIdea(BaseModel):
    project_name: str
    description: str

class DevelopmentOutput(BaseModel):
    feasibility: Dict[str, str]
    design_architecture: Dict[str, str]
    recommended_tools: List[str]
    implementation_plan: Dict[str, str]
    estimated_timeline: Optional[str]
    estimated_cost: Optional[float]
    potential_challenges: Optional[List[str]]
    additional_notes: Optional[str]
    resources: Optional[List[str]]
    contact_information: Optional[str]
    legal_disclaimer: Optional[str]