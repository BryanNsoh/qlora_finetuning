# my_healthpal_project/schemas/extraction_schema.py

from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedDataModel(BaseModel):
    seed_id: str = Field(..., description="Links to the relevant seed/transcript.")

    primary_complaint: Optional[str] = Field(None)
    diagnoses: List[str] = Field(default_factory=list)
    medications_discussed: List[str] = Field(default_factory=list)
    medication_instructions: List[str] = Field(default_factory=list)
    tests_discussed: List[str] = Field(default_factory=list)
    follow_up_instructions: List[str] = Field(default_factory=list)
    caregiver_involvement: Optional[str] = None
    ongoing_therapies_discussed: Optional[str] = None
