# my_healthpal_project/schemas/extraction_schema.py

from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedDataModel(BaseModel):
    """
    Schema for the data extracted from a clinical conversation transcript.
    """

    seed_id: str = Field(..., description="Matches the relevant seed/transcript ID.")

    primary_complaint: Optional[str] = Field(
        None,
        description="The main reason for the patient's visit, if present."
    )
    diagnoses: List[str] = Field(
        default_factory=list,
        description="List of diagnoses (new or existing) explicitly mentioned."
    )

    medications_discussed: List[str] = Field(
        default_factory=list,
        description="Medications explicitly named in the conversation."
    )
    medication_instructions: List[str] = Field(
        default_factory=list,
        description="Details on dosing, frequency, or side effects, if stated."
    )

    tests_discussed: List[str] = Field(
        default_factory=list,
        description="Any tests or imaging recommended or discussed."
    )
    follow_up_instructions: List[str] = Field(
        default_factory=list,
        description="Explicit guidance on what happens next (appointments, lifestyle, etc.)."
    )

    caregiver_involvement: Optional[str] = Field(
        None,
        description="If a caregiver or family member is involved in the conversation."
    )
    ongoing_therapies_discussed: Optional[str] = Field(
        None,
        description="Mentions or details of existing therapies if the patient is continuing them."
    )
