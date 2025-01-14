# my_healthpal_project/schemas/seed_schema.py

from pydantic import BaseModel, Field, conint
from typing import List, Optional

class SeedModel(BaseModel):
    """
    Defines the schema for a 'seed' object describing a unique clinical scenario.
    """

    seed_id: str = Field(..., description="Unique ID for this seed.")
    reasoning: str = Field(..., description="Free-text rationale for uniqueness and coverage.")

    patient_age: conint(ge=1, le=120) = Field(..., description="Patient's age in years.")
    patient_gender: str = Field(..., description="Patient's gender identity.")
    patient_background: str = Field(..., description="US-based socio-cultural or personal background.")

    doctor_specialty: str = Field(..., description="Medical specialty of the doctor.")
    consultation_location: str = Field(..., description="Location or setting of the consultation.")

    conversation_length_minutes: conint(ge=3, le=15) = Field(..., description="Approx. conversation length (3–15).")

    main_complaints: List[str] = Field(..., description="Patient's main complaints or issues.")
    possible_medications: List[str] = Field(..., description="Medications possibly mentioned.")
    possible_tests: List[str] = Field(..., description="Tests or imaging that might be recommended.")
    next_steps_suggestions: List[str] = Field(..., description="Likely follow-up or recommended actions.")

    conversation_quirks: Optional[str] = Field(
        None, 
        description="Optional nuance, e.g. caretaker presence or an unusual fear."
    )

    ongoing_therapies: Optional[str] = Field(
        None,
        description="Details of existing therapies (e.g. PT, dialysis) if relevant."
    )
