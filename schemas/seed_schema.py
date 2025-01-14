# my_healthpal_project/schemas/seed_schema.py

from pydantic import BaseModel, Field, conint
from typing import List, Optional

class SeedScenarioModel(BaseModel):
    """
    The portion the LLM will generate (no seed_id).
    We store scenario info for a US-based clinical setting.
    """
    reasoning: str = Field(..., description="Explanation of why this scenario is unique.")
    patient_age: conint(ge=1, le=120) = Field(..., description="Patient's age in years.")
    patient_gender: str = Field(..., description="Patient's gender identity.")
    patient_background: str = Field(..., description="US-based socio-cultural background.")

    doctor_specialty: str = Field(..., description="Medical specialty of the doctor.")
    consultation_location: str = Field(..., description="Where or how the consultation is happening.")

    conversation_length_minutes: conint(ge=3, le=15) = Field(..., description="Approx conversation length.")

    main_complaints: List[str] = Field(..., description="Patient's main complaints.")
    possible_medications: List[str] = Field(..., description="Medications possibly discussed.")
    possible_tests: List[str] = Field(..., description="Tests or imaging that might be recommended.")
    next_steps_suggestions: List[str] = Field(..., description="Likely follow-up or recommended actions.")

    conversation_quirks: Optional[str] = Field(
        None,
        description="Extra nuance, e.g. caretaker presence, fear of needles, etc."
    )
    ongoing_therapies: Optional[str] = Field(
        None,
        description="Existing therapies the patient is undergoing, if relevant."
    )

class SeedModel(BaseModel):
    """
    The final seed we store locally. Includes our internal seed_id,
    plus the scenario fields from SeedScenarioModel.
    """
    seed_id: str = Field(..., description="Unique ID we assign (e.g., 'seed-0003').")
    reasoning: str
    patient_age: int
    patient_gender: str
    patient_background: str
    doctor_specialty: str
    consultation_location: str
    conversation_length_minutes: int
    main_complaints: List[str]
    possible_medications: List[str]
    possible_tests: List[str]
    next_steps_suggestions: List[str]
    conversation_quirks: Optional[str] = None
    ongoing_therapies: Optional[str] = None
