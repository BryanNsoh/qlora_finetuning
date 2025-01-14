# my_healthpal_project/pipeline/generate_seed.py

import json
import asyncio
from typing import List, Optional

from pydantic import BaseModel
from llm_api.LLM_API_handler import UnifiedLLMHandler
from schemas.seed_schema import SeedScenarioModel, SeedModel

def build_seed_prompt(existing_scenarios_text: str) -> str:
    """
    Builds a prompt for generating exactly one new scenario object (SeedScenarioModel).
    We embed model_json_schema so the LLM knows the structure.
    """
    schema_str = json.dumps(SeedScenarioModel.model_json_schema(), indent=2)

    return f"""
<SEED_GENERATION_REQUEST>
  <EXISTING_SCENARIOS>
    {existing_scenarios_text}
  </EXISTING_SCENARIOS>
  <SCHEMA>
{schema_str}
  </SCHEMA>
  <INSTRUCTIONS>
    Generate exactly one new scenario in JSON that fits SeedScenarioModel. 
    Do NOT include 'seed_id'.
    Requirements:
    - Must be a US-based scenario. 
    - Avoid duplicating existing_scenarios.
    - conversation_length_minutes between 3 and 15.
    - If 'ongoing_therapies' is relevant, include it.

    Output must be strictly one JSON object with the fields from SeedScenarioModel.
  </INSTRUCTIONS>
</SEED_GENERATION_REQUEST>
"""

def generate_new_seed_id(latest_id: Optional[str]) -> str:
    """
    Generate a new 'seed_id' based on the highest ID seen so far.
    E.g. if the last known is 'seed-0012', we return 'seed-0013'.
    If None, we start with 'seed-0001'.
    """
    if not latest_id:
        return "seed-0001"
    # parse the trailing integer
    prefix, num_str = latest_id.rsplit("-", 1)
    new_num = int(num_str) + 1
    return f"{prefix}-{new_num:04d}"

async def create_single_scenario(
    existing_scenarios_text: str,
    model_name: str
) -> SeedScenarioModel:
    """
    Calls the LLM to generate one new scenario (SeedScenarioModel).
    """
    prompt = build_seed_prompt(existing_scenarios_text)
    handler = UnifiedLLMHandler(requests_per_minute=600)
    response = await handler.process(
        prompts=prompt,
        model=model_name,
        response_type=SeedScenarioModel
    )

    if not response.success or not response.data:
        raise ValueError(f"LLM call failed: {response.error or 'No data returned'}")
    return response.data  # This is a SeedScenarioModel instance

async def generate_seeds(
    num_seeds: int,
    model_name: str,
    existing_seeds: List[SeedModel]
) -> List[SeedModel]:
    """
    Generates N new seeds. We do this by generating N scenarios from the LLM, 
    each appended with a new seed_id.
    """
    # Convert existing seeds to text to avoid duplication
    existing_list_of_scenarios = []
    for s in existing_seeds:
        # We only show scenario fields, ignoring 'seed_id'
        scenario = {
            "reasoning": s.reasoning,
            "patient_age": s.patient_age,
            "patient_gender": s.patient_gender,
            "patient_background": s.patient_background,
            "doctor_specialty": s.doctor_specialty,
            "consultation_location": s.consultation_location,
            "conversation_length_minutes": s.conversation_length_minutes,
            "main_complaints": s.main_complaints,
            "possible_medications": s.possible_medications,
            "possible_tests": s.possible_tests,
            "next_steps_suggestions": s.next_steps_suggestions,
            "conversation_quirks": s.conversation_quirks,
            "ongoing_therapies": s.ongoing_therapies,
        }
        existing_list_of_scenarios.append(scenario)

    existing_scenarios_text = json.dumps(existing_list_of_scenarios, indent=2)

    # Figure out the highest seed ID so we can increment
    latest_seed_id = None
    for seed_obj in existing_seeds:
        # e.g. "seed-0012" -> 12
        # we want the max
        if latest_seed_id is None or seed_obj.seed_id > latest_seed_id:
            latest_seed_id = seed_obj.seed_id

    tasks = []
    for _ in range(num_seeds):
        tasks.append(create_single_scenario(existing_scenarios_text, model_name))

    new_scenarios = await asyncio.gather(*tasks)

    # Convert them to final seeds with new ID
    new_seeds: List[SeedModel] = []
    for scenario in new_scenarios:
        latest_seed_id = generate_new_seed_id(latest_seed_id)
        seed_obj = SeedModel(
            seed_id=latest_seed_id,
            reasoning=scenario.reasoning,
            patient_age=scenario.patient_age,
            patient_gender=scenario.patient_gender,
            patient_background=scenario.patient_background,
            doctor_specialty=scenario.doctor_specialty,
            consultation_location=scenario.consultation_location,
            conversation_length_minutes=scenario.conversation_length_minutes,
            main_complaints=scenario.main_complaints,
            possible_medications=scenario.possible_medications,
            possible_tests=scenario.possible_tests,
            next_steps_suggestions=scenario.next_steps_suggestions,
            conversation_quirks=scenario.conversation_quirks,
            ongoing_therapies=scenario.ongoing_therapies
        )
        new_seeds.append(seed_obj)

    return new_seeds
