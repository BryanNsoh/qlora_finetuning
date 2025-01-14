# my_healthpal_project/pipeline/generate_seed.py

import json
import asyncio
from typing import List, Optional

from llm_api.LLM_API_handler import UnifiedLLMHandler
from schemas.seed_schema import SeedScenarioModel, SeedModel

def build_seed_prompt(existing_scenarios_text: str) -> str:
    """
    Builds a prompt for generating exactly one new scenario (SeedScenarioModel),
    WITHOUT seed_id. Encourages realism and diversity.
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
    Generate exactly ONE new scenario in JSON that fits SeedScenarioModel.
    (Do NOT include 'seed_id'.)
    
    Begin with a reflection on what scenarios, demographics or contexts are already present in the current set. 
    Then delibrerate on how to generate a new seed that is different from the existing ones while remaining realistic for a US-based clinical setting.
    
    Requirements:
      - Must be a plausible, US-based clinical scenario.
      - conversation_length_minutes: an integer 3–15.
      - If "ongoing_therapies" is relevant, include it.
      - Must avoid duplicating existing scenarios. 
      - Strive for diversity: vary patient backgrounds, complaints, or contexts not yet used.
      - Add creative touches (e.g. caretaker presence, or unique socio-cultural nuances)
        while remaining grounded and realistic.
      - The critical constraint is that the current seed be as different from all previous seeds as possible while still being palusible and realistic
    
    Output must be STRICTLY one JSON object with the fields from SeedScenarioModel 
    (no 'seed_id', no extra keys).
  </INSTRUCTIONS>
</SEED_GENERATION_REQUEST>
"""

async def create_single_scenario(
    existing_scenarios_text: str,
    model_name: str
) -> SeedScenarioModel:
    """
    Calls the LLM to generate one new scenario (SeedScenarioModel).
    The LLM does NOT create seed_id; we handle that ourselves.
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

    return response.data  # A SeedScenarioModel instance

async def generate_seeds(
    num_seeds: int,
    model_name: str,
    existing_seeds: List[SeedModel]
) -> List[SeedModel]:
    """
    Generates N new seeds. The LLM returns only scenario data; we assign seed_id ourselves.
    """
    # Convert existing seeds to scenario dicts
    existing_list_of_scenarios = []
    for s in existing_seeds:
        scenario_dict = {
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
        existing_list_of_scenarios.append(scenario_dict)

    existing_scenarios_text = json.dumps(existing_list_of_scenarios, indent=2)

    # Find the highest seed_id so we can increment
    latest_seed_id = None
    for seed_obj in existing_seeds:
        if latest_seed_id is None or seed_obj.seed_id > latest_seed_id:
            latest_seed_id = seed_obj.seed_id

    # Async-generate new scenarios
    tasks = []
    for _ in range(num_seeds):
        tasks.append(create_single_scenario(existing_scenarios_text, model_name))
    new_scenarios = await asyncio.gather(*tasks)

    # Convert them to final seeds with newly assigned ID
    def next_seed_id(current_id: Optional[str]) -> str:
        if not current_id:
            return "seed-0001"
        prefix, num_str = current_id.rsplit("-", 1)
        new_num = int(num_str) + 1
        return f"{prefix}-{new_num:04d}"

    new_seeds_list = []
    for scenario in new_scenarios:
        latest_seed_id = next_seed_id(latest_seed_id)
        final_seed = SeedModel(
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
        new_seeds_list.append(final_seed)

    return new_seeds_list
