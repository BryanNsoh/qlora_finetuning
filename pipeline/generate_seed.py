# my_healthpal_project/pipeline/generate_seed.py

import json
from typing import List, Optional
import asyncio

from schemas.seed_schema import SeedModel
from llm_api.LLM_API_handler import UnifiedLLMHandler

# --------------- PROMPT BUILDING ---------------

def build_seed_prompt(previous_seeds_text: str) -> str:
    """
    Builds a prompt for generating exactly ONE new Seed JSON object.
    The LLM must ensure uniqueness compared to the previously generated seeds.
    """
    return f"""
<SEED_GENERATION_REQUEST>
  <CONTEXT>
    {previous_seeds_text}
  </CONTEXT>
  <INSTRUCTIONS>
    You are generating exactly ONE new seed in valid JSON that fits SeedModel:
    {{
      "seed_id": "...",
      "reasoning": "...",
      "patient_age": ...,
      "patient_gender": "...",
      "patient_background": "...",
      "doctor_specialty": "...",
      "consultation_location": "...",
      "conversation_length_minutes": ...,
      "main_complaints": [...],
      "possible_medications": [...],
      "possible_tests": [...],
      "next_steps_suggestions": [...],
      "conversation_quirks": "... or null",
      "ongoing_therapies": "... or null"
    }}

    Requirements:
    - "seed_id": a unique string (like 'seed-xxx-{{random}}')
    - "reasoning": a brief explanation of how this scenario is unique compared to <CONTEXT>.
    - Keep conversation_length_minutes between 3 and 15.
    - Reflect a plausible US-based clinical scenario.
    - If you mention "ongoing_therapies", ensure it logically fits the scenario.
    - Carefully avoid duplicating seeds from <CONTEXT>.

    Output must be strictly one JSON object with no extra text.
  </INSTRUCTIONS>
</SEED_GENERATION_REQUEST>
"""

# --------------- LLM CALL ---------------

async def create_single_seed(previous_seeds_text: str, model_name: str) -> dict:
    """
    Calls the LLM to generate one new seed. Returns a validated dictionary.
    """
    prompt = build_seed_prompt(previous_seeds_text)
    # We fix the concurrency limit: up to 600 requests/minute => 10 requests/sec
    handler = UnifiedLLMHandler(requests_per_minute=600)
    response = await handler.process(
        prompts=prompt,
        model=model_name,          # e.g. "deepseek:deepseek-chat"
        response_type=None         # We'll parse JSON ourselves
    )
    print("DEBUG LLM response:", response.model_dump(mode="json"))
    

    if not response.success or not response.data:
        raise ValueError(f"LLM call failed: {response.error or 'No data returned'}")

    raw_text = response.data.content
    try:
        seed_dict = json.loads(raw_text)
        # Validate the seed using pydantic
        seed_obj = SeedModel(**seed_dict)
        return seed_obj.dict()
    except Exception as e:
        raise ValueError(f"Parsing or validation error for seed: {e}")

# --------------- BATCH GENERATION ---------------

async def generate_seeds(
    num_seeds: int = 1,
    model_name: str = "deepseek:deepseek-chat",
    existing_seeds: Optional[List[dict]] = None
) -> List[dict]:
    """
    Generates multiple seeds asynchronously, returning a list of validated seed dicts.
    Appends them to an existing list if provided.
    """
    if existing_seeds is None:
        existing_seeds = []

    # Convert existing seeds to a text representation for context:
    previous_seeds_text = json.dumps(existing_seeds, ensure_ascii=False, indent=2)

    tasks = []
    for _ in range(num_seeds):
        tasks.append(create_single_seed(previous_seeds_text, model_name))

    # Run tasks with concurrency
    new_seeds = await asyncio.gather(*tasks)
    return new_seeds
