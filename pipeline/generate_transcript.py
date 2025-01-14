# my_healthpal_project/pipeline/generate_transcript.py

import json
import asyncio
from typing import List

from pydantic import BaseModel, Field
from llm_api.LLM_API_handler import UnifiedLLMHandler
from schemas.seed_schema import SeedModel

class TranscriptModel(BaseModel):
    seed_id: str = Field(..., description="Which seed this transcript belongs to.")
    content: str = Field(..., description="The entire conversation text.")

def build_transcript_prompt(seed_obj: SeedModel) -> str:
    """
    Show the schema for TranscriptModel but note that the LLM only needs 
    to produce the 'content'; we will enforce seed_id ourselves.
    """
    schema_str = json.dumps(TranscriptModel.model_json_schema(), indent=2)
    seed_str = json.dumps(seed_obj.model_dump(), indent=2, ensure_ascii=False)

    return f"""
<TRANSCRIPT_GENERATION_REQUEST>
  <SEED>
{seed_str}
  </SEED>
  <SCHEMA>
{schema_str}
  </SCHEMA>
  <INSTRUCTIONS>
    We already have the seed_id = "{seed_obj.seed_id}", so do NOT generate an ID.
    Return a JSON object with these two fields:
      "seed_id": "{seed_obj.seed_id}"
      "content": "the entire conversation as plain text"

    The conversation:
    - Reflect the scenario in the seed: doctor_specialty, consultation_location, main_complaints, etc.
    - ~8-20 total turns, plain text labeled "PATIENT:" / "DOCTOR:".
    - If "ongoing_therapies" is not null, mention it logically.
    - Incorporate next_steps_suggestions as doctor's instructions.

    Output strictly JSON with exactly seed_id and content (no additional keys).
  </INSTRUCTIONS>
</TRANSCRIPT_GENERATION_REQUEST>
"""

async def create_transcript_for_seed(
    seed_obj: SeedModel,
    model_name: str
) -> TranscriptModel:
    """
    Generate a transcript (TranscriptModel) for one seed.
    """
    prompt = build_transcript_prompt(seed_obj)
    handler = UnifiedLLMHandler(requests_per_minute=600)
    response = await handler.process(
        prompts=prompt,
        model=model_name,
        response_type=TranscriptModel
    )

    if not response.success or not response.data:
        raise ValueError(f"LLM call failed: {response.error or 'No data returned'}")

    # Must match the seed_id we forced
    transcript = response.data
    if transcript.seed_id != seed_obj.seed_id:
        raise ValueError(f"LLM returned unexpected seed_id: {transcript.seed_id}")
    return transcript

async def generate_transcripts_for_new_seeds(
    seeds_missing_transcripts: List[SeedModel],
    model_name: str
) -> List[TranscriptModel]:
    """
    Generate transcripts for seeds that do NOT have transcripts yet.
    """
    tasks = []
    for s in seeds_missing_transcripts:
        tasks.append(create_transcript_for_seed(s, model_name))
    results = await asyncio.gather(*tasks)
    return list(results)
