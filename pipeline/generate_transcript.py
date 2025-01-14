# my_healthpal_project/pipeline/generate_transcript.py

import json
import asyncio
from typing import List

from pydantic import BaseModel, Field
from llm_api.LLM_API_handler import UnifiedLLMHandler
from schemas.seed_schema import SeedModel

class TranscriptContentModel(BaseModel):
    """
    The LLM will return a single JSON object with:
      { "content": "the entire conversation" }
    No mention of seed_id in the LLM response.
    """
    content: str = Field(..., description="The entire conversation text.")

def build_transcript_prompt(seed_obj: SeedModel) -> str:
    """
    We embed 'TranscriptContentModel' schema but
    the LLM only needs to produce the 'content' field in a JSON.
    We do NOT pass seed_id; we handle that ourselves in code.
    """
    schema_str = json.dumps(TranscriptContentModel.model_json_schema(), indent=2)
    seed_str = json.dumps(seed_obj.model_dump(), indent=2, ensure_ascii=False)

    return f"""
<TRANSCRIPT_GENERATION_REQUEST>
  <SEED_SCENARIO>
{seed_str}
  </SEED_SCENARIO>
  <SCHEMA>
{schema_str}
  </SCHEMA>
  <INSTRUCTIONS>
    Return exactly one JSON object:
      {{
        "content": "...the entire conversation as plain text..."
      }}
    No other fields.

    The conversation must reflect the scenario details:
      - doctor_specialty, consultation_location, main_complaints, etc.
      - If "ongoing_therapies" is not null, mention it naturally.
      - Incorporate "next_steps_suggestions" as doctor's instructions.
      - If "conversation_quirks" is present, incorporate it (e.g. caretaker, fear of needles).

    Aim for a natural multi-turn dialogue (e.g. 8-20 turns).
    Label speakers as "PATIENT:" or "DOCTOR:" consistently, 
    plus extra participants if relevant.
    Keep it realistic and consistent with the scenario.
  </INSTRUCTIONS>
</TRANSCRIPT_GENERATION_REQUEST>
"""

async def create_transcript_for_seed(
    seed_obj: SeedModel,
    model_name: str
) -> str:
    """
    Generate a conversation transcript for a single seed.
    We store/return only the raw text from 'content'.
    """
    prompt = build_transcript_prompt(seed_obj)
    handler = UnifiedLLMHandler(requests_per_minute=600)
    response = await handler.process(
        prompts=prompt,
        model=model_name,
        response_type=TranscriptContentModel
    )
    if not response.success or not response.data:
        raise ValueError(f"LLM call failed: {response.error or 'No data returned'}")

    transcript_model = response.data  # an instance of TranscriptContentModel
    return transcript_model.content

async def generate_transcripts_for_new_seeds(
    seeds_missing_transcripts: List[SeedModel],
    model_name: str
) -> List[dict]:
    """
    Generate transcripts for seeds that do NOT have transcripts yet.
    Return a list of dicts: [{ "seed_id": ..., "transcript": ... }, ...]
    so we can easily append them to transcripts.jsonl.
    """
    tasks = []
    for s in seeds_missing_transcripts:
        tasks.append(create_transcript_for_seed(s, model_name))
    raw_texts = await asyncio.gather(*tasks)

    # Combine with the seeds
    results = []
    for seed, txt in zip(seeds_missing_transcripts, raw_texts):
        results.append({
            "seed_id": seed.seed_id,
            "transcript": txt
        })
    return results
