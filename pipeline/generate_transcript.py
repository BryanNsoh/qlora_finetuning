# my_healthpal_project/pipeline/generate_transcript.py

import json
import asyncio
from typing import List, Dict
from llm_api.LLM_API_handler import UnifiedLLMHandler

# --------------- PROMPT BUILDING ---------------

def build_transcript_prompt(seed_json_str: str) -> str:
    """
    Given a seed in JSON string form, generate a plausible US-based clinical conversation.
    """
    return f"""
<TRANSCRIPT_GENERATION_REQUEST>
  <SEED_INFO>
    {seed_json_str}
  </SEED_INFO>
  <INSTRUCTIONS>
    Create a fictional US-based clinical conversation between a patient and a doctor.
    - Reflect 'doctor_specialty' and 'consultation_location' from the seed.
    - Address the 'main_complaints' realistically.
    - If 'ongoing_therapies' is not null, incorporate it if relevant.
    - Mention or discuss possible medications/tests only if it fits the natural flow.
    - 'next_steps_suggestions' can appear as instructions from the doctor.
    - Keep the entire conversation in plain text, speaker-labeled: "PATIENT:", "DOCTOR:", etc.
    - Aim for 8-20 total turns, not more than ~50 lines total.
    - Integrate 'conversation_quirks' if present (e.g. caretaker, fear of meds).
    - Must remain consistent with the seed. Avoid nonsense or contradictions.
  </INSTRUCTIONS>
</TRANSCRIPT_GENERATION_REQUEST>
"""

# --------------- LLM CALL ---------------

async def create_transcript_from_seed(
    seed: Dict,
    model_name: str = "deepseek:deepseek-chat"
) -> str:
    """
    Calls the LLM to generate a conversation transcript for a single seed.
    Returns the transcript text.
    """
    prompt = build_transcript_prompt(json.dumps(seed, ensure_ascii=False))
    handler = UnifiedLLMHandler(requests_per_minute=600)
    response = await handler.process(
        prompts=prompt,
        model=model_name,
        response_type=None  # We'll parse the raw text
    )

    if not response.success or not response.data:
        raise ValueError(f"LLM call failed: {response.error or 'No data returned'}")

    return response.data.content.strip()

# --------------- BATCH GENERATION ---------------

async def generate_transcripts(
    seeds: List[dict],
    model_name: str = "deepseek:deepseek-chat"
) -> List[str]:
    """
    Generates transcripts for all seeds concurrently (up to concurrency limit).
    Returns list of transcripts, preserving order with seeds.
    """
    tasks = []
    for seed in seeds:
        tasks.append(create_transcript_from_seed(seed, model_name))

    transcripts = await asyncio.gather(*tasks)
    return transcripts
