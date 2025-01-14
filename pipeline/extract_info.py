# my_healthpal_project/pipeline/extract_info.py

import json
import asyncio
from typing import List, Tuple
from schemas.extraction_schema import ExtractedDataModel
from llm_api.LLM_API_handler import UnifiedLLMHandler

# --------------- PROMPT BUILDING ---------------

def build_extraction_prompt(transcript_text: str, seed_id: str) -> str:
    """
    Creates an extraction prompt that instructs the LLM to parse the transcript
    according to ExtractedDataModel fields.
    """
    return f"""
<EXTRACTION_REQUEST>
  <TRANSCRIPT_TEXT>
    {transcript_text}
  </TRANSCRIPT_TEXT>
  <INSTRUCTIONS>
    Return exactly 1 valid JSON object matching:
    {{
      "seed_id": "{seed_id}",
      "primary_complaint": "... or null",
      "diagnoses": [...],
      "medications_discussed": [...],
      "medication_instructions": [...],
      "tests_discussed": [...],
      "follow_up_instructions": [...],
      "caregiver_involvement": "... or null",
      "ongoing_therapies_discussed": "... or null"
    }}

    Guidance:
    - Identify the main complaint from the transcript.
    - If the doctor explicitly diagnoses or references conditions, list them in "diagnoses".
    - Collect any medication names, instructions, or test recommendations.
    - "caregiver_involvement" if a caretaker is present or mentioned.
    - "ongoing_therapies_discussed" if the conversation addresses continuing therapy.
    - Must be valid JSON with no extra keys or text. 
  </INSTRUCTIONS>
</EXTRACTION_REQUEST>
"""

# --------------- LLM CALL ---------------

async def extract_single_transcript(
    seed_id: str,
    transcript_text: str,
    model_name: str = "deepseek:deepseek-chat"
) -> dict:
    """
    Calls the LLM to extract structured info from a single transcript.
    Returns a validated dictionary according to ExtractedDataModel.
    """
    prompt = build_extraction_prompt(transcript_text, seed_id)
    handler = UnifiedLLMHandler(requests_per_minute=600)
    response = await handler.process(
        prompts=prompt,
        model=model_name,
        response_type=None
    )

    if not response.success or not response.data:
        raise ValueError(f"LLM extraction call failed: {response.error or 'No data returned'}")

    raw_text = response.data.content
    try:
        extracted_dict = json.loads(raw_text)
        # Validate with ExtractedDataModel
        validated = ExtractedDataModel(**extracted_dict)
        return validated.dict()
    except Exception as e:
        raise ValueError(f"Parsing or validation error for extracted info: {e}")

# --------------- BATCH EXTRACTION ---------------

async def extract_info_from_transcripts(
    seed_ids: List[str],
    transcripts: List[str],
    model_name: str = "deepseek:deepseek-chat"
) -> List[dict]:
    """
    Extracts structured info from multiple transcripts concurrently.
    Must match order of seed_ids and transcripts.
    """
    if len(seed_ids) != len(transcripts):
        raise ValueError("Mismatched lengths for seed_ids and transcripts.")

    tasks = []
    for sid, txt in zip(seed_ids, transcripts):
        tasks.append(extract_single_transcript(sid, txt, model_name))

    extracted_list = await asyncio.gather(*tasks)
    return extracted_list
