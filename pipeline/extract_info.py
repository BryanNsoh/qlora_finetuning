# my_healthpal_project/pipeline/extract_info.py

import json
import asyncio
from typing import List

from llm_api.LLM_API_handler import UnifiedLLMHandler
from pydantic import BaseModel, Field
from schemas.extraction_schema import ExtractedDataModel

class ExtractionInput(BaseModel):
    seed_id: str
    transcript: str

def build_extraction_prompt(input_obj: ExtractionInput) -> str:
    """
    Build a prompt instructing the LLM to parse the transcript
    into the ExtractedDataModel. We embed the JSON schema.
    """
    schema_str = json.dumps(ExtractedDataModel.model_json_schema(), indent=2)
    input_json = json.dumps(input_obj.model_dump(), indent=2)

    return f"""
<EXTRACTION_REQUEST>
  <INPUT>
{input_json}
  </INPUT>
  <SCHEMA>
{schema_str}
  </SCHEMA>
  <INSTRUCTIONS>
    Output exactly one JSON object matching ExtractedDataModel, where:
      "seed_id" = "{input_obj.seed_id}"
    Parse the transcript text to fill out the other fields:
      primary_complaint, diagnoses, medications_discussed, medication_instructions,
      tests_discussed, follow_up_instructions, caregiver_involvement,
      ongoing_therapies_discussed.
    No extra keys.
  </INSTRUCTIONS>
</EXTRACTION_REQUEST>
"""

async def extract_single(
    seed_id: str,
    transcript: str,
    model_name: str
) -> ExtractedDataModel:
    """
    Return an ExtractedDataModel parsed from the transcript.
    """
    input_obj = ExtractionInput(seed_id=seed_id, transcript=transcript)
    prompt = build_extraction_prompt(input_obj)

    handler = UnifiedLLMHandler(requests_per_minute=600)
    response = await handler.process(
        prompts=prompt,
        model=model_name,
        response_type=ExtractedDataModel
    )

    if not response.success or not response.data:
        raise ValueError(f"Extraction LLM call failed: {response.error or 'No data returned'}")
    extracted = response.data

    # Check the seed_id
    if extracted.seed_id != seed_id:
        raise ValueError(f"Expected seed_id={seed_id}, got {extracted.seed_id} from LLM")

    return extracted

async def extract_info_for_new_transcripts(
    transcripts_data: List[dict],
    model_name: str
) -> List[ExtractedDataModel]:
    """
    transcripts_data: a list of dicts like {'seed_id': 'seed-0001', 'transcript': '...'}
    Returns: list of ExtractedDataModel for those that lack extraction.
    """
    tasks = []
    for tdata in transcripts_data:
        sid = tdata["seed_id"]
        txt = tdata["transcript"]
        tasks.append(extract_single(sid, txt, model_name))

    results = await asyncio.gather(*tasks)
    return list(results)
