# my_healthpal_project/main.py

import asyncio
import json
from pathlib import Path
from typing import List, Dict

from schemas.seed_schema import SeedModel
from pipeline.generate_seed import generate_seeds
from pipeline.generate_transcript import generate_transcripts_for_new_seeds, TranscriptModel
from pipeline.extract_info import extract_info_for_new_transcripts
from schemas.extraction_schema import ExtractedDataModel

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SEEDS_FILE = DATA_DIR / "seeds.jsonl"
TRANSCRIPTS_FILE = DATA_DIR / "transcripts.jsonl"
EXTRACTED_FILE = DATA_DIR / "extracted.jsonl"

async def orchestrate_pipeline(
    num_new_seeds: int = 10,
    model_name: str = "openai:gpt-4o-mini"
):
    """
    1) Load existing seeds from seeds.jsonl.
    2) Generate num_new_seeds more, append them to seeds.jsonl.
    3) Identify seeds that have no transcript, generate transcripts for them, append to transcripts.jsonl.
    4) Identify transcripts that have no extraction, extract them, append to extracted.jsonl.
    """

    # ---- 1) LOAD EXISTING SEEDS ----
    existing_seeds: List[SeedModel] = []
    if SEEDS_FILE.exists():
        with SEEDS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    seed_obj = SeedModel(**data)
                    existing_seeds.append(seed_obj)

    # ---- 2) GENERATE N NEW SEEDS & SAVE ----
    print(f"Generating {num_new_seeds} new seeds with {model_name}...")
    new_seeds: List[SeedModel] = []
    if num_new_seeds > 0:
        new_seeds = await generate_seeds(num_new_seeds, model_name, existing_seeds)
        # Append them to seeds.jsonl
        with SEEDS_FILE.open("a", encoding="utf-8") as f:
            for s in new_seeds:
                f.write(json.dumps(s.model_dump()) + "\n")

    all_seeds = existing_seeds + new_seeds

    # ---- 3) GENERATE TRANSCRIPTS FOR SEEDS THAT DON'T HAVE ONE ----
    existing_transcripts = []
    if TRANSCRIPTS_FILE.exists():
        with TRANSCRIPTS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_transcripts.append(json.loads(line))
                    # format: {"seed_id": <>, "transcript": <>}

    existing_seed_ids_with_transcripts = {item["seed_id"] for item in existing_transcripts}

    # any seeds missing transcripts?
    seeds_missing_transcripts = []
    for s in all_seeds:
        if s.seed_id not in existing_seed_ids_with_transcripts:
            seeds_missing_transcripts.append(s)

    if seeds_missing_transcripts:
        print(f"Generating transcripts for {len(seeds_missing_transcripts)} seeds without transcripts...")
        new_transcripts_models = await generate_transcripts_for_new_seeds(seeds_missing_transcripts, model_name)
        # Append them to transcripts.jsonl
        with TRANSCRIPTS_FILE.open("a", encoding="utf-8") as f:
            for tm in new_transcripts_models:
                data_to_write = {
                    "seed_id": tm.seed_id,
                    "transcript": tm.content
                }
                f.write(json.dumps(data_to_write) + "\n")
    else:
        print("No seeds are missing transcripts.")

    # ---- 4) EXTRACT INFO FOR TRANSCRIPTS THAT DON'T HAVE EXTRACTION ----
    existing_extracted = []
    if EXTRACTED_FILE.exists():
        with EXTRACTED_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_extracted.append(json.loads(line))
                    # format matches ExtractedDataModel

    existing_seed_ids_with_extraction = {item["seed_id"] for item in existing_extracted}

    # Let's rebuild a map seed_id -> transcript text
    all_transcripts_map = {}
    if TRANSCRIPTS_FILE.exists():
        with TRANSCRIPTS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    sid = rec["seed_id"]
                    txt = rec["transcript"]
                    all_transcripts_map[sid] = txt

    transcripts_for_extraction = []
    for sid, txt in all_transcripts_map.items():
        if sid not in existing_seed_ids_with_extraction:
            # we need extraction for this transcript
            transcripts_for_extraction.append({"seed_id": sid, "transcript": txt})

    if transcripts_for_extraction:
        print(f"Extracting info for {len(transcripts_for_extraction)} transcripts with no extraction...")
        new_extractions = await extract_info_for_new_transcripts(transcripts_for_extraction, model_name)
        # Append them to extracted.jsonl
        with EXTRACTED_FILE.open("a", encoding="utf-8") as f:
            for eobj in new_extractions:
                f.write(json.dumps(eobj.model_dump()) + "\n")
    else:
        print("No transcripts are missing extraction.")

    print("Pipeline complete!\n"
          f"Seeds total: {len(all_seeds)}\n"
          f"Transcripts total: {len(all_transcripts_map)}\n"
          f"Extractions total: {len(existing_extracted) + (len(transcripts_for_extraction) if transcripts_for_extraction else 0)}")

def main():
    # Example usage: generate 2 new seeds each run
    asyncio.run(orchestrate_pipeline(num_new_seeds=2))

if __name__ == "__main__":
    main()
