# my_healthpal_project/main.py

import asyncio
import json
from pathlib import Path
from typing import List, Dict

# Our schemas
from schemas.seed_schema import SeedModel
from schemas.extraction_schema import ExtractedDataModel

# Our pipeline steps
from pipeline.generate_seed import generate_seeds
from pipeline.generate_transcript import generate_transcripts_for_new_seeds
from pipeline.extract_info import extract_info_for_new_transcripts

# Data directory and files
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SEEDS_FILE = DATA_DIR / "seeds.jsonl"
TRANSCRIPTS_FILE = DATA_DIR / "transcripts.jsonl"
EXTRACTED_FILE = DATA_DIR / "extracted.jsonl"

async def orchestrate_pipeline(
    num_new_seeds: int = 100,
    model_name: str = "openai:gpt-4o-mini" 
    #model_name: str = "deepseek:deepseek-chat"
):
    """
    Orchestrates the pipeline:
      1) Load existing seeds from seeds.jsonl
      2) Generate `num_new_seeds` new seeds (with automatic seed_id assignment),
         appending them to seeds.jsonl
      3) Generate transcripts only for seeds that don't have any, append to transcripts.jsonl
      4) Extract data only for transcripts that don't have any extraction, append to extracted.jsonl

    If you want a fresh run, delete these files:
      - seeds.jsonl
      - transcripts.jsonl
      - extracted.jsonl
    """

    # --- Step 1: Load existing seeds ---
    existing_seeds: List[SeedModel] = []
    if SEEDS_FILE.exists():
        with SEEDS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seed_data = json.loads(line)
                    existing_seeds.append(SeedModel(**seed_data))

    # --- Step 2: Generate new seeds & append to seeds.jsonl ---
    new_seeds: List[SeedModel] = []
    if num_new_seeds > 0:
        print(f"Generating {num_new_seeds} new seed(s) with model={model_name}...")
        new_seeds = await generate_seeds(num_new_seeds, model_name, existing_seeds)

        with SEEDS_FILE.open("a", encoding="utf-8") as f:
            for seed in new_seeds:
                f.write(json.dumps(seed.model_dump()) + "\n")

    # Combine in-memory
    all_seeds = existing_seeds + new_seeds

    # --- Step 3: Generate transcripts for seeds missing them ---
    existing_transcripts = []
    if TRANSCRIPTS_FILE.exists():
        with TRANSCRIPTS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_transcripts.append(json.loads(line))
                    # Each entry: {"seed_id": "...", "transcript": "..."}

    seeds_with_transcripts = {t["seed_id"] for t in existing_transcripts}
    seeds_missing_transcripts = [s for s in all_seeds if s.seed_id not in seeds_with_transcripts]

    if seeds_missing_transcripts:
        print(f"Generating transcripts for {len(seeds_missing_transcripts)} seed(s)...")
        new_transcripts_data = await generate_transcripts_for_new_seeds(
            seeds_missing_transcripts,
            model_name
        )
        # new_transcripts_data is a list of dicts: {"seed_id":..., "transcript":...}
        with TRANSCRIPTS_FILE.open("a", encoding="utf-8") as f:
            for record in new_transcripts_data:
                f.write(json.dumps(record) + "\n")
    else:
        print("No seeds are missing transcripts. Skipping transcript generation.")

    # --- Step 4: Extract data for transcripts missing extraction ---
    existing_extractions = []
    if EXTRACTED_FILE.exists():
        with EXTRACTED_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_extractions.append(json.loads(line))
                    # Format: matches ExtractedDataModel fields

    seeds_with_extractions = {e["seed_id"] for e in existing_extractions}

    # Reload transcripts so we have them all
    all_transcripts = []
    if TRANSCRIPTS_FILE.exists():
        with TRANSCRIPTS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_transcripts.append(json.loads(line))

    transcripts_missing_extraction = [
        t for t in all_transcripts if t["seed_id"] not in seeds_with_extractions
    ]

    if transcripts_missing_extraction:
        print(f"Extracting data for {len(transcripts_missing_extraction)} transcript(s)...")
        new_extractions = await extract_info_for_new_transcripts(
            transcripts_missing_extraction,
            model_name
        )
        # new_extractions is a list of ExtractedDataModel
        with EXTRACTED_FILE.open("a", encoding="utf-8") as f:
            for extraction in new_extractions:
                f.write(json.dumps(extraction.model_dump()) + "\n")
    else:
        print("No transcripts are missing extraction. Skipping extraction step.")

    # --- Done ---
    print("\n--- Pipeline Complete ---")
    print(f"  Seeds total:        {len(all_seeds)}")
    print(f"  Transcripts total:  {len(all_transcripts)}")
    print(f"  Extractions total:  {len(existing_extractions) + len(transcripts_missing_extraction)}")

def main():
    # Example: generate 3 new seeds on each run
    asyncio.run(orchestrate_pipeline())

if __name__ == "__main__":
    main()
