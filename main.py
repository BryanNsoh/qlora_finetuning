# my_healthpal_project/main.py

import asyncio
import json
from pathlib import Path
from typing import List

from pipeline.generate_seed import generate_seeds
from pipeline.generate_transcript import generate_transcripts
from pipeline.extract_info import extract_info_from_transcripts

DATA_DIR = Path(__file__).parent / "data"
SEEDS_FILE = DATA_DIR / "seeds.jsonl"
TRANSCRIPTS_FILE = DATA_DIR / "transcripts.jsonl"
EXTRACTED_FILE = DATA_DIR / "extracted.jsonl"

async def orchestrate_pipeline(
    num_new_seeds: int = 3,
    model_name: str = "deepseek:deepseek-chat"
):
    """
    Orchestrates the pipeline to:
    1) Load existing seeds (if any).
    2) Generate N new seeds and append them to the existing list.
    3) Overwrite seeds.jsonl with the new full list of seeds.
    4) Generate transcripts for ALL seeds (old + new).
    5) Overwrite transcripts.jsonl with the newly generated transcripts for ALL seeds.
    6) Extract data from ALL transcripts and overwrite extracted.jsonl.

    If you want a fresh run with no prior seeds or transcripts, simply delete
    seeds.jsonl, transcripts.jsonl, and extracted.jsonl. They will be recreated.
    """

    existing_seeds = []
    if SEEDS_FILE.exists():
        with SEEDS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_seeds.append(json.loads(line))

    # 1) Generate new seeds
    print(f"Generating {num_new_seeds} new seeds using {model_name}...")
    new_seeds = await generate_seeds(
        num_seeds=num_new_seeds,
        model_name=model_name,
        existing_seeds=existing_seeds
    )

    # 2) Combine existing + new
    all_seeds = existing_seeds + new_seeds

    # 3) Overwrite seeds.jsonl with the combined list
    with SEEDS_FILE.open("w", encoding="utf-8") as f:
        for seed in all_seeds:
            f.write(json.dumps(seed) + "\n")

    # 4) Generate transcripts for ALL seeds
    print("Generating transcripts for ALL seeds (existing + new)...")
    all_transcripts = await generate_transcripts(all_seeds, model_name)

    # 5) Overwrite transcripts.jsonl with transcripts for ALL seeds
    #    We'll keep the same order: each item in all_seeds -> all_transcripts
    with TRANSCRIPTS_FILE.open("w", encoding="utf-8") as f:
        for seed_obj, transcript in zip(all_seeds, all_transcripts):
            data_to_write = {
                "seed_id": seed_obj["seed_id"],
                "transcript": transcript
            }
            f.write(json.dumps(data_to_write) + "\n")

    # 6) Extract structured data from ALL transcripts
    print("Extracting data from ALL transcripts...")
    all_seed_ids = [s["seed_id"] for s in all_seeds]
    extracted_list = await extract_info_from_transcripts(all_seed_ids, all_transcripts, model_name)

    # 7) Overwrite extracted.jsonl
    with EXTRACTED_FILE.open("w", encoding="utf-8") as f:
        for extracted_item in extracted_list:
            f.write(json.dumps(extracted_item) + "\n")

    print("Pipeline complete! All seeds (existing + new) have been processed.")

def main():
    # Feel free to adjust the number of new seeds here:
    num_seeds = 3
    asyncio.run(orchestrate_pipeline(num_new_seeds=num_seeds))

if __name__ == "__main__":
    main()
