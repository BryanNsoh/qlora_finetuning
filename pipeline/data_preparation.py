# my_healthpal_project/pipeline/data_preparation.py

import json
import random
from pathlib import Path

def prepare_finetune_data(
    transcripts_file: str = "./data/transcripts.jsonl",
    extracted_file:   str = "./data/extracted.jsonl",
    output_dir:       str = "./data/fine_tuning",
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
):
    """
    1) Reads transcripts.jsonl and extracted.jsonl from ./data/.
    2) Merges them via seed_id, removing 'seed_id' from the final JSON.
    3) Creates a richly instructive XML-style prompt for the model, 
       telling it EXACTLY how to parse the conversation into the 
       8 fields of the extraction schema (minus seed_id).
    4) Splits into train/val/test, writes them to ./data/fine_tuning.
    """
    transcripts_path = Path(transcripts_file)
    extracted_path = Path(extracted_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load transcripts: {seed_id -> transcript}
    transcripts_dict = {}
    with transcripts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record["seed_id"]
            transcripts_dict[sid] = record["transcript"]

    # 2) Load extracted data: {seed_id -> dict minus seed_id}
    extracted_dict = {}
    with extracted_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record["seed_id"]
            record.pop("seed_id", None)  # remove so it won't appear in completions
            extracted_dict[sid] = record

    # 3) Build (prompt, completion) pairs with strong, detailed XML instructions
    pairs = []
    for sid, transcript_text in transcripts_dict.items():
        if sid not in extracted_dict:
            # skip if no extraction
            continue

        # The final JSON we expect:
        completion_obj = extracted_dict[sid]
        completion_str = json.dumps(completion_obj, ensure_ascii=False)

        # A more instructive, thorough XML prompt:
        # We clarify EXACTLY how to interpret the conversation 
        # and produce the final JSON object.
        prompt_str = f"""
<EXTRACTION_REQUEST>
  <TRANSCRIPT_TEXT>
{transcript_text}
  </TRANSCRIPT_TEXT>
  <TASK_DETAILS>
    You are an AI assistant tasked with extracting key clinical information 
    from the above conversation between doctor and patient (possibly others). 
    Focus on:
      - Primary complaint(s)
      - Diagnoses (new or existing)
      - Medications named
      - Medication instructions (dosing, side effects, usage details)
      - Tests/imaging discussed
      - Follow-up instructions (appointments, lifestyle, next steps)
      - Caregiver or family involvement, if any
      - Any ongoing therapies

    If the transcript does NOT mention a particular field, 
    set it to null (for strings) or an empty array (for lists).

    Be sure to capture these fields strictly in the final JSON. 
    No extra fields. 
  </TASK_DETAILS>
  <SCHEMA_REQUIREMENTS>
    You MUST return a single valid JSON object with these keys:
      "primary_complaint": string or null
      "diagnoses": array of strings
      "medications_discussed": array of strings
      "medication_instructions": array of strings
      "tests_discussed": array of strings
      "follow_up_instructions": array of strings
      "caregiver_involvement": string or null
      "ongoing_therapies_discussed": string or null

    Absolutely no other keys or text. 
    If uncertain, set the field to null or an empty array. 
  </SCHEMA_REQUIREMENTS>
  <FINAL_OUTPUT>
    Please produce ONLY the JSON object, nothing else:
    {{
      "primary_complaint": "...",
      "diagnoses": [...],
      "medications_discussed": [...],
      "medication_instructions": [...],
      "tests_discussed": [...],
      "follow_up_instructions": [...],
      "caregiver_involvement": "... or null",
      "ongoing_therapies_discussed": "... or null"
    }}
  </FINAL_OUTPUT>
</EXTRACTION_REQUEST>
""".strip("\n")

        pairs.append({
            "prompt": prompt_str,
            "completion": completion_str
        })

    # 4) Shuffle and split
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(train_ratio * n)
    val_end   = int((train_ratio + val_ratio) * n)

    train_data = pairs[:train_end]
    val_data   = pairs[train_end:val_end]
    test_data  = pairs[val_end:]

    # 5) Write to JSONL
    def save_jsonl(lst, name):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for item in lst:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_data, "train.jsonl")
    save_jsonl(val_data,   "val.jsonl")
    save_jsonl(test_data,  "test.jsonl")

    print(
        f"[Data Prep] Created {len(train_data)} train, "
        f"{len(val_data)} val, {len(test_data)} test pairs "
        f"in {output_dir}."
    )

if __name__ == "__main__":
    prepare_finetune_data()
