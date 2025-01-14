# my_healthpal_project/pipeline/fine_tune.py

"""
A stub file for (optional) fine-tuning steps (e.g., QLoRA).
Currently not implemented, per specification.
"""

def prepare_training_data(transcripts: list, extracted_data: list) -> None:
    """
    Example function: combine transcripts with extracted data
    into a training-friendly format (JSONL, etc.).
    This is only a stub.
    """
    # Potentially build pairs (transcript -> extracted JSON)
    # Save to data/training.jsonl
    pass

def run_finetuning_model():
    """
    Stub for QLoRA or LoRA finetuning with a local model.
    Implementation is out of scope for this iteration.
    """
    pass
