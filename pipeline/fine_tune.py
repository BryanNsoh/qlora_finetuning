# my_healthpal_project/pipeline/fine_tune.py

import os
import json
import torch
from pathlib import Path
from typing import Dict
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

def tokenize_example(example: Dict[str, str], tokenizer, max_length=512):
    """
    We combine prompt + completion into a single text,
    so the model sees how the prompt leads to that JSON.

    A more sophisticated approach could mask out the prompt portion,
    but this simpler method still works in practice for instruction fine-tuning.
    """
    text = example["prompt"] + "\n" + example["completion"]
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length
    )
    return tokenized

def main_finetune(
    train_file: str = "./data/fine_tuning/train.jsonl",
    val_file:   str = "./data/fine_tuning/val.jsonl",
    base_model: str =  "meta-llama/Llama-3.2-3B-Instruct",
    output_dir: str = "./data/fine_tuning/checkpoints-qlora",
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
):
    """
    Fine-tunes a 4-bit quantized model (QLoRA) on (prompt -> JSON) pairs.

    Usage:
      python fine_tune.py
    """

    # 1) Load training + validation data
    data_files = {}
    if Path(train_file).exists():
        data_files["train"] = train_file
    if Path(val_file).exists():
        data_files["validation"] = val_file
    dataset = load_dataset("json", data_files=data_files)

    # 2) Load tokenizer
    print(f"[Fine-Tune] Loading tokenizer for: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Tokenize
    def _tokenize(ex):
        return tokenize_example(ex, tokenizer, max_length=512)

    train_ds = dataset["train"].map(_tokenize, batched=False)
    val_ds   = dataset["validation"].map(_tokenize, batched=False) if "validation" in dataset else None

    # 4) QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        
        bnb_4bit_use_double_quant=True,   
        bnb_4bit_compute_dtype=torch.bfloat16, 
    )
    print("[Fine-Tune] Loading base model in 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 5) Prepare model for k-bit
    model = prepare_model_for_kbit_training(model)

    # 6) LoRA config
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],  # typical QLoRA approach
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 7) Data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 8) Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        fp16=True,  # or bf16 if your GPU supports it
        evaluation_strategy="epoch" if val_ds else "no",
        save_strategy="epoch" if val_ds else "no",
        logging_steps=50,
        do_eval=bool(val_ds),
        report_to="none",  # or "tensorboard"
    )

    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 10) Train
    print("[Fine-Tune] Starting QLoRA training...")
    trainer.train()
    print("[Fine-Tune] Training done.")

    # 11) Save final LoRA adapter
    model.save_pretrained(output_dir)
    print(f"[Fine-Tune] QLoRA adapter saved to {output_dir}")

if __name__ == "__main__":
    main_finetune()
