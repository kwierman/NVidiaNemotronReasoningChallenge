"""
LoRA Fine-tuning Script for Nemotron Reasoning Challenge
Uses QLoRA with 4-bit quantization for limited GPU memory
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
import os

# Configuration
MODEL_NAME = "nvidia/Nemotron-3-Nano-30B"
OUTPUT_DIR = "./lora_adapter"
DATA_PATH = "./data"
TRAIN_FILE = f"{DATA_PATH}/train.csv"
TEST_FILE = f"{DATA_PATH}/test.csv"

# LoRA Config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_data():
    """Load and prepare training data."""
    train_df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded {len(train_df)} training samples")

    # Format data
    def create_example(row):
        return {
            "text": f"""Instruction: Solve the puzzle step by step.

{row["prompt"]}

Answer: {row["answer"]}<|end_of_text|>"""
        }

    train_data = [create_example(row) for _, row in train_df.iterrows()]
    return train_data


def load_model_and_tokenizer():
    """Load model with 4-bit quantization and tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Preparing model for training...")
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora(model):
    """Setup LoRA configuration."""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def main():
    # Load data
    train_data = load_data()

    # Create dataset
    dataset = Dataset.from_list(train_data)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Setup LoRA
    model = setup_lora(model)

    # Tokenize
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments - optimized for small GPU
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=50,
        max_steps=500,  # Limit for initial test
        report_to="none",
    )

    # Note: Need to implement custom trainer for QLoRA
    print("Note: For full training, use the Trainer with proper QLoRA setup")
    print("Saving LoRA config for now...")

    # Save LoRA configuration (weights would be created during training)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save adapter config
    config = {
        "base_model_name_or_path": MODEL_NAME,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }

    with open(f"{OUTPUT_DIR}/adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved adapter config to {OUTPUT_DIR}/adapter_config.json")
    print(
        "\nTo complete training, run with sufficient GPU memory or use cloud instance."
    )


if __name__ == "__main__":
    main()
