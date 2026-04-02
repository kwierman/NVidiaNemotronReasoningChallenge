# Nemotron Reasoning Challenge

This repository contains work for the NVIDIA Nemotron Reasoning Challenge, focusing on improving reasoning accuracy using NVIDIA Nemotron models on a logical reasoning puzzle benchmark.

## Overview

The challenge involves solving logical reasoning puzzles that require identification and application of underlying transformation rules (e.g., bit manipulation, algebraic equations). The goal is to create a LoRA adapter for the Nemotron-3-Nano-30B base model.

## Repository Structure

```
├── data/                    # Training and test data
│   ├── train.csv           # Puzzle training set with prompts and answers
│   └── test.csv            # Test set for submission
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_analysis.ipynb       # Data exploration and analysis
│   ├── 02_prompt_engineering.ipynb  # Prompt strategy experiments
│   ├── 03_lora_finetuning.ipynb     # LoRA fine-tuning setup
│   └── 04_solution_complete.ipynb   # Complete solution pipeline
├── scripts/                # Python scripts
│   ├── solution_baseline.py   # Baseline solution
│   ├── solution_complete.py  # Complete solution
│   ├── solution_final.py     # Final optimized solution
│   └── train_lora.py         # LoRA training script
├── lora_adapter/           # Trained LoRA adapter
│   └── adapter_config.json
└── submission.csv          # Competition submission
```

## Approach

1. **Data Analysis**: Explored the puzzle dataset to understand the reasoning task
2. **Prompt Engineering**: Developed prompting strategies for solving the puzzles
3. **LoRA Fine-tuning**: Trained a LoRA adapter on the Nemotron-3-Nano-30B model
4. **Evaluation**: Generated predictions for the test set

## Getting Started

For more details on:
- Dataset structure, see `DATA.md`
- Competition description, see `DESCRIPTION.md`
- NVIDIA Nemotron resources, see `GETTING_STARTED.md`

## Requirements

- Python 3.8+
- transformers
- peft (for LoRA)
- NVIDIA Nemotron-3-Nano-30B model

## Submission

The final submission is a LoRA adapter compatible with the Nemotron-3-Nano-30B base model, along with predictions in `submission.csv`.