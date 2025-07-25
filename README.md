# Self‑Retrieval‑Augmented Generation based on Qwen3‑8B
**Self‑Retrieval‑Augmented Generation for the LM‑KBC 2025 Challenge**

This repository contains our submission to the **4ᵗʰ LM‑KBC Challenge** at ISWC 2025.  
Our system combines the 8‑billion‐parameter **Qwen3** language model with a *Self‑Retrieval‑Augmented Generation (Self‑RAG)* strategy: for each subject–relation query, the model first generates a focused **entity description**, then injects this context back into a tailored prompt to improve object prediction quality.

---

## Repository Layout

```
├── models/
│ └── baseline_qwen_3_model.py # Thin wrapper around HuggingFace Qwen3‑8B
├── self_rag_model.py # Self‑RAG logic (description + prompt)
├── configs/
│ └── config.yaml # Default inference configuration
├── prompt_templates/
│ └── question_prompts.csv # Relation‑specific prompt stems
├── data/ # Place LM‑KBC splits here
│ ├── train.jsonl
│ └── val.jsonl
├── main.py # CLI entry point
├── requirements.txt # Python dependencies
└── README.md # (you are here)
```
---

## Quick Start

### 1. Set up

```bash
git clone https://github.com/JingboHH/self-rag-qwen3.git
cd self-rag-qwen3

# create & activate Python 3.11 env (conda or venv)
python -m venv self-rag-qwen3
source self-rag-qwen3/bin/activate

pip install -r requirements.txt
```

GPU memory
Qwen3‑8B requires ≈16 GB (FP16).
If resources are limited, set `use_quantization: true` in `config.yaml`.

### 2. Configure

configs/self-rag-qwen-3.yaml (excerpt):

```yaml
model: "self_rag"
llm_path: "Qwen/Qwen3-8B"
max_new_tokens: 32768

# Self‑RAG
use_description: true
description_max_tokens: 150

# Few‑shot in‑context examples
few_shot: 5
```

Adjust paths to `train_data_file` and other parameters as needed.

### 3. Run Inference

```bash
python main.py --config configs/self-rag-qwen-3.yaml \
               --input data/val.jsonl \
               --output results/self-rag-qwen3-val.jsonl
```

Evaluate on the validation split:

```bash
python evaluate.py \
  -g data/val.jsonl \
  -p results/selfrag-qwen3-val.jsonl
```
---

## Method Overview

```pgsql
Subject + Relation
        │
        ▼
1️⃣  Entity Description
    • Relation‑guided prompt
    • max 150 tokens
        │
        ▼
2️⃣  Enriched Prompt
    “Given this information about <S>: <DESC> … ?”
        │
        ▼
3️⃣  Qwen3‑8B Inference (temperature 0.1)
        │
        ▼
4️⃣  Post‑processing
    • Clean <think> tags
    • Regex for numeric relations
    • Split / filter candidate entities
        │
        ▼
Predicted Object(s)
```

Key idea: the self‑generated description grounds the model, yielding higher precision especially for relations with many or numeric objects.

---

# Validation Results

| Relation                     |   Macro P  |   Macro R  |  Macro F1  |
| ---------------------------- | :--------: | :--------: | :--------: |
| awardWonBy                   |   0.7000   |   0.0000   |   0.0000   |
| companyTradesAtStockExchange |   0.5950   |   0.4392   |   0.3282   |
| countryLandBordersCountry    |   0.6977   |   0.8690   |   0.6640   |
| hasArea                      |   0.1600   |   0.1400   |   0.1400   |
| hasCapacity                  |   0.2700   |   0.0000   |   0.0000   |
| personHasCityOfDeath         |   0.4200   |   0.5700   |   0.2900   |
| **All Relations**            | **0.4162** | **0.3640** | **0.2531** |

# Test Results

| Relation                     |   Macro P  |   Macro R  |  Macro F1  |
| ---------------------------- | :--------: | :--------: | :--------: |
| awardWonBy                   |   0.5611   |   0.0391   |   0.0629   |
| companyTradesAtStockExchange |   0.6050   |   0.5343   |   0.4015   |
| countryLandBordersCountry    |   0.7724   |   0.8049   |   0.7012   |
| hasArea                      |   0.1600   |   0.1100   |   0.1100   |
| hasCapacity                  |   0.2100   |   0.0200   |   0.0200   |
| personHasCityOfDeath         |   0.3900   |   0.6700   |   0.3033   |
| **All Relations**            | **0.4064** | **0.3936** | **0.2748** |





