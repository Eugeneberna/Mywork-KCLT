# Mywork-KCLT
## Knowledge-Enhanced Contrastive Learning for Multilingual Symptom-to-Disease Classification

This repository contains the source code and supporting files for the KCLT project. The project focuses on multilingual symptom-to-disease classification using transformer-based language models, knowledge graph information, and contrastive learning.

---

## Project Overview

KCLT combines multilingual language representations with external medical knowledge to improve symptom-to-disease classification.

The overall workflow consists of the following stages:

```text
Input Symptom-Disease Data
          │
          ▼
Multilingual Data Preparation
          │
          ▼
Language Translation / Processing
          │
          ▼
Knowledge Graph Construction
          │
          ▼
Medical Concept Linking
          │
          ▼
Knowledge-Enhanced Model
          │
          ▼
Contrastive Learning
          │
          ▼
Model Training
          │
          ▼
Evaluation and Statistical Analysis
```

---

## Repository Structure

```text
KCLT/
│
├── train.py
├── train_xl.py
├── baselines_rnn.py
├── common.py
├── data.py
├── model.py
├── losses.py
│
├── s01_translate.py
├── s02_split.py
├── s03_build_kg.py
├── s03b_doid_triples.py
├── s04_link_concepts.py
│
├── kg_check.py
├── stat_validation.py
├── summarise.py
│
├── requirements.txt
├── Symptom2Disease.csv
├── doid_triples.csv
├── doid.obo
│
├── data/
├── kg/
├── kg_doid/
├── preds/
├── runs/
├── logs/
├── tables/
├── xl_k0/
├── xl_k5/
└── backup_doid/
```

---

# 1. Main Source Code

## `train.py`

Main training script for the KCLT model and related model configurations.

It is used for:

- Model training
- Loading training and validation data
- Model configuration
- Knowledge-enhanced training
- Model evaluation
- Saving prediction and training results

---

## `train_xl.py`

Extended training implementation used for multilingual experiments and XL configurations.

It supports experiments involving different knowledge/contrastive-learning configurations, including:

```text
xl_k0/
xl_k5/
```

---

## `baselines_rnn.py`

Contains baseline recurrent neural network implementations used for comparison with the proposed KCLT approach.

The baseline experiments include:

- BiLSTM + Attention
- BiGRU
- RCNN

These models provide comparison results against transformer-based approaches.

---

## `model.py`

Contains the main model architecture and model components used by the project.

The experiments include:

- XLM-RoBERTa
- mBERT
- MediMultiBERT-KCLT
- Knowledge-enhanced models
- Contrastive-learning configurations

---

## `losses.py`

Contains the loss functions used during model training, including the losses required for contrastive-learning experiments.

---

## `data.py`

Contains data-loading and preprocessing utilities used by the training pipeline.

It handles the preparation of datasets required by the models.

---

## `common.py`

Contains common utilities and shared functions used across training, evaluation, and experiment processing.

---

# 2. Data Processing Pipeline

The data-processing workflow is divided into multiple stages.

## `s01_translate.py`

Performs the translation and multilingual data preparation stage.

The project contains multilingual data for languages including:

```text
Hindi
Tamil
French
Spanish
```

Cached multilingual files are maintained under:

```text
data/cache/
```

Examples include:

```text
nllb_hi.csv
nllb_ta.csv
nllb_fr.csv
nllb_es.csv
```

---

## `s02_split.py`

Creates and validates the dataset splits used for experimentation.

The processed dataset contains:

```text
data/train.csv
data/val.csv
data/test.csv
```

A split audit is maintained in:

```text
data/split_audit.json
```

---

## `s03_build_kg.py`

Builds the medical knowledge graph used by the knowledge-enhanced models.

The generated knowledge graph resources are stored under:

```text
kg/
```

Important files include:

```text
nodes.csv
edges.csv
graphs.pt
node_emb.pt
kg_stats.json
linking_stats.json
seed_map.csv
```

---

## `s03b_doid_triples.py`

Processes Disease Ontology (DOID)-related triples used for knowledge graph construction.

The project includes:

```text
doid_triples.csv
doid.obo
```

The corresponding DOID knowledge graph is maintained under:

```text
kg_doid/
```

---

## `s04_link_concepts.py`

Performs medical concept linking between symptom/disease information and knowledge graph concepts.

The generated linking information is stored in the knowledge graph directories.

---

# 3. Knowledge Graph

The project contains knowledge graph resources under:

```text
kg/
kg_doid/
```

### `kg/`

Contains the primary knowledge graph resources:

```text
nodes.csv
edges.csv
graphs.pt
node_emb.pt
kg_stats.json
linking_stats.json
seed_map.csv
```

### `kg_doid/`

Contains the Disease Ontology-based knowledge graph:

```text
nodes.csv
edges.csv
graphs.pt
node_emb.pt
kg_stats.json
linking_stats.json
seed_map.csv
```

These resources support the knowledge-enhanced experiments.

---

# 4. Dataset

The repository contains the base symptom-to-disease dataset and processed datasets used by the experiments.

Important files include:

```text
Symptom2Disease.csv
data/train.csv
data/val.csv
data/test.csv
data/multilingual.csv
```

Language-specific review files are available under:

```text
data/review/
```

including:

```text
review_es.csv
review_fr.csv
review_hi.csv
review_ta.csv
```

---

# 5. Model Experiments

The repository contains experiments for several model configurations.

## Baseline Models

```text
BiLSTM + Attention
BiGRU (subword)
RCNN
```

## Transformer Models

```text
mBERT (Multilingual)
XLM-RoBERTa (Base)
```

## Knowledge-Enhanced Model

```text
XLM-R + KG (no CL)
```

## Contrastive-Learning Model

```text
XLM-R + CL (no KG)
```

## Proposed Model

```text
MediMultiBERT-KCLT
```

The experiments are organized using multiple random seeds to support reproducibility and repeated evaluation.

---

# 6. Experimental Configurations

The repository contains experimental configurations under:

```text
xl_k0/
xl_k5/
```

These directories contain:

- Predictions
- Training runs
- Epoch logs
- Evaluation results
- Model-specific outputs

The configurations are used to investigate the effect of knowledge and contrastive-learning components.

---

# 7. Predictions

Prediction files are stored under:

```text
preds/
```

Additional experimental predictions are available under:

```text
xl_k0/preds/
xl_k0/preds_all/

xl_k5/preds/
xl_k5/preds_all/
```

Predictions are organized according to model and random seed.

Example:

```text
preds/
├── XLM-RoBERTa (Base)/
├── mBERT (Multilingual)/
├── XLM-R + KG (no CL)/
├── XLM-R + CL (no KG)/
├── MediMultiBERT-KCLT/
├── BiLSTM + Attention/
├── BiGRU (subword)/
└── RCNN/
```

---

# 8. Training Runs

Detailed experiment outputs are stored under:

```text
runs/
```

Each experiment may contain:

```text
results.json
classification_report.txt
epoch_log.csv
explanations.csv
```

These files provide:

- Training progress
- Evaluation results
- Classification reports
- Prediction explanations
- Epoch-level performance

---

# 9. Logs

Training and experiment logs are stored under:

```text
logs/
```

The log files record execution details for different models, configurations, and random seeds.

Examples include:

```text
XLM-RoBERTa_(Base)_seed*.log
mBERT_(Multilingual)_seed*.log
MediMultiBERT-KCLT_seed*.log
BiLSTM_+_Attention_seed*.log
BiGRU_(subword)_seed*.log
RCNN_seed*.log
```

---

# 10. Results and Tables

Summary tables are available under:

```text
tables/
```

Important result files include:

```text
main_results.csv
per_language.csv
complexity.csv
epoch_log_KCLT_seed42.csv
```

Statistical analysis files include:

```text
tableA_repeated_runs.csv
tableB_significance.csv
tableC_bootstrap_diff.csv
```

These files contain aggregated experimental results and statistical validation information.

---

# 11. Statistical Validation

The script:

```text
stat_validation.py
```

is used for statistical analysis and validation of experimental results.

It works with repeated-run results and statistical comparison files.

The repository includes:

```text
tableA_repeated_runs.csv
tableB_significance.csv
tableC_bootstrap_diff.csv
```

These files support reporting and validation of experimental comparisons.

---

# 12. Result Summarization

The script:

```text
summarise.py
```

is used to summarize experiment outputs and generate consolidated results from different experimental runs.

---

# 13. Knowledge Graph Validation

The script:

```text
kg_check.py
```

is used to inspect and validate the generated knowledge graph resources.

It can be used to verify:

- Knowledge graph nodes
- Knowledge graph edges
- Graph statistics
- Concept linking information
- Generated graph resources

---

# 14. Requirements

The Python dependencies required for running the project are specified in:

```text
requirements.txt
```

Create a Python virtual environment before installing the dependencies.

### Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 15. Running the Project

After installing the dependencies and preparing the required data, the processing pipeline can be followed in this order.

### Step 1 – Prepare Multilingual Data

```bash
python s01_translate.py
```

### Step 2 – Create Dataset Splits

```bash
python s02_split.py
```

### Step 3 – Build the Knowledge Graph

```bash
python s03_build_kg.py
```

### Step 4 – Generate DOID Triples

```bash
python s03b_doid_triples.py
```

### Step 5 – Link Medical Concepts

```bash
python s04_link_concepts.py
```

### Step 6 – Train the Model

```bash
python train.py
```

For extended experiments:

```bash
python train_xl.py
```

The appropriate command-line arguments should be selected according to the required experiment and configuration.

---

# 16. Baseline Experiments

Baseline models can be evaluated using:

```bash
python baselines_rnn.py
```

The baseline results can then be compared with the transformer-based and knowledge-enhanced models.

---

# 17. Reproducibility

Experiments use multiple random seeds to evaluate the stability of the results.

The repository contains results for multiple seeds, including:

```text
42
101
202
303
404
```

Repeated runs and their corresponding results are stored in:

```text
runs/
preds/
logs/
xl_k0/
xl_k5/
```

---

# 18. Output Files

Depending on the experiment, the following output files may be generated:

```text
results.json
classification_report.txt
epoch_log.csv
predictions.csv
explanations.csv
```

These outputs can be used for:

- Model comparison
- Per-language evaluation
- Statistical analysis
- Error analysis
- Reproducibility checks

---

# 19. Recommended Execution Order

For a fresh setup, the recommended workflow is:

```text
1. Install dependencies
        │
        ▼
2. Prepare required data
        │
        ▼
3. Run multilingual preprocessing
        │
        ▼
4. Create train/validation/test splits
        │
        ▼
5. Build knowledge graph
        │
        ▼
6. Generate DOID knowledge graph resources
        │
        ▼
7. Link medical concepts
        │
        ▼
8. Train baseline models
        │
        ▼
9. Train transformer models
        │
        ▼
10. Train KCLT model
        │
        ▼
11. Generate predictions
        │
        ▼
12. Perform statistical validation
        │
        ▼
13. Summarize experimental results
```

---

# 20. Project Files at a Glance

| File / Folder | Purpose |
|---|---|
| `train.py` | Main model training |
| `train_xl.py` | Extended training experiments |
| `baselines_rnn.py` | Baseline model experiments |
| `model.py` | Model architecture |
| `data.py` | Data loading and preprocessing |
| `losses.py` | Training loss functions |
| `common.py` | Common utilities |
| `s01_translate.py` | Multilingual preprocessing |
| `s02_split.py` | Dataset splitting |
| `s03_build_kg.py` | Knowledge graph construction |
| `s03b_doid_triples.py` | DOID triple processing |
| `s04_link_concepts.py` | Medical concept linking |
| `kg_check.py` | Knowledge graph validation |
| `stat_validation.py` | Statistical validation |
| `summarise.py` | Result summarization |
| `data/` | Dataset and preprocessing files |
| `kg/` | Knowledge graph resources |
| `kg_doid/` | DOID knowledge graph resources |
| `preds/` | Model predictions |
| `runs/` | Experiment results |
| `logs/` | Experiment logs |
| `tables/` | Summary and statistical tables |
| `xl_k0/` | Extended experiments with K=0 configuration |
| `xl_k5/` | Extended experiments with K=5 configuration |
| `requirements.txt` | Python dependencies |

---

# 21. Google Drive Data

The large dataset and result files are maintained separately from this GitHub repository due to their size.

The required data can be accessed from the following Google Drive folder:

**Google Drive:**  
https://drive.google.com/drive/folders/1PwBnbxEer67RKT0tVSYQYAsPWAzIwTWp?usp=sharing

Please download the required data from the Google Drive folder when running the corresponding experiments.
