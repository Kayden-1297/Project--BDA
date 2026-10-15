# Project--BDA

This repository now includes a from-scratch Big Data ML training script for lung cancer prediction:

- `/home/runner/work/Project--BDA/Project--BDA/train_lung_cancer_model.py`

## What it does

- Loads lung-cancer dataset from CSV or Parquet
- Handles missing values and categorical encoding using Spark ML pipeline stages
- Builds and tunes **Logistic Regression** and **Random Forest** models with cross-validation
- Evaluates using AUC, accuracy, and F1
- Saves the best model and metrics JSON

## Setup (works in any IDE)

1. Open the repo in your IDE.
2. Create and activate a Python environment.
3. Install dependencies:

```bash
pip install -r /home/runner/work/Project--BDA/Project--BDA/requirements.txt
```

## Run

```bash
python /home/runner/work/Project--BDA/Project--BDA/train_lung_cancer_model.py \
  --dataset-path /absolute/path/to/lung_cancer_dataset.csv \
  --target-col LUNG_CANCER \
  --dataset-format auto \
  --test-size 0.2 \
  --seed 42 \
  --output-dir /absolute/path/to/output_artifacts
```

## Output

- Best model folder: `<output-dir>/best_model`
- Metrics file: `<output-dir>/metrics.json`