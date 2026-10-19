# Diabetes Data Processing & Exploratory Data Analysis

A Python/Jupyter portfolio project that inspects, cleans, summarises, and visualises **two diabetes datasets**:

1. **Uploaded dataset — `Diabetes Dataset(1).csv`**: 20,000 rows and 15 columns. Its target is categorical `Outcome` (e.g. `Diabetic` / `Non-diabetic`). The uploaded CSV is included under `data/raw/`.
2. **CDC Diabetes Health Indicators**: fetched from UCI ML Repository dataset 891 using `ucimlrepo.fetch_ucirepo(id=891)`. The features and target are combined and cached locally. The target is normally `Diabetes_binary`.

## Quick start — Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.run_eda --dataset uploaded
python -m src.run_eda --dataset cdc
python -m src.run_eda --dataset all
jupyter lab
```

The CDC dataset requires internet access the first time it is fetched. It is cached at `data/raw/cdc_diabetes_health_indicators.csv`.

## Notebooks

- `notebooks/01_uploaded_dataset_eda.ipynb` — your uploaded CSV.
- `notebooks/02_cdc_uci_891_eda.ipynb` — imported CDC dataset.
- `notebooks/03_preprocessing_math.ipynb` — descriptive statistics, standardisation and leakage-safe split example.

## What it demonstrates

- Dataset dimensions, schema, data types, missingness, duplicates, unique values, ranges and target balance.
- Histograms, boxplots, missingness charts, target distribution, correlation heatmaps and feature/target comparisons.
- Descriptive statistics including mean, median, variance, standard deviation, IQR and skewness.
- A conservative candidate cleaning step that imputes actual missing values only. It does not assume zero means missing.
- Standardisation and train/test split-aware scaling examples.
- CSV reports and PNG plots saved under `outputs/`.

## Interpretation cautions

- The uploaded dataset has categorical columns and a categorical target; it is not assumed to have the Pima schema.
- Zero is a valid value in many variables. Special zero-as-missing rules should only be applied after validating the data dictionary.
- CDC indicator variables often encode categories as integers; Pearson correlation is exploratory and may not be ideal for every feature.
- Correlation is not causation. EDA helps identify patterns and data issues; it does not reveal an ML model's complete decision function.
- For predictive modelling, split first and fit imputers/scalers on training rows only to avoid data leakage.
- Educational/research use only; not a medical diagnostic tool.
