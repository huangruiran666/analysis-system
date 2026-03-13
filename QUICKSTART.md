# Quick Start

## Install

```bash
git clone https://github.com/huangruiran666/analysis-system.git
cd analysis-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the Demo

```bash
python example_pipeline.py
```

You should see:

- synthetic transaction generation
- ETL progress output
- sparse matrix statistics
- latent factor shapes
- model comparison metrics

## Explore the Modules

- `src/etl/extract.py`
- `src/features/matrix_factorization.py`
- `src/features/time_series.py`
- `src/models/xgboost_trainer.py`
- `src/models/lightgbm_trainer.py`

## Install as a Package

```bash
pip install -e .
hdcdp-pipeline
```
