# Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Focus](https://img.shields.io/badge/Focus-ETL%20%2B%20ML-0ea5e9.svg)](.)

Compact demo repository for alternative-data style ETL, feature engineering, and model training workflows aimed at financial analytics and data-engineering portfolios.

## What Is In This Repo

- `example_pipeline.py`: runnable end-to-end demonstration
- `src/etl/extract.py`: missing-value handling, outlier filtering, sparse matrix creation
- `src/features/`: latent factor generation and time-series feature helpers
- `src/models/`: XGBoost and LightGBM style trainer wrappers with graceful fallbacks
- `QUICKSTART.md` and `PROJECT_SUMMARY.md`: usage and portfolio framing

## Quick Start

```bash
git clone https://github.com/huangruiran666/analysis-system.git
cd analysis-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python example_pipeline.py
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
python example_pipeline.py
```

## Demo Workflow

The example pipeline:

1. Generates synthetic transaction data
2. Cleans missing values and filters outliers
3. Builds a sparse user-item matrix
4. Produces latent-factor and time-series features
5. Trains two regressor pipelines and compares metrics

## Repository Layout

```text
analysis-system/
├── example_pipeline.py
├── src/
│   ├── etl/
│   ├── features/
│   └── models/
├── QUICKSTART.md
├── PROJECT_SUMMARY.md
├── requirements.txt
└── setup.py
```

## Notes

- This is a compact reference implementation, not a full production platform.
- Trainer modules fall back to scikit-learn estimators when XGBoost or LightGBM are unavailable.
- The repository is optimized for learning, demos, and portfolio presentation.

## License

MIT
