# Streamlit GUI

This folder contains a simple Streamlit interface for the project.

## Run

From the project root:

```bash
conda activate nifty-rtx5060
streamlit run streamlit_gui/app.py
```

## What it does

- Project Status provides dropdowns for available files and browse/upload for custom files
- Runs training via `src/scripts/train.py` using selected train/test/model/scaler paths
- Runs prediction via `src/scripts/predict.py` using selected model/scaler/test/output paths
- Displays and filters the selected predictions CSV
