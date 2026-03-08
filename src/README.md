# src/
This folder contains all the pipeline execution scripts.
To run the project, execute the numbered scripts sequentially:
- `01_data_cleaning.py`: Cleans raw market data and generates health metrics.
- `02_news_separator.py`: Filters the news dataset for the selected pilot tickers.
- `03_finbert_score.py`: English sentiment extraction using FinBERT.
- `04_muril_score.py`: Hindi sentiment extraction using MuRIL.
- `05_handshake.py`: Fuses English and Hindi scores into daily `handshake.csv` files.
- `generate_synthetic_hindi.py`: Generates synthetic data.
- `train_hindi_muril.py`: Fine-tunes the MuRIL model.

