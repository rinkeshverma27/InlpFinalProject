import os
import sys
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# Ensure the parent directory is in the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.paths import RAW_DATA_DIR, PROJECT_ROOT

def train_muril_sentiment():
    print("Initiating MuRIL Fine-Tuning for Financial Sentiment...")
    print("Optimization target: 4GB VRAM strict constraints")
    
    data_path = RAW_DATA_DIR / "synthetic_hindi_financial_train.csv"
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")
    
    # Split into train/validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    model_name = "google/muril-base-cased"
    
    print(f"Downloading/Loading tokenizer and base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
        
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # We load num_labels=2 for Binary Sentiment (Positive/Negative)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    output_dir = PROJECT_ROOT / "models" / "muril_financial_sentiment_v1"
    
    # Strict parameters to run steadily without crashing
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,              
        per_device_train_batch_size=4,   
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        use_cpu=True,                    # FORCE CPU TRAINING TO BYPASS 4GB VRAM LIMIT
        logging_dir='./logs/muril_train',
        logging_steps=10,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )
    
    print("Starting Training Loop...")
    trainer.train()
    
    print(f"Training Complete! Saving final optimized model to {output_dir}")
    trainer.save_model(output_dir)
    print("Done.")

if __name__ == "__main__":
    train_muril_sentiment()
