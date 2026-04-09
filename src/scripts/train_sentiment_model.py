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
# Define paths locally since src.utils.paths is missing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def train_muril_sentiment():
    print("Initiating MuRIL Fine-Tuning for Financial Sentiment...")
    print("Optimization target: 4GB VRAM strict constraints")
    
    data_path = PROJECT_ROOT / "data" / "inputs" / "mega_synthetic_hindi_train.csv"
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    # ── Focus on Distinct Sentences ──
    df = df.drop_duplicates(subset=['text']).copy()
    print(f"Loaded {len(df)} distinct samples.")
    
    # Split into train/validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # ... (Dataset conversion same) ...
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    model_name = "google/muril-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32) # Compact
        
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # ── FREEZE ALL BRAIN CELLS (Encoder) ──
    # We only train the classification head
    for name, param in model.named_parameters():
        if "classifier" not in name: # Only keep the 'head' active
            param.requires_grad = False
    
    output_dir = PROJECT_ROOT / "models" / "muril_financial_sentiment_v1"
    
    print(f"🚀 Turbo Training on: GPU (CUDA + Frozen Encoder)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,               # More epochs since we only train the head
        per_device_train_batch_size=8,    # Can fit more in VRAM now!
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,   
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=1e-3,               # Higher LR for head training
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,                        # Essential
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
