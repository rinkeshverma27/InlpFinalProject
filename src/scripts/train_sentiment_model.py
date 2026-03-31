"""
train_sentiment_model.py
Fine-tunes google/muril-base-cased on synthetic Hindi financial sentiment data.

Uses a manual training loop instead of HuggingFace Trainer to avoid
PyTorch distributed/Trainer compatibility issues across versions.

Usage:
    python src/scripts/train_sentiment_model.py
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# Resolve paths locally
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class SentimentDataset(Dataset):
    """Simple PyTorch dataset for tokenized text + labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_muril_sentiment():
    print("=" * 70)
    print("MuRIL Fine-Tuning for Hindi Financial Sentiment")
    print("=" * 70)

    data_path = RAW_DATA_DIR / "synthetic_hindi_financial_train.csv"
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        print("Run src/sentiment/synthetic_data_gen.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")

    # Split into train/validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

    model_name = "google/muril-base-cased"
    output_dir = PROJECT_ROOT / "models" / "muril_financial_sentiment_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(
        train_df['text'].tolist(),
        padding='max_length', truncation=True, max_length=64, return_tensors='pt'
    )
    val_encodings = tokenizer(
        val_df['text'].tolist(),
        padding='max_length', truncation=True, max_length=64, return_tensors='pt'
    )

    train_dataset = SentimentDataset(train_encodings, train_df['label'].tolist())
    val_dataset = SentimentDataset(val_encodings, val_df['label'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────
    print(f"Loading base model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # ── Training Config ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_epochs = 3
    best_val_acc = 0.0

    print(f"\nTraining for {num_epochs} epochs on CPU...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("-" * 50)

    for epoch in range(num_epochs):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  → Best model saved! (Val Acc: {val_acc:.4f})")

    print("-" * 50)
    print(f"\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    train_muril_sentiment()
