import csv
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from datetime import datetime

# Configuration
MASTER_FILE = "/home/rinkesh-verma/Desktop/INLP Final Project/InlpFinalProject/data/master_news_dataset.csv"
OUTPUT_FILE = "/home/rinkesh-verma/Desktop/INLP Final Project/InlpFinalProject/data/full_dataset_v1.csv"
CHECKPOINT_FILE = "/home/rinkesh-verma/Desktop/INLP Final Project/InlpFinalProject/data/augmentation_checkpoint.csv"
SAMPLE_RATIO = 0.45
BATCH_SIZE = 4 # Reduced batch size for safety on 4GB VRAM

def augment():
    if not os.path.exists(MASTER_FILE):
        print(f"Error: {MASTER_FILE} not found.")
        return

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Models & Tokenizers Directly
    print("Loading Summarization model (DistilBART)...")
    sum_model_name = "sshleifer/distilbart-cnn-12-6"
    sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name).to(device)
    
    print("Loading Translation model (Opus-MT)...")
    trans_model_name = "Helsinki-NLP/opus-mt-en-hi"
    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name).to(device)

    # Read All Original Rows
    print(f"Reading {MASTER_FILE}...")
    original_rows = []
    english_rows = []
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            original_rows.append(row)
            if row['language'] == 'en':
                english_rows.append(row)

    print(f"Total rows: {len(original_rows)}")
    print(f"English rows: {len(english_rows)}")

    # Sample for translation
    target_count = int(len(english_rows) * SAMPLE_RATIO)
    sampled_rows = random.sample(english_rows, target_count)
    print(f"Sampled {len(sampled_rows)} rows for Hindi augmentation.")

    generated_rows = []
    
    # Process in batches
    for i in tqdm(range(0, len(sampled_rows), BATCH_SIZE), desc="Augmenting"):
        batch = sampled_rows[i:i + BATCH_SIZE]
        
        # 1. Summarize content
        contents = [r['content'] for r in batch]
        try:
            inputs = sum_tokenizer(contents, max_length=1024, return_tensors="pt", truncation=True, padding=True).to(device)
            summary_ids = sum_model.generate(inputs["input_ids"], max_length=80, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summaries_text = [sum_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        except Exception as e:
            print(f"Summarization error at batch {i}: {e}")
            summaries_text = [r['content'][:300] for r in batch]

        # 2. Translate Headlines and Summaries
        headlines = [r['headline'] for r in batch]
        
        try:
            # Headlines
            h_inputs = trans_tokenizer(headlines, return_tensors="pt", padding=True, truncation=True).to(device)
            h_translated_ids = trans_model.generate(**h_inputs)
            translated_headlines = [trans_tokenizer.decode(t, skip_special_tokens=True) for t in h_translated_ids]
            
            # Summaries
            s_inputs = trans_tokenizer(summaries_text, return_tensors="pt", padding=True, truncation=True).to(device)
            s_translated_ids = trans_model.generate(**s_inputs)
            translated_summaries = [trans_tokenizer.decode(t, skip_special_tokens=True) for t in s_translated_ids]
            
            # Create new rows
            for idx, r in enumerate(batch):
                new_row = r.copy()
                new_row['headline'] = translated_headlines[idx]
                new_row['content'] = translated_summaries[idx]
                new_row['language'] = 'hi'
                generated_rows.append(new_row)
        except Exception as e:
            print(f"Translation error at batch {i}: {e}")
            continue

        # Incremental Save (Checkpoint) every 500 rows
        if len(generated_rows) % 500 < BATCH_SIZE:
             with open(CHECKPOINT_FILE, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(generated_rows)

    # Final Merge and Sort
    print("Combining datasets...")
    all_rows = original_rows + generated_rows
    
    print("Sorting by date (descending)...")
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except:
            return datetime.min
    
    all_rows.sort(key=lambda x: parse_date(x['date']), reverse=True)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        
    print(f"Success! Final dataset size: {len(all_rows)}")

if __name__ == "__main__":
    augment()
