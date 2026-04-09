import csv
import sys
import argparse
from pathlib import Path

# Increase the CSV field size limit to handle absolutely massive scraped article bodies
csv.field_size_limit(sys.maxsize)

def clean_scraped_directory(input_dir: str = "data/raw/news", use_ai_summarizer: bool = False):
    """
    Scans the raw news directory for scraped CSVs, fixes column names, 
    and either truncates massive article bodies or uses AI to summarize them.
    """
    raw_dir = Path(input_dir)
    if not raw_dir.exists():
        print(f"Directory not found: {raw_dir}")
        return

    csv_files = [f for f in raw_dir.glob("*.csv") if not f.name.startswith("cleaned_")]

    if not csv_files:
        print("No raw scraped CSVs found to clean.")
        return

    summarizer = None
    if use_ai_summarizer:
        print("Loading AI Summarizer (this may take a moment)...")
        from transformers import pipeline
        # Using a fast, lightweight model for summarization
        # Note: This is primarily trained on English.
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        print("AI Summarizer loaded successfully!")

    for input_file in csv_files:
        output_file = raw_dir / f"cleaned_{input_file.name}"
        backup_file = raw_dir / f"{input_file.name}.bak"
        
        print(f"Processing: {input_file.name}...")
        
        valid_rows = 0
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8', newline='') as fout:
            
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            
            try:
                header = next(reader)
            except StopIteration:
                print(f"  -> File is empty. Skipping.")
                continue
                
            new_header = []
            body_idx = -1
            headline_idx = -1
            
            for i, col in enumerate(header):
                col_lower = col.strip().lower()
                if col_lower == 'timestamp':
                    new_header.append('datetime')
                elif col_lower == 'summary':
                    new_header.append('body')
                    body_idx = i
                else:
                    new_header.append(col_lower)
                
                if col_lower == 'headline':
                    headline_idx = i
                    
            writer.writerow(new_header)
            
            for row in reader:
                if len(row) != len(header):
                    continue  
                
                if headline_idx != -1 and row[headline_idx]:
                    row[headline_idx] = row[headline_idx].replace('\n', ' ').replace('\r', '')
                
                if body_idx != -1 and row[body_idx]:
                    raw_text = row[body_idx].replace('\n', ' ').replace('\r', '')
                    
                    if use_ai_summarizer and summarizer and len(raw_text) > 150:
                        try:
                            # distilbart roughly handles 1024 tokens (~4000 characters). 
                            # We slice to 4000 chars to avoid memory/crash issues before summarizing.
                            input_text = raw_text[:4000]
                            # Generate a summary (min 10 words, max 50 words)
                            summary_output = summarizer(input_text, max_length=50, min_length=10, do_sample=False)
                            row[body_idx] = summary_output[0]['summary_text']
                            print(f"  [AI Summarized Row {valid_rows+1}]")
                        except Exception as e:
                            # Fallback if the summarizer fails (e.g. language unsupported / too long)
                            row[body_idx] = raw_text[:500]
                    else:
                        row[body_idx] = raw_text[:500]
                    
                writer.writerow(row)
                valid_rows += 1
                
        # Rename original file to .bak
        input_file.rename(backup_file)
        print(f"  -> Saved clean file to {output_file.name} ({valid_rows} records)")
        print(f"  -> Backed up original to {backup_file.name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess scraped Nifty news")
    parser.add_argument("--summarize", action="store_true", help="Use an AI model to actually summarize the article bodies.")
    args = parser.parse_args()

    print("Starting News Preprocessing Utility...")
    clean_scraped_directory(use_ai_summarizer=args.summarize)
    print("Done!")
