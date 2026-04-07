import pandas as pd
import os

def merge_parts(output_filename="final_nifty50_context_complete.csv"):
    parts = ["nifty_context_part1.csv", "nifty_context_part2.csv", "nifty_context_part3.csv"]
    
    all_dfs = []
    for part in parts:
        if os.path.exists(part):
            print(f"Merging {part}...")
            all_dfs.append(pd.read_csv(part))
        else:
            print(f"Warning: {part} not found. Skipping.")
            
    if not all_dfs:
        print("No part files found to merge.")
        return
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates just in case
    final_df.drop_duplicates(subset=['url'], inplace=True)
    
    final_df.to_csv(output_filename, index=False)
    print(f"\n✅ Merge Complete! Final dataset: {output_filename}")
    print(f"Total Rows: {len(final_df)}")

if __name__ == "__main__":
    merge_parts()
