import os
import pandas as pd
import torch
from preprocess import preprocess_emails
from summarizer import summarize_text
from config import CSV_FOLDER, OUTPUT_CSV, BATCH_SIZE

def load_all_csvs(folder_path):
    """
    Loads and merges all CSV files in the given folder into a single DataFrame.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    print(f"Found {len(all_files)} CSV files. Merging all into a single dataset...")
    df_list = [pd.read_csv(file) for file in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    
    return merged_df

def process_and_summarize():
    """
    Loads all CSV files, processes them, summarizes emails, and saves results.
    """
    print("Loading and cleaning emails from all CSV files...")
    df = load_all_csvs(CSV_FOLDER)
    
    # Preprocess emails
    df = preprocess_emails(df)

    print("Summarizing emails in batches...")

    # Convert DataFrame to List for batch processing
    email_texts = df["cleaned_body"].tolist()

    # Batch processing for efficiency
    summaries = []
    for i in range(0, len(email_texts), BATCH_SIZE):
        batch = email_texts[i:i + BATCH_SIZE]
        summaries.extend([summarize_text(text) for text in batch])

    df["summary"] = summaries

    # Save summarized data
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Summarized emails saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    torch.set_num_threads(8)  # Use multiple CPU cores efficiently
    process_and_summarize()
