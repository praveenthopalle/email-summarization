import os

MODEL_NAME = "facebook/bart-large-cnn"  # High-quality BART model
CSV_FOLDER = "emails_data"  # Folder containing multiple CSV files
OUTPUT_CSV = "summarized_emails.csv"  # Output summarized file
BATCH_SIZE = 8  # Process multiple emails at once for CPU optimization
MAX_LENGTH = 150  # Maintains high-quality summary length
USE_ONNX = True  # Enables ONNX optimization for CPU

# Ensure folder exists
if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER)
