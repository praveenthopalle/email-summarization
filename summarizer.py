import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from config import MODEL_NAME, MAX_LENGTH

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(text):
    """
    Summarizes the given text using the BART model.
    """
    if not text.strip():
        return "No meaningful content to summarize."

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=MAX_LENGTH,
            num_return_sequences=1,
            num_beams=2,
            early_stopping=True
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
