import re
import pandas as pd

def clean_email_body(body):
    """
    Cleans email text by removing disclaimers, signatures, and unnecessary content.
    """
    if not isinstance(body, str):
        return ""  # Handle cases where body is NaN or not a string

    # Remove signatures (text after "--" or similar separators)
    body = re.sub(r'\s*--\s*.*$', '', body, flags=re.MULTILINE)

    # Remove disclaimers (common words used in disclaimers)
    body = re.sub(r'\b(confidential|disclaimer|unsubscribe|automated message)\b.*', '', body, flags=re.IGNORECASE)

    # Remove extra spaces and newlines
    body = re.sub(r'\s+', ' ', body).strip()

    return body

def preprocess_emails(input_csv):
    """
    Reads CSV, extracts necessary columns, and cleans email bodies.
    """
    df = pd.read_csv(input_csv)

    if "body" not in df.columns:
        raise ValueError("CSV file must contain a 'body' column.")

    df["cleaned_body"] = df["body"].apply(clean_email_body)
    return df
