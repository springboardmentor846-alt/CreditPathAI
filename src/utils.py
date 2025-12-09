import pandas as pd

def safe_read_csv(path):
    """
    Reads a CSV file with multiple fallback encodings.
    Returns None if the file cannot be read.
    """
    encodings = ["utf-8", "latin1", "iso-8859-1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    print(f"[WARNING] Could not read file: {path}")
    return None
