import os
import pandas as pd
from sqlalchemy import create_engine
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, DB_PATH
from src.utils import safe_read_csv

def load_raw_data():
    """
    Loads all CSV files from data/raw folder.
    Returns a dictionary: {filename: dataframe}
    """
    data = {}

    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith(".csv"):
            path = os.path.join(RAW_DATA_PATH, file)
            df = safe_read_csv(path)

            if df is not None:
                data[file] = df
            else:
                print(f"[WARNING] Skipped unreadable file: {file}")

    return data


def save_processed_csv(df: pd.DataFrame, name: str):
    """
    Saves cleaned CSV files to data/processed
    """
    output_path = os.path.join(PROCESSED_DATA_PATH, name)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved: {output_path}")


def load_to_sqlite(df: pd.DataFrame, table_name: str):
    """
    Loads a DataFrame into SQLite database creditpath.db
    """
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    print(f"[INFO] Loaded table '{table_name}' into SQLite DB")