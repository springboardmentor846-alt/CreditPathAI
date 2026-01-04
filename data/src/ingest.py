import pandas as pd
from sqlalchemy import create_engine
import glob
import os

# Create SQLite DB
engine = create_engine("sqlite:///data/loans.db")

# Ingest Kaggle Dataset
kaggle_files = glob.glob("data/kaggle/*.csv")
for file in kaggle_files:
    df = pd.read_csv(file)
    table_name = "kaggle_" + os.path.splitext(os.path.basename(file))[0]
    df.to_sql(table_name, engine, if_exists="replace", index=False)

# Ingest Microsoft Dataset
ms_files = glob.glob("data/microsoft/*.csv")
for file in ms_files:
    df = pd.read_csv(file)
    table_name = "ms_" + os.path.splitext(os.path.basename(file))[0]
    df.to_sql(table_name, engine, if_exists="replace", index=False)

print("Data ingestion complete!")
