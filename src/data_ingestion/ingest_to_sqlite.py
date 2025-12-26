import pandas as pd
import sqlite3

# Load raw CSV
raw_df = pd.read_csv("loan_master.csv")

# Load processed CSV
final_df = pd.read_csv("data/processed/loan_master_final.csv")

# Connect to SQLite DB
conn = sqlite3.connect("creditpathai.db")

# Store raw data
raw_df.to_sql(
    name="loan_data_raw",
    con=conn,
    if_exists="replace",
    index=False
)

# Store final data
final_df.to_sql(
    name="loan_data_final",
    con=conn,
    if_exists="replace",
    index=False
)

# Verify ingestion
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM loan_data_raw")
print("Raw data rows:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM loan_data_final")
print("Final data rows:", cursor.fetchone()[0])

conn.close()
