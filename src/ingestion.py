import pandas as pd
from pathlib import Path

def ingest_data():

    # Correct path: go up one level to find data folder
    raw_path = Path("../data/prosperLoanData.csv")

    df = pd.read_csv(raw_path, low_memory=False)

    # Clean column names
    df.columns = (
        df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
                  .str.lower()
    )

    # Drop ID-like columns
    id_cols = ["listingkey", "listingnumber", "memberkey", "loankey", "channel"]
    df = df.drop(columns=id_cols, errors="ignore")

    # Parse date columns
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove duplicates
    df = df.drop_duplicates()

    # Save output in data folder (one level up)
    output_path = Path("../data/prosper_ingested.csv")
    df.to_csv(output_path, index=False)

    print("Ingestion completed!")
    print("Saved to:", output_path)
    print("Final shape:", df.shape)

if __name__ == "__main__":
    ingest_data()
