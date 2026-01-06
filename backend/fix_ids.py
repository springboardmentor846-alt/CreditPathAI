import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import re

# --- CONFIGURATION ---
DATA_DIR = r"C:/Users/R Sai Charan/OneDrive/Documents/CreditPathAI/home-credit-default-risk/"
SOURCE_FILE = "train_master_full.csv"      # The Joined Data (Has correct IDs like 100002)
TARGET_FILE = "train_ready_for_model.csv"  # The Output File (Corrupted one we need to fix)

def fix_data_ids():
    print("--- REGENERATING DATA (PRESERVING IDs) ---")
    
    # 1. Load Source
    path = os.path.join(DATA_DIR, SOURCE_FILE)
    if not os.path.exists(path):
        print(f"❌ Error: Source file {SOURCE_FILE} not found.")
        return

    print(f"Loading {SOURCE_FILE}...")
    df = pd.read_csv(path)

    # 2. SEPARATE ID AND TARGET (The Protection Step)
    print("Separating IDs to protect them from scaling...")
    
    if 'SK_ID_CURR' not in df.columns:
        print("❌ Error: SK_ID_CURR missing in master file.")
        return

    # SAVE THEM ASIDE
    ids = df['SK_ID_CURR']  
    target = df['TARGET']   
    
    # DROP THEM from the processing table
    df_proc = df.drop(columns=['SK_ID_CURR', 'TARGET'])

    # 3. PROCESS THE REST (Cleaning & Scaling)
    print("Processing features...")
    
    # Feature Extraction (Ratios)
    df_proc['CREDIT_INCOME_PERCENT'] = df_proc['AMT_CREDIT'] / df_proc['AMT_INCOME_TOTAL']
    df_proc['ANNUITY_INCOME_PERCENT'] = df_proc['AMT_ANNUITY'] / df_proc['AMT_INCOME_TOTAL']
    df_proc['CREDIT_TERM'] = df_proc['AMT_CREDIT'] / df_proc['AMT_ANNUITY']
    
    # Cleaning Logic
    if 'CC_UTILIZATION' in df_proc.columns:
        df_proc['CC_UTILIZATION'] = df_proc['CC_UTILIZATION'].replace([np.inf, -np.inf], 0)
    if 'DAYS_EMPLOYED' in df_proc.columns:
        df_proc['DAYS_EMPLOYED'] = df_proc['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    days_cols = [col for col in df_proc.columns if 'DAYS_' in col]
    for col in days_cols:
        df_proc[col] = df_proc[col].abs()

    # Encoding
    df_proc = pd.get_dummies(df_proc, drop_first=True)
    df_proc = df_proc.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    # Imputing (Filling NaNs)
    print("Filling missing values...")
    imputer = SimpleImputer(strategy='median')
    cols = df_proc.columns
    data_imputed = imputer.fit_transform(df_proc)
    
    # Scaling (Normalizing)
    print("Scaling values (Excluding IDs)...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    # Convert back to DataFrame
    df_final = pd.DataFrame(data_scaled, columns=cols)

    # 4. RE-ATTACH THE PROTECTED COLUMNS
    print("Attaching correct IDs back to the file...")
    
    # .values ensures we just paste the numbers without index issues
    df_final['SK_ID_CURR'] = ids.values
    df_final['TARGET'] = target.values

    # 5. SAVE
    out_path = os.path.join(DATA_DIR, TARGET_FILE)
    df_final.to_csv(out_path, index=False)
    
    print("\n" + "="*40)
    print("✅ REPAIR COMPLETE")
    print("="*40)
    print(f"File saved to: {TARGET_FILE}")
    print("First 5 IDs in the new file (Should be 100002, 100003...):")
    print(df_final['SK_ID_CURR'].head().tolist())

if __name__ == "__main__":
    fix_data_ids()