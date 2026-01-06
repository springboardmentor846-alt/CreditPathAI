import pandas as pd
import numpy as np
import gc
import os

# --- 1. SETUP PATHS ---
# UPDATE THIS PATH to your folder containing all the CSVs
DATA_DIR = r"C:/Users/R Sai Charan/OneDrive/Documents/CreditPathAI/home-credit-default-risk/"

def load_data(file_name):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"⚠️ Warning: {file_name} not found. Skipping...")
        return None
    print(f"Loading {file_name}...")
    return pd.read_csv(path)

# --- 2. PROCESS BUREAU (External History) ---
def process_bureau_and_balance():
    bureau = load_data('bureau.csv')
    bb = load_data('bureau_balance.csv')
    
    if bureau is None: return None

    # Step A: Aggregating Bureau Balance (Month-by-Month status)
    # We count how many times a loan was 'Closed' (C) or 'X' status
    if bb is not None:
        bb_agg = bb.groupby('SK_ID_BUREAU').agg({
            'MONTHS_BALANCE': 'count' # Duration of history
        }).reset_index()
        bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Step B: Aggregate Bureau Data by Customer (SK_ID_CURR)
    # We want Total Debt, Max Overdue, and Count of Active Loans
    print("Aggregating Bureau Data...")
    
    # Create a flag: Is the loan currently active?
    bureau['CREDIT_ACTIVE_BINARY'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',                 # Total number of external loans
        'AMT_CREDIT_SUM': 'sum',                 # Total amount borrowed externally
        'AMT_CREDIT_SUM_DEBT': 'sum',            # Total current external debt
        'AMT_CREDIT_SUM_OVERDUE': 'sum',         # Total amount overdue
        'CREDIT_ACTIVE_BINARY': 'mean',          # % of loans that are still active
        'DAYS_CREDIT': 'max'                     # How recent was the last loan?
    }).reset_index()
    
    # Rename columns so we know they came from Bureau
    bureau_agg.columns = ['SK_ID_CURR', 'BUREAU_COUNT', 'BUREAU_TOTAL_CREDIT', 
                          'BUREAU_TOTAL_DEBT', 'BUREAU_TOTAL_OVERDUE', 
                          'BUREAU_ACTIVE_RATIO', 'BUREAU_LAST_LOAN_DAYS']
    
    del bureau, bb
    gc.collect()
    return bureau_agg

# --- 3. PROCESS PREVIOUS APPLICATIONS (Internal History) ---
def process_previous_apps():
    prev = load_data('previous_application.csv')
    if prev is None: return None
    
    print("Aggregating Previous Applications...")
    
    # Flag: Was it refused?
    prev['APP_REFUSED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',              # How many times did they apply before?
        'AMT_APPLICATION': 'mean',          # Average amount they asked for
        'APP_REFUSED': 'sum',               # Total times we rejected them
        'CNT_PAYMENT': 'mean'               # Average term (months) of previous loans
    }).reset_index()
    
    prev_agg.columns = ['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_AVG_AMT', 
                        'PREV_TOTAL_REFUSALS', 'PREV_AVG_TERM']
    
    del prev
    gc.collect()
    return prev_agg

# --- 4. PROCESS INSTALLMENTS (Repayment Behavior) ---
# THIS IS THE MOST IMPORTANT TABLE FOR RISK
def process_installments():
    inst = load_data('installments_payments.csv')
    if inst is None: return None
    
    print("Aggregating Installment Payments...")
    
    # Feature 1: Days Late (Actual - Scheduled)
    # Positive = Late, Negative = Early
    inst['DAYS_LATE'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['DAYS_LATE'] = inst['DAYS_LATE'].apply(lambda x: x if x > 0 else 0) # Only count lateness
    
    # Feature 2: Underpayment (Scheduled - Actual)
    # Positive = They paid less than they should have
    inst['AMT_UNDERPAID'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['AMT_UNDERPAID'] = inst['AMT_UNDERPAID'].apply(lambda x: x if x > 0 else 0)

    inst_agg = inst.groupby('SK_ID_CURR').agg({
        'DAYS_LATE': ['mean', 'max'],       # Avg lateness and Max lateness
        'AMT_UNDERPAID': ['sum', 'mean']    # Total money they failed to pay
    }).reset_index()
    
    # Flatten columns
    inst_agg.columns = ['SK_ID_CURR', 'INST_AVG_LATE', 'INST_MAX_LATE', 
                        'INST_TOTAL_UNDERPAID', 'INST_AVG_UNDERPAID']
    
    del inst
    gc.collect()
    return inst_agg

# --- 5. PROCESS POS CASH (Point of Sale) ---
def process_pos_cash():
    pos = load_data('POS_CASH_balance.csv')
    if pos is None: return None
    
    print("Aggregating POS Cash Balance...")
    
    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'SK_DPD': 'mean',          # Average Days Past Due
        'SK_DPD_DEF': 'max'        # Max Days Past Due (tolerance)
    }).reset_index()
    
    pos_agg.columns = ['SK_ID_CURR', 'POS_AVG_DPD', 'POS_MAX_DPD']
    
    del pos
    gc.collect()
    return pos_agg

# --- 6. PROCESS CREDIT CARD ---
def process_credit_card():
    cc = load_data('credit_card_balance.csv')
    if cc is None: return None
    
    print("Aggregating Credit Card Balance...")
    
    # Utilization calculation requires Limit > 0
    cc['AMT_CREDIT_LIMIT_ACTUAL'] = cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    cc['UTILIZATION'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    
    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'UTILIZATION': 'mean',              # Average card usage %
        'AMT_DRAWINGS_ATM_CURRENT': 'sum',  # Total Cash Withdrawals (High Risk)
        'SK_DPD': 'mean'                    # Days past due
    }).reset_index()
    
    cc_agg.columns = ['SK_ID_CURR', 'CC_UTILIZATION', 'CC_ATM_DRAWINGS', 'CC_AVG_DPD']
    
    del cc
    gc.collect()
    return cc_agg

# --- MAIN EXECUTION ---
def main():
    # 1. Load Main Train/Test
    df_train = load_data('application_train.csv')
    df_test = load_data('application_test.csv')
    
    print(f"Base Train Shape: {df_train.shape}")
    
    # 2. Get all aggregated features
    bureau_data = process_bureau_and_balance()
    prev_data = process_previous_apps()
    inst_data = process_installments()
    pos_data = process_pos_cash()
    cc_data = process_credit_card()
    
    # 3. Merge everything into Train and Test
    # We use a LIST of dataframes to loop through and merge
    aux_datasets = [bureau_data, prev_data, inst_data, pos_data, cc_data]
    
    print("Merging all datasets...")
    for aux_df in aux_datasets:
        if aux_df is not None:
            df_train = df_train.merge(aux_df, on='SK_ID_CURR', how='left')
            df_test = df_test.merge(aux_df, on='SK_ID_CURR', how='left')
            
    # 4. Fill NaNs created by merging
    # (If a user has no Credit Card, their CC_UTILIZATION is 0, not NaN)
    print("Handling Missing Values...")
    new_cols = [c for c in df_train.columns if c not in load_data('application_train.csv').columns]
    df_train[new_cols] = df_train[new_cols].fillna(0)
    df_test[new_cols] = df_test[new_cols].fillna(0)
    
    # 5. Save Final Files
    print("Saving Master Files...")
    df_train.to_csv(os.path.join(DATA_DIR, "train_master_full.csv"), index=False)
    df_test.to_csv(os.path.join(DATA_DIR, "test_master_full.csv"), index=False)
    
    print(f"✅ SUCCESS! Final Train Shape: {df_train.shape}")
    print("You can now use 'train_master_full.csv' for Modeling.")

if __name__ == "__main__":
    main()