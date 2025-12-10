import pandas as pd

def missing_value_summary(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing


def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()