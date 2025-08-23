"""
Preprocessing for Multivariate Time Series Anomaly Detection Hackathon
Loads train.csv, validates timestamps, handles missing values, splits normal/analysis periods.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import os

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load CSV, parse timestamps, validate columns."""
    df = pd.read_csv(file_path, parse_dates=[0])
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df = df.set_index("timestamp")
    # Validate timestamps
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Timestamp column must be datetime type.")
    # Handle missing values
    df = df.ffill().bfill()
    return df

def split_periods(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into normal (training) and full analysis periods."""
    normal_start = pd.Timestamp("2004-01-01 00:00")
    normal_end = pd.Timestamp("2004-01-05 23:59")
    analysis_start = pd.Timestamp("2004-01-01 00:00")
    analysis_end = pd.Timestamp("2004-01-19 07:59")
    train_df = df.loc[normal_start:normal_end].copy()
    analysis_df = df.loc[analysis_start:analysis_end].copy()
    return train_df, analysis_df

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    input_path = os.path.join(data_dir, "train.csv")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_validate_data(input_path)

    # ...existing code...

    train_df, analysis_df = split_periods(df)
    train_df.to_csv(os.path.join(output_dir, "normal_period.csv"))
    analysis_df.to_csv(os.path.join(output_dir, "analysis_period.csv"))
    print("Preprocessing complete. Files saved to outputs/ directory.")

if __name__ == "__main__":
    main()
