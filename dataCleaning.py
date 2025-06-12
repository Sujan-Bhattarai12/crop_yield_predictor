import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Rename columns for clarity
    df.rename(columns={'Irrigation_Access_%': 'Irrigation_Access_percent'}, inplace=True) 
    return df

def main():
    filepath = "climate_change_impact_on_agriculture_2024.csv"
    df = load_data(filepath)
    df_clean = clean_data(df)

    print("Cleaned data shape:", df_clean.shape)
    print("Columns:", df_clean.columns.tolist())

if __name__ == "__main__":
    main()
