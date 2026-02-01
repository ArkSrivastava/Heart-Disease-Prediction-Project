import pandas as pd

def load_kaggle_data(path):
    df = pd.read_csv(path)

    # Kaggle uses "target", my project uses "num"
    if "target" in df.columns:
        df = df.rename(columns={"target": "num"})

    return df
