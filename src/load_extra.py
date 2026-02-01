import pandas as pd

def load_extra_data(path):
    df = pd.read_csv(path)

    # this dataset uses "target", my project uses "num"
    df = df.rename(columns={"target": "num"})

    return df
