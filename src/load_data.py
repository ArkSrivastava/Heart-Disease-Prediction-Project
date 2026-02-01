import pandas as pd

COLUMN_NAMES=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]

def load_heart_data(path="data/processed.cleveland.data"):
    df=pd.read_csv(path,header=None,names=COLUMN_NAMES)
    df=df.replace("?",pd.NA)
    df=df.apply(pd.to_numeric,errors='coerce')
    df['num']=(df['num']>0).astype(int)
    return df
