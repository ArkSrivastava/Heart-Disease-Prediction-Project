from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_heart(df):
    df=df.copy()
    y = (df["num"] > 0).astype(int)
    X=df.drop(columns=['num'])
    imp=SimpleImputer(strategy='median')
    X=imp.fit_transform(X)
    sc=StandardScaler()
    X=sc.fit_transform(X)
    return X,y
