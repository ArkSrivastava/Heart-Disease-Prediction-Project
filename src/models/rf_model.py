from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train,y_train,n_estimators=150,random_state=42):
    clf=RandomForestClassifier(n_estimators=n_estimators,random_state=random_state)
    clf.fit(X_train,y_train)
    return clf
