from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42):
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state, solver='lbfgs')
    clf.fit(X_train, y_train)
    return clf
