from sklearn.svm import SVC

def train_svm(X_train, y_train, C=1.0, kernel='rbf'):
    clf = SVC(C=C, kernel=kernel, probability=True)
    clf.fit(X_train, y_train)
    return clf
