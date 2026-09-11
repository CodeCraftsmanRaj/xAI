from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data(random_state=42, test_size=0.2):
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names, data.target_names
