import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(random_state=42, test_size=0.2):
    data = load_breast_cancer(as_frame=True)

    X = data.data.copy()
    y = data.target.copy()

    # Convert sklearn column names to readable names
    X.columns = [
        column.replace("_", " ")
        for column in X.columns
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """
    Random Forest does not require scaling.
    This function is kept modular so preprocessing
    can easily be changed for another dataset/model.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler