import json
import joblib
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from dataset import load_data


def main():

    # -----------------------------
    # Load configuration
    # -----------------------------
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # -----------------------------
    # Load dataset
    # -----------------------------
    X_train, X_test, y_train, y_test = load_data(
        random_state=config["experiment"]["random_state"],
        test_size=config["data"]["test_size"]
    )

    # -----------------------------
    # Load trained model
    # -----------------------------
    model = joblib.load(config["paths"]["model"])

    print("\nModel loaded successfully.")

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.tolist()
    }

    # -----------------------------
    # Print results
    # -----------------------------
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Malignant",
                "Benign"
            ]
        )
    )

    # -----------------------------
    # Save metrics
    # -----------------------------
    with open(config["paths"]["metrics"], "w") as file:
        json.dump(metrics, file, indent=4)

    print(
        f"\nMetrics saved to: "
        f"{config['paths']['metrics']}"
    )


if __name__ == "__main__":
    main()