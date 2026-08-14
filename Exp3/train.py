import joblib
import yaml

from dataset import load_data
from model import create_model
from utils import create_directories, plot_feature_importance


def main():

    # -----------------------------
    # Load configuration
    # -----------------------------
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    create_directories(config)

    # -----------------------------
    # Load dataset
    # -----------------------------
    X_train, X_test, y_train, y_test = load_data(
        random_state=config["experiment"]["random_state"],
        test_size=config["data"]["test_size"]
    )

    print("\nDataset loaded")
    print("-" * 50)

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    # -----------------------------
    # Create model
    # -----------------------------
    model = create_model(config)

    # -----------------------------
    # Train model
    # -----------------------------
    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # -----------------------------
    # Save model
    # -----------------------------
    model_path = config["paths"]["model"]

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")

    # -----------------------------
    # Feature importance
    # -----------------------------
    plot_feature_importance(
        model,
        X_train.columns,
        config["paths"]["importance_plot"]
    )

    print(
        f"Feature importance plot saved to: "
        f"{config['paths']['importance_plot']}"
    )


if __name__ == "__main__":
    main()