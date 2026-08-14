import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_pdp(
    model,
    X_test,
    feature,
    points=50,
    lower_percentile=5,
    upper_percentile=95
):

    # ------------------------------------------------
    # Step 1:
    # Determine feature range from observed data
    # ------------------------------------------------
    lower = np.percentile(
        X_test[feature],
        lower_percentile
    )

    upper = np.percentile(
        X_test[feature],
        upper_percentile
    )

    print("\nPDP Configuration")
    print("-" * 50)

    print(f"Feature          : {feature}")
    print(f"Lower percentile : {lower_percentile}")
    print(f"Upper percentile : {upper_percentile}")
    print(f"Feature minimum  : {lower:.4f}")
    print(f"Feature maximum  : {upper:.4f}")

    # ------------------------------------------------
    # Step 2:
    # Generate K equally spaced values
    # ------------------------------------------------
    R = np.linspace(
        lower,
        upper,
        points
    )

    # ------------------------------------------------
    # Step 3:
    # Store partial dependence values
    # ------------------------------------------------
    PD = []

    # ------------------------------------------------
    # Step 4:
    # Repeat for every value in R
    # ------------------------------------------------
    for value in R:

        # Make a copy of test data
        X_copy = X_test.copy()

        # Replace entire feature column
        X_copy[feature] = value

        # Predict probability
        predictions = model.predict_proba(
            X_copy
        )[:, 1]

        # Average predictions
        average_prediction = np.mean(
            predictions
        )

        # Store result
        PD.append(average_prediction)

    # ------------------------------------------------
    # Convert to numpy array
    # ------------------------------------------------
    PD = np.array(PD)

    return R, PD


def plot_pdp(R, PD, feature, path):

    plt.figure(figsize=(10, 6))

    plt.plot(
        R,
        PD,
        linewidth=2
    )

    plt.xlabel(feature)
    plt.ylabel("Average Predicted Probability")

    plt.title(
        f"Partial Dependence Plot - {feature}"
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


def main():

    # ------------------------------------------------
    # Load configuration
    # ------------------------------------------------
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # ------------------------------------------------
    # Load model
    # ------------------------------------------------
    model = joblib.load(
        config["paths"]["model"]
    )

    # ------------------------------------------------
    # Load dataset
    # ------------------------------------------------
    from dataset import load_data

    _, X_test, _, _ = load_data(
        random_state=config["experiment"]["random_state"],
        test_size=config["data"]["test_size"]
    )

    # ------------------------------------------------
    # PDP configuration
    # ------------------------------------------------
    feature = config["pdp"]["feature"]

    points = config["pdp"]["points"]

    lower_percentile = config["pdp"]["lower_percentile"]

    upper_percentile = config["pdp"]["upper_percentile"]

    # ------------------------------------------------
    # Calculate PDP
    # ------------------------------------------------
    R, PD = calculate_pdp(
        model=model,
        X_test=X_test,
        feature=feature,
        points=points,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile
    )

    # ------------------------------------------------
    # Save PDP data
    # ------------------------------------------------
    pdp_dataframe = pd.DataFrame({
        feature: R,
        "partial_dependence": PD
    })

    pdp_dataframe.to_csv(
        config["paths"]["pdp_data"],
        index=False
    )

    # ------------------------------------------------
    # Plot PDP
    # ------------------------------------------------
    plot_pdp(
        R,
        PD,
        feature,
        config["paths"]["pdp_plot"]
    )

    # ------------------------------------------------
    # Print results
    # ------------------------------------------------
    print("\n" + "=" * 60)
    print("PARTIAL DEPENDENCE RESULTS")
    print("=" * 60)

    print(
        pdp_dataframe.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nPDP data saved to: "
        f"{config['paths']['pdp_data']}"
    )

    print(
        f"PDP plot saved to: "
        f"{config['paths']['pdp_plot']}"
    )


if __name__ == "__main__":
    main()