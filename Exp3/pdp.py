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

        # Predict probability of class 1
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


def calculate_sensitivity(R, PD):
    """
    Calculate numerical derivative of the PDP.

    Sensitivity = d(PD) / d(feature)

    np.gradient() estimates the derivative at
    every point in R.
    """

    sensitivity = np.gradient(
        PD,
        R
    )

    return sensitivity


def plot_pdp(
    R,
    PD,
    feature,
    path
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        R,
        PD,
        linewidth=2
    )

    plt.xlabel(feature)

    plt.ylabel(
        "Average Predicted Probability"
    )

    plt.title(
        f"Partial Dependence Plot - {feature}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


def plot_sensitivity(
    R,
    sensitivity,
    feature,
    path
):

    plt.figure(figsize=(10, 6))

    plt.plot(
        R,
        sensitivity,
        linewidth=2
    )

    # Zero line helps identify positive/negative
    # sensitivity regions.
    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(feature)

    plt.ylabel(
        "Sensitivity (dP/dFeature)"
    )

    plt.title(
        f"PDP Sensitivity - {feature}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


def print_sensitivity_summary(
    R,
    sensitivity,
    feature
):

    # ------------------------------------------------
    # Find strongest positive sensitivity
    # ------------------------------------------------
    max_index = np.argmax(
        sensitivity
    )

    max_sensitivity = sensitivity[
        max_index
    ]

    max_feature_value = R[
        max_index
    ]

    # ------------------------------------------------
    # Find strongest negative sensitivity
    # ------------------------------------------------
    min_index = np.argmin(
        sensitivity
    )

    min_sensitivity = sensitivity[
        min_index
    ]

    min_feature_value = R[
        min_index
    ]

    # ------------------------------------------------
    # Overall/secant sensitivity
    # ------------------------------------------------
    overall_sensitivity = (
        (sensitivity * 0)  # keeps this calculation independent
    )

    # Overall change in PDP / overall change in feature
    # This is the slope between the first and last PDP points.
    # The actual values are calculated below in main.
    
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    print(
        f"Feature: {feature}"
    )

    print(
        f"\nMaximum positive sensitivity:"
    )

    print(
        f"  Feature value : {max_feature_value:.6f}"
    )

    print(
        f"  Sensitivity   : {max_sensitivity:.6f}"
    )

    print(
        f"\nMaximum negative sensitivity:"
    )

    print(
        f"  Feature value : {min_feature_value:.6f}"
    )

    print(
        f"  Sensitivity   : {min_sensitivity:.6f}"
    )

    print(
        "\nInterpretation:"
    )

    if max_sensitivity > 0:
        print(
            f"  Around feature value "
            f"{max_feature_value:.4f}, increasing the "
            f"feature increases the model's predicted "
            f"probability."
        )

    if min_sensitivity < 0:
        print(
            f"  Around feature value "
            f"{min_feature_value:.4f}, increasing the "
            f"feature decreases the model's predicted "
            f"probability."
        )


def main():

    # ------------------------------------------------
    # Load configuration
    # ------------------------------------------------
    with open(
        "config.yaml",
        "r"
    ) as file:

        config = yaml.safe_load(
            file
        )

    # ------------------------------------------------
    # Load trained model
    # ------------------------------------------------
    model = joblib.load(
        config["paths"]["model"]
    )

    print(
        "\nModel loaded successfully."
    )

    # ------------------------------------------------
    # Load dataset
    # ------------------------------------------------
    from dataset import load_data

    _, X_test, _, _ = load_data(
        random_state=config[
            "experiment"
        ]["random_state"],

        test_size=config[
            "data"
        ]["test_size"]
    )

    # ------------------------------------------------
    # PDP configuration
    # ------------------------------------------------
    feature = config[
        "pdp"
    ]["feature"]

    points = config[
        "pdp"
    ]["points"]

    lower_percentile = config[
        "pdp"
    ]["lower_percentile"]

    upper_percentile = config[
        "pdp"
    ]["upper_percentile"]

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
    # Calculate sensitivity
    # ------------------------------------------------
    sensitivity = calculate_sensitivity(
        R,
        PD
    )

    # ------------------------------------------------
    # Calculate overall/secant sensitivity
    # ------------------------------------------------
    overall_sensitivity = (
        (PD[-1] - PD[0])
        /
        (R[-1] - R[0])
    )

    # ------------------------------------------------
    # Create results dataframe
    # ------------------------------------------------
    pdp_dataframe = pd.DataFrame({
        feature: R,

        "partial_dependence": PD,

        "sensitivity": sensitivity
    })

    # ------------------------------------------------
    # Save PDP + sensitivity data
    # ------------------------------------------------
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
    # Plot sensitivity
    # ------------------------------------------------
    plot_sensitivity(
        R,
        sensitivity,
        feature,
        "plots/pdp_sensitivity.png"
    )

    # ------------------------------------------------
    # Print PDP results
    # ------------------------------------------------
    print("\n" + "=" * 60)
    print("PARTIAL DEPENDENCE RESULTS")
    print("=" * 60)

    print(
        pdp_dataframe.head(
            10
        ).to_string(
            index=False
        )
    )

    # ------------------------------------------------
    # Print sensitivity summary
    # ------------------------------------------------
    print_sensitivity_summary(
        R,
        sensitivity,
        feature
    )

    # ------------------------------------------------
    # Print overall sensitivity
    # ------------------------------------------------
    print(
        "\nOverall / Secant Sensitivity:"
    )

    print(
        f"  {overall_sensitivity:.6f}"
    )

    print(
        "\nThis represents the average change "
        "in predicted probability per unit change "
        "in the feature across the entire PDP range."
    )

    # ------------------------------------------------
    # Print output locations
    # ------------------------------------------------
    print(
        f"\nPDP + sensitivity data saved to: "
        f"{config['paths']['pdp_data']}"
    )

    print(
        f"PDP plot saved to: "
        f"{config['paths']['pdp_plot']}"
    )

    print(
        "Sensitivity plot saved to: "
        "plots/pdp_sensitivity.png"
    )


if __name__ == "__main__":
    main()