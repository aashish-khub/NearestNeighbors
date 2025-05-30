"""Script to generate a side-by-side boxplot comparing imputation errors
of aw_nn, row_row, and USVT against increasing matrix sizes.

Example usage (from root of repo):
```bash
python src/nearest_neighbors/simulations/compare_imputation_errors.py
```
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from scipy.stats import linregress

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define file path
data_file = "out/results/est_errors-row-row-lbo.csv"

# Create output directory
# output_dir = "out/results"
output_dir = "out/AWNN_vs_row_row"
os.makedirs(output_dir, exist_ok=True)

# Load data
if os.path.exists(data_file):
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)

    # Get unique miss_prob, snr, and model_used values for filename
    model_used = df["latent_factor_combination_model"].iloc[
        0
    ]  # Assuming it's consistent across rows
    miss_prob = df["miss_prob"].iloc[0]
    snr = df["snr"].iloc[0]

    # Create side-by-side boxplot for imputation errors
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="size", y="est_errors", hue="estimation_method", data=df)
    plt.title(
        f"Imputation Error Comparison by Method and Matrix Size\n(model={model_used}, miss_prob={miss_prob}, snr={snr})"
    )
    plt.xlabel("Matrix Size (2^n)")
    plt.ylabel("Absolute Error")
    plt.yscale("log")  # Use log scale for better visualization
    plt.legend(title="Method")
    plt.tight_layout()
    # Save the plot with model description in the filename
    boxplot_filename = f"imputation_error_comparison_model{model_used}_missprob{miss_prob}_snr{snr}.png"
    plt.savefig(os.path.join(output_dir, boxplot_filename), dpi=300)
    plt.close()
    logger.info(f"Generated imputation error comparison plot: {boxplot_filename}")

    # Print summary statistics
    logger.info("Summary statistics for imputation errors:")
    df["est_errors_squared"] = df["est_errors"] ** 2

    # Group by and get summary stats of squared errors
    summary_stats = df.groupby(["size", "estimation_method", "miss_prob", "snr"])[
        "est_errors_squared"
    ].describe()
    logger.info(summary_stats)
    # Reset index to add size and estimation_method as columns
    summary_stats = summary_stats.reset_index()
    summary_stats_filename = (
        f"summary_stats_model{model_used}_missprob{miss_prob}_snr{snr}.csv"
    )
    summary_stats.to_csv(os.path.join(output_dir, summary_stats_filename), index=False)
    logger.info(f"Saved summary statistics: {summary_stats_filename}")

    # Plot mean error with regression lines
    plt.figure(figsize=(14, 8))
    ymins = []
    ymaxs = []
    for idx, method in enumerate(summary_stats["estimation_method"].unique()):
        subset = summary_stats[summary_stats["estimation_method"] == method]
        # Perform linear regression of log(mean) on size
        # Convert pandas Series to lists to avoid type issues
        x_data = subset["size"].tolist()
        y_data = np.log(subset["mean"]).tolist()
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

        label_with_slope = f"{method} (n^{{{slope:.3f}}})"
        color_code = sns.color_palette(
            "husl", len(summary_stats["estimation_method"].unique())
        )[idx]
        # Plot dots
        plt.scatter(
            subset["size"],
            subset["mean"],
            label=label_with_slope,
            color=color_code,
            alpha=0.7,
            s=80,  # Adjust marker size as needed
        )
        # Plot the regression line
        x_vals = np.linspace(subset["size"].min(), subset["size"].max(), 100)
        y_vals = np.exp(intercept + slope * x_vals)  # type: ignore
        plt.plot(x_vals, y_vals, linestyle="--", alpha=0.7, color=color_code)
        # Track y-axis limits
        # Convert pandas Series to lists before extending
        ymins.extend((subset["mean"] - subset["std"]).tolist())
        ymaxs.extend((subset["mean"] + subset["std"]).tolist())

    # Add more ticks to the y-axis
    y_ticks = np.logspace(
        np.floor(np.log10(min(ymins))).astype(int),
        np.ceil(np.log10(max(ymaxs))).astype(int),
        num=10,
    )  # Adjust 'num' for more or fewer ticks
    plt.yticks(y_ticks, [f"{tick:.1e}" for tick in y_ticks])

    plt.title(
        f"Mean Imputation Error with Regression Lines\n(model={model_used}, miss_prob={miss_prob}, snr={snr})"
    )
    plt.xlabel("Matrix Size (2^n)")
    plt.ylabel("Mean Squared Error")
    plt.yscale("log")
    plt.legend(title="Method")
    plt.tight_layout()
    # Save the plot with model description in the filename
    regression_plot_filename = (
        f"mean_error_with_regression_model{model_used}_missprob{miss_prob}_snr{snr}.png"
    )
    plt.savefig(os.path.join(output_dir, regression_plot_filename), dpi=300)
    plt.close()
    logger.info(
        f"Generated mean error plot with regression lines: {regression_plot_filename}"
    )
else:
    logger.error(f"Data file not found: {data_file}")
