"""Script to generate a side-by-side boxplot comparing imputation errors
of star_nn, row_row, and USVT against increasing matrix sizes.

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

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define file path
data_file = "out/results/est_errors-row-row-lbo.csv"

# Create output directory
output_dir = "out/results"
os.makedirs(output_dir, exist_ok=True)

# Load data
if os.path.exists(data_file):
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)

    # Create side-by-side boxplot for imputation errors
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="size", y="est_errors", hue="estimation_method", data=df)
    plt.title("Imputation Error Comparison by Method and Matrix Size")
    plt.xlabel("Matrix Size (2^n)")
    plt.ylabel("Absolute Error")
    plt.yscale("log")  # Use log scale for better visualization
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "imputation_error_comparison.png"), dpi=300)
    plt.close()
    logger.info("Generated imputation error comparison plot")

    # Create running time comparison between row_row and star_nn
    # time_df = df[df["estimation_method"].isin(["row-row", "star_nn"])]
    # if not time_df.empty:
    #     plt.figure(figsize=(14, 8))
    #     sns.boxplot(x="size", y="time_impute", hue="estimation_method", data=time_df)
    #     plt.title("Imputation Time Comparison: Row-Row vs Star NN")
    #     plt.xlabel("Matrix Size (2^n)")
    #     plt.ylabel("Imputation Time (seconds)")
    #     plt.yscale("log")  # Use log scale for better visualization
    #     plt.legend(title="Method")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "running_time_comparison.png"), dpi=300)
    #     plt.close()
    #     logger.info("Generated running time comparison plot")

    # Print summary statistics
    logger.info("Summary statistics for imputation errors:")
    summary_stats = df.groupby(["size", "estimation_method"])["est_errors"].describe()
    logger.info(summary_stats)
    # Reset index to add size and estimation_method as columns
    summary_stats = summary_stats.reset_index()
    summary_stats.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)
    # logger.info("Summary statistics for imputation times:")
    # logger.info(df.groupby(["size", "estimation_method"])["time_impute"].describe())
else:
    logger.error(f"Data file not found: {data_file}")
