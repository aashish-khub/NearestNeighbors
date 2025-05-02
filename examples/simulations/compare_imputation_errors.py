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
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    
    # Get unique miss_prob and snr values for filename
    miss_prob = df['miss_prob'].iloc[0]
    snr = df['snr'].iloc[0]
    
    # Create side-by-side boxplot for imputation errors
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="size", y="est_errors", hue="estimation_method", data=df)
    plt.title(f"Imputation Error Comparison by Method and Matrix Size\n(miss_prob={miss_prob}, snr={snr})")
    plt.xlabel("Matrix Size (2^n)")
    plt.ylabel("Absolute Error")
    plt.yscale("log")  # Use log scale for better visualization
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"imputation_error_comparison_missprob{miss_prob}_snr{snr}.png"), dpi=300)
    plt.close()
    logger.info("Generated imputation error comparison plot")
    
    # Print summary statistics
    logger.info("Summary statistics for imputation errors:")
    summary_stats = df.groupby(["size", "estimation_method", "miss_prob", "snr"])["est_errors"].describe()
    logger.info(summary_stats)
    # Reset index to add size and estimation_method as columns
    summary_stats = summary_stats.reset_index()
    summary_stats.to_csv(os.path.join(output_dir, f"summary_stats_missprob{miss_prob}_snr{snr}.csv"), index=False)
else:
    logger.error(f"Data file not found: {data_file}") 