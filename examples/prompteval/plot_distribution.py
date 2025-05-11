"""Plot the first 12 histograms of imputed and ground truth distributions
for a given estimation method, fit method, data type, and propensity.

Example usage:
```bash
python plot_distribution.py -od ./output -em row-row -fm lbo --data_type kernel_mmd -p 0.5
```
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nearest_neighbors.utils.experiments import get_base_parser, setup_logging
import logging

parser = get_base_parser()
parser.add_argument(
    "--data_type",
    type=str,
    default="kernel_mmd",
    choices=["kernel_mmd", "wasserstein_quantile"],
)
parser.add_argument(
    "--tuning_parameter",
    "-tp",
    type=float,
    default=0.5,
)
args = parser.parse_args()
output_dir = args.output_dir
propensity = args.propensity
estimation_method = args.estimation_method
fit_method = args.fit_method
data_type = args.data_type
tuning_parameter = args.tuning_parameter

setup_logging(args.log_level)
logger = logging.getLogger(__name__)

results_dir = os.path.join(output_dir, "results")
results_path = os.path.join(
    results_dir,
    f"est_errors-{estimation_method}-{fit_method}-{data_type}-p{propensity}-tp{tuning_parameter}.csv",
)
df = pd.read_csv(results_path)
# Convert string representations of lists to actual lists
df["imputation"] = df["imputation"].apply(lambda x: np.array(eval(x)))
df["ground_truth"] = df["ground_truth"].apply(lambda x: np.array(eval(x)))

# Create a figure with a 4x3 grid of subplots
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

# Plot histograms for each entry
for i, row in df.head(12).iterrows():
    ax = axes[i]
    ax.hist(
        row["imputation"],
        bins=20,
        alpha=0.5,
        label="Imputed",
        edgecolor="black",
        density=True,
    )
    ax.hist(
        row["ground_truth"],
        bins=20,
        alpha=0.5,
        label="Ground Truth",
        edgecolor="black",
        density=True,
    )
    ax.set_title(f"r: {row.row}, c: {row.col}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    # Only add legend to the first subplot to avoid clutter
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.suptitle(f"p={propensity}, data type: {data_type}", y=1.02, fontsize=16)
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
figures_path = os.path.join(
    figures_dir,
    f"distributions-{estimation_method}-{fit_method}-{data_type}-p{propensity}-tp{tuning_parameter}.pdf",
)
logger.info(f"Saving figure to {figures_path}")
plt.savefig(figures_path, dpi=300, bbox_inches="tight")
