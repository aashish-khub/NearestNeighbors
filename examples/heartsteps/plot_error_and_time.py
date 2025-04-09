"""Script to plot the MSE as a boxplot

Example usage:
```bash
python plot_error.py -od OUTPUT_DIR
```
"""

import os

import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from utils import get_base_parser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir
results_dir = os.path.join(output_dir, "results")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

files = glob(os.path.join(results_dir, "est_errors-*.csv"))
df_list = []
for file in files:
    df = pd.read_csv(file)
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)
# aggregate into list by estimation method and fit method
df_grouped = df.groupby(["estimation_method", "fit_method"]).agg(list).reset_index()

for col_name, alias in [
    ("est_errors", "Absolute error"),
    ("time_impute", "Imputation time"),
    ("time_fit", "Fit time"),
]:
    plt.figure()
    # Create boxplot
    box = plt.boxplot(
        df_grouped[col_name], patch_artist=True, widths=0.6, showfliers=False
    )
    # colors = ['red', 'green', 'orange', 'blue']
    colors = ["red", "blue"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    for median in box["medians"]:
        median.set_color("black")
    # Set y-axis limit
    # plt.ylim(0, 0.4)
    plt.ylim(0, None)
    # Add labels and title
    labels: list[str] = list(df_grouped["estimation_method"])
    plt.xticks(list(range(1, len(labels) + 1)), labels, fontsize=15)
    plt.ylabel(alias, fontsize=15)
    ax1 = plt.gca()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.grid(True, alpha=0.4)
    plt.xlabel(r"Estimation method", fontsize=15)
    save_path = os.path.join(figures_dir, f"{col_name}_boxplot.pdf")
    logger.info(f"Saving plot to {save_path}...")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
