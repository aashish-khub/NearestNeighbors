"""Script to plot the MSE as a boxplot

NOTE: imputation time is per imputation, fit time is for the entire fitting procedure
TODO: change fit time to be a bar plot

Example usage:
```bash
python plot_error_and_time.py -od OUTPUT_DIR
```
"""

import os

import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import logging

from nearest_neighbors.utils.experiments import get_base_parser
from nearest_neighbors.utils import plotting_utils

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
# NOTE: filter out NaN values when aggregating
df_grouped = (
    df.groupby(["estimation_method", "fit_method"])
    .agg(lambda x: list([val for val in x if pd.notna(val)]))
    .reset_index()
)
# rearrange the order of the estimation methods by
# "usvt", "row-row", "col-col", "dr", "ts", "star"
ORDER = ["usvt", "col-col", "row-row", "dr", "ts", "star"]
df_grouped = df_grouped.sort_values(
    by="estimation_method", key=lambda x: x.map(lambda y: ORDER.index(y))
)

for col_name, alias in [
    ("est_errors", "Absolute error"),
    ("time_impute", "Imputation time"),
    ("time_fit", "Fit time"),
]:
    # NOTE: set the width to be the physical size of the figure in inches
    # The NeurIPS text is 5.5 inches wide and 9 inches long
    # If we use wrapfigure with 0.4\textwidth, then the figure needs to be 2.2 inches wide
    fig = plt.figure(figsize=(2.2, 2))
    # Create boxplot
    ax = fig.add_subplot(111)
    box = ax.boxplot(
        df_grouped[col_name], patch_artist=True, widths=0.6, showfliers=False
    )
    colors = ["orange"] * len(df_grouped)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    for median in box["medians"]:
        median.set_color("black")
    # Set y-axis limit
    # plt.ylim(0, 0.4)
    ax.set_ylim(0, None)
    # Add labels and title
    labels: list[str] = [
        str(plotting_utils.METHOD_ALIASES.get(method, method) or "")
        for method in df_grouped["estimation_method"]
    ]
    ax.set_xticks(
        list(range(1, len(labels) + 1)), labels, fontsize=plotting_utils.TICK_FONT_SIZE
    )
    ax.tick_params(axis="x", length=0)  # Set tick length to 0
    ax.set_ylabel(alias, fontsize=plotting_utils.LABEL_FONT_SIZE)
    # ax.set_xlabel("Estimation method", fontsize=plotting_utils.LABEL_FONT_SIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(True, alpha=0.4)

    save_path = os.path.join(figures_dir, f"hs_{col_name}_boxplot.pdf")
    logger.info(f"Saving plot to {save_path}...")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
