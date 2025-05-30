"""Script to plot the distributional estimation error as a boxplot

NOTE: we use the Kolmogorov-Smirnov distance to measure the error between the
imputed and ground truth distributions.

Example usage:
```bash
python plot_dist_error.py -od OUTPUT_DIR
```
"""

import os

import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import logging
from tabulate import tabulate
import numpy as np

from nsquared.utils.experiments import get_base_parser
from nsquared.utils import plotting_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir
propensity = args.propensity

results_dir = os.path.join(output_dir, "results")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

files = glob(os.path.join(results_dir, f"est_errors-*-p{propensity}-*.csv"))
df_list = []
for file in files:
    df = pd.read_csv(file)
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)
# aggregate into list by estimation method and fit method
# NOTE: filter out NaN values when aggregating
df_grouped = (
    df.groupby(["estimation_method", "fit_method", "data_type"])
    .agg(lambda x: list([val for val in x if pd.notna(val)]))
    .reset_index()
)
# rearrange the order of the estimation methods by
# "usvt", "row-row", "col-col", "dr", "ts", "aw"
ORDER = ["usvt", "col-col", "row-row", "dr", "ts", "aw"]
df_grouped = df_grouped.sort_values(
    by="estimation_method", key=lambda x: x.map(lambda y: ORDER.index(y))
)


def mean_std_error(x: list[float]) -> str:
    """Format the mean and standard deviation of a list of floats as a string

    Args:
        x: list of floats
    Returns:
        str: formatted string

    """
    return f"${np.mean(x):.4f} \\pm {np.std(x, ddof=1) / np.sqrt(len(x)):.4f}$"


df_mean_std = (
    df.groupby(["estimation_method", "fit_method", "data_type"])["est_errors"]
    .agg(mean_std_error)
    .reset_index()
)
# drop the fit method column
df_mean_std = df_mean_std.drop(columns=["fit_method"])
print(tabulate(df_mean_std, headers="keys", tablefmt="github", showindex=False))

for col_name, alias in [
    ("est_errors", "Kolmogorov-Smirnov distance"),
    ("time_impute", "Imputation time"),
    ("time_fit", "Fit time"),
]:
    # NOTE: set the width to be the physical size of the figure in inches
    # The NeurIPS text is 5.5 inches wide and 9 inches long
    fig = plt.figure(figsize=(5.5 / 2, 2.5))
    # Create boxplot
    ax = fig.add_subplot(111)
    box = ax.boxplot(
        df_grouped[col_name], patch_artist=True, widths=0.6, showfliers=True
    )
    # make fliers smaller
    for flier in box["fliers"]:
        # flier.set_marker("o")
        # flier.set_markerfacecolor("black")
        # flier.set_markeredgecolor("black")
        flier.set_markersize(2)
    colors = [
        plotting_utils.COLORS[method] for method in df_grouped["estimation_method"]
    ]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    for median in box["medians"]:
        median.set_color("black")
    # Set y-axis limit
    # plt.ylim(0, 0.4)
    ax.set_ylim(0, None)
    # Add labels and title
    labels: list[str] = [
        f"{plotting_utils.METHOD_ALIASES.get(item.estimation_method, item.estimation_method)}"  # type: ignore
        f"\n({plotting_utils.DATA_TYPE_ALIASES.get(item.data_type, item.data_type)})"  # type: ignore
        for item in df_grouped.itertuples(index=False)
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

    save_path = os.path.join(
        figures_dir, f"prompteval_{col_name}_boxplot-p{propensity}.pdf"
    )
    logger.info(f"Saving plot to {save_path}...")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
