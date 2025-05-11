"""Script to plot error of the synthetic control vs the observed value for control states.

Example usage:
```bash
python plot_sc_error.py -od OUTPUT_DIR
```
"""

import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import logging

from nearest_neighbors.utils.experiments import get_base_parser
from nearest_neighbors.utils import plotting_utils
from nearest_neighbors.datasets.prop99.loader import Prop99DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir

figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# get the subdirectories in the synthetic control directory
sc_dir = os.path.join(output_dir, "sc", "*", "sc-*.csv")
files = glob(sc_dir)
logger.info(f"Found {len(files)} files in {sc_dir}")
df_list = []
for file in files:
    df = pd.read_csv(file, index_col=0)
    df_list.append(df)
df = pd.concat(df_list)
df = df.reset_index(names=["year"])
logger.info("Only keeping years in the post-intervention period (after 1988)")
df = df[df["year"] > 1988]
# only include control states
df = df[df["state"].isin(Prop99DataLoader.CONTROL_STATES)]  # type: ignore

# aggregate into list by estimation method and fit method
df_grouped = (
    df.groupby(["estimation_method", "fit_method"])  # type: ignore
    .agg(lambda x: list([val for val in x if pd.notna(val)]))  # type: ignore
    .reset_index()  # type: ignore
)

# rearrange the order of the estimation methods by
# "usvt", "row-row", "col-col", "dr", "ts", "star"
ORDER = ["usvt", "col-col", "row-row", "dr", "ts", "star", "sc"]
df_grouped = df_grouped.sort_values(
    by="estimation_method", key=lambda x: x.map(lambda y: ORDER.index(y))
)

for col_name, alias in [
    ("est_errors", "Absolute error"),
]:
    # NOTE: set the width to be the physical size of the figure in inches
    # The NeurIPS text is 5.5 inches wide and 9 inches long
    # If we use wrapfigure with 0.4\textwidth, then the figure needs to be 2.2 inches wide
    fig = plt.figure(figsize=(plotting_utils.NEURIPS_TEXTWIDTH / 2, 2.5))
    # Create boxplot
    ax = fig.add_subplot(111)
    box = ax.boxplot(
        df_grouped[col_name], patch_artist=True, widths=0.6, showfliers=False
    )
    colors = [
        plotting_utils.COLORS[method] for method in df_grouped["estimation_method"]
    ]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    for median in box["medians"]:
        median.set_color("black")
    # Set y-axis limit
    ax.set_ylim(0, None)
    # Add labels and title
    labels: list[str] = [
        plotting_utils.METHOD_ALIASES.get(method, method)  # type: ignore
        for method in df_grouped["estimation_method"]  # type: ignore
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

    save_path = os.path.join(figures_dir, f"sc_{col_name}_boxplot.pdf")
    logger.info(f"Saving plot to {save_path}...")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
