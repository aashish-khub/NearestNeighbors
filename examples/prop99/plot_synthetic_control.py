import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import logging

from nearest_neighbors.utils.experiments import get_base_parser
from nearest_neighbors.utils import plotting_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir
california_dir = os.path.join(output_dir, "california")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

files = glob(os.path.join(california_dir, "california-*.csv"))
df_list = []

fig = plt.figure(figsize=(5.5, 3))
ax = fig.add_subplot(111)
for i, file in enumerate(files):
    df = pd.read_csv(file, index_col=0)
    method = plotting_utils.METHOD_ALIASES_SINGLE_LINE.get(
        df.iloc[0]["estimation_method"], df.iloc[0]["estimation_method"]
    )
    line_style = plotting_utils.METHOD_LINE_STYLES.get(
        df.iloc[0]["estimation_method"], "-"
    )
    ax.plot(df.index, df["control"], label=f"Control ({method})", linestyle=line_style)
    if i == 0:
        ax.plot(df.index, df["obs"], label="Observed")
# add a vertical line at 1989 called Proposal 99
ax.axvline(x=1989, color="k", alpha=0.25)
# add a text label at 1989 called Proposal 99
ax.text(1989, ax.get_ylim()[1], "Proposal 99", color="k", fontsize=12)
# set the xtick labels to be the years 1970 to 2000
ax.set_xticks(
    range(1970, 2001, 10),
    [str(year) for year in range(1970, 2001, 10)],
    fontsize=plotting_utils.TICK_FONT_SIZE,
)
ax.set_xlim(1970, 2000)
ax.set_ylim(40, 160)
# set the y-axis label to be the number of cigarettes smoked per capita
ax.set_ylabel(
    "Cigarette Consumption\n(Pack Sales Per Capita)",
    fontsize=plotting_utils.LABEL_FONT_SIZE,
)
# set the x-axis label to be the year
ax.set_xlabel("Year", fontsize=plotting_utils.LABEL_FONT_SIZE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position(
    ("outward", plotting_utils.OUTWARD)
)  # Move x-axis outward
ax.spines["left"].set_position(
    ("outward", plotting_utils.OUTWARD)
)  # Move y-axis outward
ax.legend(fontsize=plotting_utils.LEGEND_FONT_SIZE)
plt.subplots_adjust(
    left=0.15,
    right=0.95,
    # top=0.9,
    # bottom=0.2,
)

# plt.show()
save_path = os.path.join(figures_dir, "california-synthetic-control.pdf")
logger.info(f"Saving figure to {save_path}...")
plt.savefig(save_path)
