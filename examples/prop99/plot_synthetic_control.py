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

figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

synthetic_control_dir = os.path.join(output_dir, "sc")
# get the subdirectories in the synthetic control directory
subdirs = glob(os.path.join(synthetic_control_dir, "*"))

for subdir in subdirs:
    state = os.path.basename(subdir)
    if state != "CA":
        continue
    files = glob(os.path.join(subdir, f"sc-{state}-*.csv"))

    fig = plt.figure(figsize=(plotting_utils.NEURIPS_TEXTWIDTH / 2, 2.5))
    ax = fig.add_subplot(111)
    for i, file in enumerate(files):
        df = pd.read_csv(file, index_col=0)
        estimation_method = df.iloc[0]["estimation_method"]
        method = plotting_utils.METHOD_ALIASES_SINGLE_LINE.get(
            estimation_method, estimation_method
        )
        ax.plot(
            df.index,
            df["control"],
            label=f"Control ({method})",
            linestyle="-",  # plotting_utils.METHOD_LINE_STYLES.get(estimation_method, "-"),
            color=plotting_utils.COLORS[estimation_method],
            alpha=0.75,
        )
        # add a text label at the end of the line segment
        # Get the last non-nan value for the control
        last_valid_value = df["control"].dropna().iloc[-1]
        # NOTE: va is opposite of what you'd expect (set top to nudge down, bottom to nudge up)
        match estimation_method:
            case "dr":
                va = "bottom"
            # case "col-col":
            #     va = "bottom"
            case "row-row":
                va = "top"
            case _:
                va = "center"
        ax.text(
            2001,
            last_valid_value,
            method,
            fontsize=plotting_utils.TICK_FONT_SIZE,
            va=va,
        )
        if i == 0:
            ax.plot(
                df.index, df["obs"], label="Observed", linestyle="-", color="orange"
            )
            ax.text(
                2001, df["obs"].iloc[-1], "Obs.", fontsize=plotting_utils.TICK_FONT_SIZE
            )
    # add a vertical line at 1989 called Proposition 99
    ax.axvline(x=1989, color="k", alpha=0.25, linestyle="dotted")
    # add a text label at 1989 called Proposition 99
    ax.text(
        1989,
        ax.get_ylim()[1],
        "Proposition 99",
        color="k",
        fontsize=plotting_utils.LABEL_FONT_SIZE,
        ha="center",
    )

    # set the major xtick labels to be the years 1970 to 2000
    ax.set_xticks(
        range(1970, 2001, 10),
        [str(year) for year in range(1970, 2001, 10)],
        fontsize=plotting_utils.TICK_FONT_SIZE,
    )
    # set the minor xtick labels to be every year between 1970 and 2000
    ax.set_xticks(range(1970, 2001, 1), minor=True)
    ax.set_xlim(1970, 2000)
    # set the x-axis label to be the year
    ax.set_xlabel("Year", fontsize=plotting_utils.LABEL_FONT_SIZE)

    # ax.set_ylim(40, 160)
    # set the y-axis label to be the number of cigarettes smoked per capita
    ax.set_ylabel(
        "Cigarette Consumption\n(Pack Sales Per Capita)",
        fontsize=plotting_utils.LABEL_FONT_SIZE,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(
        ("outward", plotting_utils.OUTWARD)
    )  # Move x-axis outward
    ax.spines["left"].set_position(
        ("outward", plotting_utils.OUTWARD)
    )  # Move y-axis outward
    # ax.legend(fontsize=plotting_utils.LEGEND_FONT_SIZE)
    plt.subplots_adjust(
        left=0.15,
        right=0.85,
        top=0.95,
        bottom=0.2,
    )

    # plt.show()
    save_path = os.path.join(figures_dir, f"{state}-synthetic-control.pdf")
    logger.info(f"Saving figure to {save_path}...")
    plt.savefig(save_path)
