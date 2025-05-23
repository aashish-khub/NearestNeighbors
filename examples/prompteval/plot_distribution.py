"""For fixed propensity, overlay histograms of the ground-truth distributions and the imputed distributions
for various estimation methods, fit methods, and data types.

Example usage:
```bash
python plot_distribution.py -od ./output -em row-row -fm lbo --data_type kernel_mmd -p 0.5
```
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from glob import glob

from nsquared.utils.experiments import get_base_parser, setup_logging
from nsquared.utils import plotting_utils

parser = get_base_parser()
parser.add_argument(
    "--data_type",
    type=str,
    default="kernel_mmd",
    choices=["kernel_mmd", "wasserstein_samples"],
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

figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

results_dir = os.path.join(output_dir, "results")
files = glob(
    os.path.join(results_dir, f"est_errors-*-p{propensity}-tp{tuning_parameter}.csv")
)
df_list = []
for file in files:
    df = pd.read_csv(file)
    # Convert string representations of lists to actual lists
    df["imputation"] = df["imputation"].apply(lambda x: np.array(eval(x)))
    df["ground_truth"] = df["ground_truth"].apply(lambda x: np.array(eval(x)))
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)
# each row in the dataframe is a ground truth and imputation pair for a given estimation method, fit method, and data type at row r and column c
# groupby r, c and plot the histograms of the ground truth and imputation for each group
df = (
    df.groupby(["row", "col"])
    .apply(
        lambda x: {
            "imputations": {
                (row["estimation_method"], row["fit_method"], row["data_type"]): row[
                    "imputation"
                ]
                for _, row in x.iterrows()
            },
            "ground_truth": x["ground_truth"].iloc[
                0
            ],  # Take first ground truth since it should be same for all methods
        }
    )
    .reset_index()
    .rename(columns={0: "data"})
)

COLORS = {
    ("row-row", "lbo", "kernel_mmd"): "teal",
    ("col-col", "lbo", "kernel_mmd"): "blue",
}

for i, row in df.head(20).iterrows():
    fig, ax = plt.subplots(figsize=(5.5 / 2, 5.5 / 2))
    # Determine common bins based on all data
    all_data = [row["data"]["ground_truth"]]
    for imputation in row["data"]["imputations"].values():  # type: ignore
        all_data.append(imputation)

    # Calculate the min and max across all data to create common bins
    bins = list(np.linspace(0, 1, 40))  # 11 points create 10 bins

    # Plot imputation for each method
    for (est_method, fit_method, data_type), imputation in row["data"][  # type: ignore
        "imputations"
    ].items():  # type: ignore
        ax.hist(
            imputation,
            bins=bins,
            alpha=0.5,
            label=f"{plotting_utils.METHOD_ALIASES_SINGLE_LINE.get(est_method, est_method)}"
            f"\n({plotting_utils.DATA_TYPE_ALIASES.get(data_type, data_type)})",
            edgecolor="black",
            density=True,
            color=COLORS[(est_method, fit_method, data_type)],
        )
    # Plot ground truth
    ax.hist(
        row["data"]["ground_truth"],
        bins=bins,
        alpha=0.5,
        label="Ground Truth",
        edgecolor="black",
        density=True,
        color="yellow",
    )
    ax.set_xlabel("Score", fontsize=plotting_utils.LABEL_FONT_SIZE)
    ax.set_ylabel("Density", fontsize=plotting_utils.LABEL_FONT_SIZE)
    ax.set_xlim(0, 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_position(
    #     ("outward", plotting_utils.OUTWARD)
    # )  # Move x-axis outward
    ax.spines["left"].set_position(
        ("outward", plotting_utils.OUTWARD)
    )  # Move y-axis outward

    # Only add legend to the first subplot to avoid clutter
    ax.legend()

    figures_path = os.path.join(
        figures_dir,
        f"distributions-r{row.row}-c{row.col}-p{propensity}-tp{tuning_parameter}.pdf",
    )
    logger.info(f"Saving figure to {figures_path}")
    plt.savefig(figures_path, dpi=300, bbox_inches="tight")
