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
    default=4.0,
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
# print(os.path.join(results_dir, f"est_errors-*-p{0.3}-tp{tuning_parameter}.csv"))
# files01 = glob(
#     os.path.join(results_dir, f"est_errors-*-p{0.3}-tp{tuning_parameter}.csv")
# )
files05 = glob(
    os.path.join(results_dir, f"est_errors-*-p{0.7}-tp{tuning_parameter}.csv")
)
# files09 = glob(
#     os.path.join(results_dir, f"est_errors-*-p{0.7}-tp{tuning_parameter}.csv")
# )
df_list_01 = []
# for file in files01:
#     df = pd.read_csv(file)
#     # Convert string representations of lists to actual lists
#     df["imputation"] = df["imputation"].apply(lambda x: np.array(eval(x)))
#     df["ground_truth"] = df["ground_truth"].apply(lambda x: np.array(eval(x)))
#     df_list_01.append(df)
df_list_05 = []
for file in files05:
    df = pd.read_csv(file)
    # Convert string representations of lists to actual lists
    df["imputation"] = df["imputation"].apply(lambda x: np.array(eval(x)))
    df["ground_truth"] = df["ground_truth"].apply(lambda x: np.array(eval(x)))
    df_list_05.append(df)
# df_list_09 = []
# for file in files09:
#     df = pd.read_csv(file)
#     # Convert string representations of lists to actual lists
#     df["imputation"] = df["imputation"].apply(lambda x: np.array(eval(x)))
#     df["ground_truth"] = df["ground_truth"].apply(lambda x: np.array(eval(x)))
#     df_list_09.append(df)

# df_01 = pd.concat(df_list_01, ignore_index=True)
df_05 = pd.concat(df_list_05, ignore_index=True)
# df_09 = pd.concat(df_list_09, ignore_index=True)
# each row in the dataframe is a ground truth and imputation pair for a given estimation method, fit method, and data type at row r and column c
# groupby r, c and plot the histograms of the ground truth and imputation for each group
# df_01 = (
#     df_01.groupby(["row", "col"])
#     .apply(
#         lambda x: {
#             "imputations": {
#                 (row["estimation_method"], row["fit_method"], row["data_type"]): row[
#                     "imputation"
#                 ]
#                 for _, row in x.iterrows()
#             },
#             "ground_truth": x["ground_truth"].iloc[
#                 0
#             ],  # Take first ground truth since it should be same for all methods
#         }
#     )
#     .reset_index()
#     .rename(columns={0: "data"})
# )
df_05 = (
    df_05.groupby(["row", "col"])
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
# df_09 = (
#     df_09.groupby(["row", "col"])
#     .apply(
#         lambda x: {
#             "imputations": {
#                 (row["estimation_method"], row["fit_method"], row["data_type"]): row[
#                     "imputation"
#                 ]
#                 for _, row in x.iterrows()
#             },
#             "ground_truth": x["ground_truth"].iloc[
#                 0
#             ],  # Take first ground truth since it should be same for all methods
#         }
#     )
#     .reset_index()
#     .rename(columns={0: "data"})
# )

COLORS = {
    ("row-row", "lbo", "kernel_mmd"): "teal",
    ("col-col", "lbo", "kernel_mmd"): "blue",
    ("row-row", "lbo", "wasserstein_samples"): "green",
    ("col-col", "lbo", "wasserstein_samples"): "orange",
}

# Determine common bins based on all data


# Calculate the min and max across all data to create common bins
bins = list(np.linspace(0, 1, 30))  # 11 points create 10 bins

for i in range(0, 100):
    # row_09 = df_09.iloc[i]
    row_05 = df_05.iloc[i]
    # row_01 = df_01.iloc[i]
    fig, ax2 = plt.subplots(figsize=(3, 2.5))

    # all_data = row_09["data"]["ground_truth"]
    # for imputation in row_09["data"]["imputations"].values():  # type: ignore
    #     all_data.append(imputation)
    # Plot imputation for each method
    # for (est_method, fit_method, data_type), imputation in row_09["data"][  # type: ignore
    #     "imputations"
    # ].items():  # type: ignore
    #     weights = np.ones_like(imputation) / len(imputation)
    #     ax1.hist(
    #         imputation,
    #         bins=bins,
    #         alpha=0.6,
    #         label=f"{plotting_utils.METHOD_ALIASES_SINGLE_LINE.get(est_method, est_method)}"
    #         f"\n({plotting_utils.DATA_TYPE_ALIASES.get(data_type, data_type)})",
    #         weights=weights,
    #         color=COLORS[(est_method, fit_method, data_type)],
    #     )
    # # Plot ground truth
    # weights = np.ones_like(row_09["data"]["ground_truth"]) / len(row_09["data"]["ground_truth"])
    # ax1.hist(
    #     row_09["data"]["ground_truth"],
    #     weights=weights,
    #     bins=bins,
    #     alpha=0.6,
    #     label="Ground\nTruth",
    #     edgecolor="black",
    #     color="white",
    #     linestyle="--",
    # )
    # ax1.set_xlabel("Score", fontsize=plotting_utils.LABEL_FONT_SIZE)
    # ax1.set_ylabel("Density", fontsize=plotting_utils.LABEL_FONT_SIZE)
    # ax1.set_xlim(0, 1)

    # ax1.spines["top"].set_visible(False)
    # ax1.spines["right"].set_visible(False)
    # # ax.spines["bottom"].set_position(
    # #     ("outward", plotting_utils.OUTWARD)
    # # )  # Move x-axis outward
    # ax1.spines["left"].set_position(
    #     ("outward", plotting_utils.OUTWARD)
    # )  # Move y-axis outward

    # # Only add legend to the first subplot to avoid clutter
    # ax1.legend(loc='upper right', fontsize=plotting_utils.LEGEND_FONT_SIZE)

    # all_data = row_05["data"]["ground_truth"]
    # for imputation in row_05["data"]["imputations"].values():  # type: ignore
    #     all_data.append(imputation)
    # Plot imputation for each method
    weights = np.ones_like(row_05["data"]["ground_truth"]) / len(
        row_05["data"]["ground_truth"]
    )
    ax2.hist(
        row_05["data"]["ground_truth"],
        bins=bins,
        weights=weights,
        alpha=0.6,
        label="Ground\nTruth",
        edgecolor="black",
        linestyle="--",
        color="white",
    )
    for (est_method, fit_method, data_type), imputation in row_05["data"][  # type: ignore
        "imputations"
    ].items():  # type: ignore
        if data_type == "wasserstein_samples":
            # Convert to numpy array if it's not already
            est_method_dt = "wasserstein_samples"
        else:
            est_method_dt = "kernel"
        if est_method == "col-col":
            est_method_lbl = "col"
        else:
            est_method_lbl = "row"

        weights = np.ones_like(imputation) / len(imputation)
        ax2.hist(
            imputation,
            bins=bins,
            weights=weights,
            alpha=0.6,
            label=f"{plotting_utils.METHOD_ALIASES_SINGLE_LINE.get(est_method_dt, est_method_dt)}"
            f"\n({est_method_lbl})",
            color=COLORS[(est_method, fit_method, data_type)],
        )
    # Plot ground truth

    ax2.set_ylabel("Proportion", fontsize=plotting_utils.LABEL_FONT_SIZE)
    ax2.set_xlabel("Score", fontsize=plotting_utils.LABEL_FONT_SIZE)
    ax2.legend(loc="upper right", fontsize=plotting_utils.LEGEND_FONT_SIZE)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(True, alpha=0.4)
    # ax.spines["bottom"].set_position(
    #     ("outward", plotting_utils.OUTWARD)
    # )  # Move x-axis outward
    # ax2.spines["left"].set_position(
    #     ("outward", plotting_utils.OUTWARD)
    # )  # Move y-axis outward

    # Only add legend to the first subplot to avoid clutter
    ax2.legend(loc="upper left", fontsize=plotting_utils.LEGEND_FONT_SIZE)

    figures_path = os.path.join(
        figures_dir,
        f"distributions-entry{i}-tp{tuning_parameter}.pdf",
    )
    logger.info(f"Saving figure to {figures_path}")
    plt.savefig(figures_path, dpi=300, bbox_inches="tight")
    plt.close()
