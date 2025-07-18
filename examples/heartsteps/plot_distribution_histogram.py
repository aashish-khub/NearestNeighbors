"""Script to plot the distribution of step counts for different estimation methods

NOTE: imputation time is per imputation, fit time is for the entire fitting procedure
TODO: change fit time to be a bar plot

Example usage from the examples/heartsteps directory:
```bash
python plot_distribution_histogram.py -od OUTPUT_DIR
```
"""

import os

import matplotlib.pyplot as plt
import logging
import numpy as np

from nsquared.utils.experiments import get_base_parser
from nsquared.utils import plotting_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir
results_dir = os.path.join(output_dir, "results")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

wasserstein_ests = np.load(
    os.path.join(results_dir, "imputations-col-col-lbo-wasserstein_samples.npy"),
    allow_pickle=True,
)
kernel_ests = np.load(
    os.path.join(results_dir, "imputations-col-col-lbo-kernel.npy"), allow_pickle=True
)
ground_truth = np.load(
    os.path.join(results_dir, "ground_truth-col-col-lbo.npy"), allow_pickle=True
)

# NOTE: set the width to be the physical size of the figure in inches
# The NeurIPS text is 5.5 inches wide and 9 inches long
# If we use wrapfigure with 0.4\textwidth, then the figure needs to be 2.2 inches wide
for i in range(0, len(ground_truth)):
    fig = plt.figure(figsize=(2.2, 2))
    # Create boxplot
    ax = fig.add_subplot(111)
    # box = ax.boxplot(
    #     df_grouped[col_name], patch_artist=True, widths=0.6, showfliers=False
    # )
    bins = list(np.linspace(0, 8, 12))
    weights_gt = np.ones_like(ground_truth[i]) / len(ground_truth[i])
    ax.hist(
        ground_truth[i],
        bins=bins,
        weights=weights_gt,
        alpha=0.6,
        label="Ground truth",
        color="white",
        edgecolor="black",
        linestyle="--",
    )
    weights_kernel = np.ones_like(kernel_ests[i]) / len(kernel_ests[i])
    ax.hist(
        kernel_ests[i],
        bins=bins,
        weights=weights_kernel,
        alpha=0.6,
        label=str(plotting_utils.METHOD_ALIASES_SINGLE_LINE.get("kernel", "kernel")),
        color="teal",
    )
    weights_wasserstein = np.ones_like(wasserstein_ests[i]) / len(wasserstein_ests[i])
    ax.hist(
        wasserstein_ests[i],
        bins=bins,
        weights=weights_wasserstein,
        alpha=0.6,
        label=str(
            plotting_utils.METHOD_ALIASES_SINGLE_LINE.get("wasserstein_samples", "W2S")
        ),
        color="orange",
    )

    ax.set_ylim(0, None)

    ax.set_ylabel("Proportion", fontsize=plotting_utils.LABEL_FONT_SIZE)
    ax.set_xlabel("Step count", fontsize=plotting_utils.LABEL_FONT_SIZE)
    ax.legend(loc="upper right", fontsize=plotting_utils.LEGEND_FONT_SIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(True, alpha=0.4)

    save_path = os.path.join(figures_dir, f"hs_distplot_{i}.pdf")
    logger.info(f"Saving plot to {save_path}...")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
