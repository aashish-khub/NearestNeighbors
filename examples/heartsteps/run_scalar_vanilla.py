"""Run the vanilla NN imputer on the heartsteps dataset
using the last 8 participants and 25 timesteps as a test block.

Example usage:
```bash
python run_scalar_vanilla.py -od OUTPUT_DIR
```
"""

import os
from typing import Tuple
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from joblib import Memory

from nearest_neighbors.data_types import Scalar
from nearest_neighbors.datasets.dataloader_factory import NNData
from nearest_neighbors.vanilla_nn import row_row
from nearest_neighbors.fit_methods import LeaveBlockOutValidation
from baselines import usvt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = ArgumentParser()
parser.add_argument(
    "--output_dir", "-od", type=str, default="out", help="Output directory"
)
args = parser.parse_args()
output_dir = args.output_dir

# Create a memory cache in a '.joblib_cache' directory
memory = Memory(".joblib_cache", verbose=1)


@memory.cache
def get_heartsteps_data() -> Tuple[np.ndarray, np.ndarray]:
    """Get the heartsteps dataset."""
    hs_dataloader = NNData.create("heartsteps")
    data, mask = hs_dataloader.process_data_scalar()
    return data, mask


# Replace the original line with:
data, mask = get_heartsteps_data()
data = data[:, :200]  # only use the first 200 timesteps
mask = mask[:, :200]

# Load NNImputer
imputer = row_row()

holdout_inds = np.nonzero(mask == 1)
# test_inds = np.nonzero(mask == 0)
inds_rows = holdout_inds[0]
inds_cols = holdout_inds[1]

# test_inds_rows = test_inds[0]
# test_inds_cols = test_inds[1]

full_mask_test_inds = np.nonzero((inds_rows > 21) & (inds_cols > 159))
test_inds_rows = tuple(inds_rows[full_mask_test_inds])
test_inds_cols = tuple(inds_cols[full_mask_test_inds])

full_mask_inds = np.nonzero((inds_rows < 9) & (inds_cols < 25))
holdout_inds_rows = list(inds_rows[full_mask_inds])
holdout_inds_cols = list(inds_cols[full_mask_inds])

block = list(zip(holdout_inds_rows, holdout_inds_cols))
test_block = list(zip(test_inds_rows, test_inds_cols))

print("Fitting imputer using leave-block-out validation...")
fit_method = LeaveBlockOutValidation(
    block,
    distance_threshold_range=(0, 4_000_000),
    n_trials=200,
    data_type=Scalar(),
)
best_distance_threshold = fit_method.fit(data, mask, imputer)
print(f"Best distance threshold: {best_distance_threshold}")

# Impute missing values
imputations = []
for row, col in tqdm(test_block, desc="Imputing missing values"):
    imputed_value = imputer.impute(row, col, data, mask)
    imputations.append(imputed_value)

ground_truth = data[test_inds_rows, test_inds_cols]
imputations = np.array(imputations)

vanilla_nn_errs = np.abs(imputations - ground_truth)

# setup usvt imputation
usvt_data = data.copy()
usvt_mask = mask.copy()
usvt_mask[test_inds_rows, test_inds_cols] = 0
usvt_data[mask != 1] = np.nan
usvt_imputed = usvt(usvt_data)
usvt_errs = list(np.abs(usvt_imputed[test_inds_rows, test_inds_cols] - ground_truth))

print(f"Vanilla-NN mean absolute error: {np.mean(vanilla_nn_errs)}")
print(f"USVT mean absolute error: {np.mean(usvt_errs)}")
# Plot the error boxplot
# usvt_data = []
vanilla_nn_data = vanilla_nn_errs

data = [usvt_errs, vanilla_nn_data]

# Set up the plot
plt.figure()

# Create boxplot
box = plt.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)

# Customizing colors for each boxplot
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
# plt.xticks([1, 2, 3, 4], ['USVT', 'User-NN', 'Time-NN', 'DR-NN'], fontsize=15)
plt.xticks([1, 2], ["USVT", "Vanilla-NN"], fontsize=15)
plt.ylabel(r"Absolute error", fontsize=15)
# plt.title(r'Variation of error across users  $(N = T = 128, \; 1 \; \text{Trial})$', fontsize=20)

ax1 = plt.gca()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.grid(True, alpha=0.4)

plt.xlabel(r"", fontsize=15)

# Customize grid lines
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Show the plot
save_path = os.path.join(output_dir, "vanilla_nn_error_boxplot.pdf")
print(f"Saving plot to {save_path}")
plt.savefig(save_path, bbox_inches="tight")
