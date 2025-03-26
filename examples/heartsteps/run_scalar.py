"""Script to run NN imputers + USVT baseline on the heartsteps dataset
using the last 8 participants and 25 timesteps as a test block.

Example usage (from root of repo):
```bash
python examples/heartsteps/run_scalar.py -od OUTPUT_DIR -em ESTIMATION_METHOD -fm FIT_METHOD
```
"""

import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from baselines import usvt
from joblib import Memory
from argparse import ArgumentParser
from typing import Tuple
import os
from time import time

from nearest_neighbors.data_types import Scalar
from nearest_neighbors.estimation_methods import DREstimator
from nearest_neighbors import NearestNeighborImputer
from nearest_neighbors.fit_methods import (
    DRLeaveBlockOutValidation,
    LeaveBlockOutValidation,
)
from nearest_neighbors.datasets.dataloader_factory import NNData
from nearest_neighbors.vanilla_nn import row_row

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = ArgumentParser()
parser.add_argument(
    "--output_dir", "-od", type=str, default="out", help="Output directory"
)
parser.add_argument(
    "--estimation_method",
    "-em",
    type=str,
    default="vanilla",
    choices=["dr", "vanilla"],
    help="Estimation method to use",
)
parser.add_argument(
    "--fit_method",
    "-fm",
    type=str,
    default="lbo",
    choices=["dr", "lbo"],
    help="Fit method to use",
)
args = parser.parse_args()
output_dir = args.output_dir
estimation_method = args.estimation_method
fit_method = args.fit_method

os.makedirs(output_dir, exist_ok=True)

# Create a memory cache in a '.joblib_cache' directory
memory = Memory(".joblib_cache", verbose=1)


@memory.cache
def get_heartsteps_data() -> Tuple[np.ndarray, np.ndarray]:
    """Get the heartsteps dataset."""
    hs_dataloader = NNData.create("heartsteps")
    data, mask = hs_dataloader.process_data_scalar()
    return data, mask


# Load the heartsteps dataset
data, mask = get_heartsteps_data()
data = data[:, :200]  # only use the first 200 timesteps
mask = mask[:, :200]

print("Using scalar data type")
data_type = Scalar()

if estimation_method == "dr":
    print("Using doubly robust estimation")
    estimator = DREstimator()
    imputer = NearestNeighborImputer(estimator, data_type)
elif estimation_method == "vanilla":
    print("Using row-row estimation")
    imputer = row_row()
else:
    raise ValueError(f"Estimation method {estimation_method} not supported")

holdout_inds = np.nonzero(mask == 1)
inds_rows = holdout_inds[0]
inds_cols = holdout_inds[1]

full_mask_test_inds = np.nonzero((inds_rows > 21) & (inds_cols > 159))
test_inds_rows = tuple(inds_rows[full_mask_test_inds])
test_inds_cols = tuple(inds_cols[full_mask_test_inds])

full_mask_inds = np.nonzero((inds_rows < 9) & (inds_cols < 25))
holdout_inds_rows = list(inds_rows[full_mask_inds])
holdout_inds_cols = list(inds_cols[full_mask_inds])

block = list(zip(holdout_inds_rows, holdout_inds_cols))
test_block = list(zip(test_inds_rows, test_inds_cols))

if fit_method == "dr":
    print("Using doubly robust fit method")
    # Fit the imputer using leave-block-out validation
    fitter = DRLeaveBlockOutValidation(
        block,
        distance_threshold_range_row=(0, 4000000),
        distance_threshold_range_col=(0, 4000000),
        n_trials=200,
        data_type=data_type,
    )
elif fit_method == "lbo":
    print("Using leave-block-out validation")
    fitter = LeaveBlockOutValidation(
        block,
        distance_threshold_range=(0, 4_000_000),
        n_trials=200,
        data_type=data_type,
    )
else:
    raise ValueError(f"Fit method {fit_method} not supported")

start_time = time()
fitter.fit(data, mask, imputer)
end_time = time()
print(f"Time taken to fit imputer: {end_time - start_time} seconds")

# Impute missing values
imputations = []
for row, col in tqdm(test_block, desc="Imputing missing values"):
    imputed_value = imputer.impute(row, col, data, mask)
    imputations.append(imputed_value)

ground_truth = data[test_inds_rows, test_inds_cols]
imputations = np.array(imputations)

drnn_errs = np.abs(imputations - ground_truth)

# setup usvt imputation
usvt_data = data.copy()
usvt_mask = mask.copy()
usvt_mask[test_inds_rows, test_inds_cols] = 0
usvt_data[mask != 1] = np.nan
usvt_imputed = usvt(usvt_data)
usvt_errs = list(np.abs(usvt_imputed[test_inds_rows, test_inds_cols] - ground_truth))
logging.debug(f"DR-NN mean absolute error: {np.mean(drnn_errs)}")
logging.debug(f"USVT mean absolute error: {np.mean(usvt_errs)}")
# Plot the error boxplot
dr_nn_data = drnn_errs

data = [usvt_errs, dr_nn_data]

# TODO (Albert): move plotting functionality to a separate script
plt.figure()

# Create boxplot
box = plt.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
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
plt.xticks([1, 2], ["USVT", "DR-NN"], fontsize=15)
plt.ylabel(r"Absolute error", fontsize=15)
ax1 = plt.gca()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.grid(True, alpha=0.4)

plt.xlabel(r"", fontsize=15)
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
save_path = os.path.join(
    figures_dir, f"{estimation_method}_{fit_method}_nn_error_boxplot.pdf"
)
print(f"Saving plot to {save_path}...")
plt.savefig(save_path, bbox_inches="tight")
