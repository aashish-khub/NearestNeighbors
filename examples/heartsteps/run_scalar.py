"""Script to run NN imputers + USVT baseline on the heartsteps dataset
using 20% of the observed indices as a test block.

Example usage (from root of repo):
```bash
python run_scalar.py -od OUTPUT_DIR -em ESTIMATION_METHOD -fm FIT_METHOD
```
"""

# standard imports
import numpy as np
from tqdm import tqdm
import logging
import os
from time import time
import pandas as pd

# import baseline methods
from baselines import usvt

# import nearest neighbor methods
from nearest_neighbors.data_types import Scalar
from nearest_neighbors.estimation_methods import TSEstimator
from nearest_neighbors import NearestNeighborImputer
from nearest_neighbors.fit_methods import (
    DRLeaveBlockOutValidation,
    TSLeaveBlockOutValidation,
    LeaveBlockOutValidation,
)
from nearest_neighbors.datasets.dataloader_factory import NNData
from nearest_neighbors.vanilla_nn import row_row, col_col
from nearest_neighbors.dr_nn import dr_nn

from nearest_neighbors.utils.experiments import get_base_parser, setup_logging

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir
estimation_method = args.estimation_method
fit_method = args.fit_method
seed = args.seed
log_level = args.log_level

setup_logging(log_level)
logger = logging.getLogger(__name__)

os.makedirs(output_dir, exist_ok=True)
results_dir = os.path.join(output_dir, "results")
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(
    results_dir, f"est_errors-{estimation_method}-{fit_method}.csv"
)

if os.path.exists(save_path) and not args.force:
    logger.info(f"Results already exist at {save_path}. Use --force to overwrite.")
    exit()

rng = np.random.default_rng(seed=seed)

# Load the heartsteps dataset
# NOTE: the raw and processed data is cached in .joblib_cache
start_time = time()
hs_dataloader = NNData.create("heartsteps")
data, mask = hs_dataloader.process_data_scalar()
data = data[:, :200]  # only use the first 200 timesteps
mask = mask[:, :200]
elapsed_time = time() - start_time
logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")

logger.info("Using scalar data type")
data_type = Scalar()

holdout_inds = np.nonzero(mask == 1)
inds_rows = holdout_inds[0]
inds_cols = holdout_inds[1]
range_inds = np.arange(len(inds_rows))

# randomly shuffle indices
rng.shuffle(range_inds)
# 20% of the indices will be used for testing
test_size = int(0.2 * len(range_inds))
test_inds = range_inds[:test_size]
# 80% of the indices will be used for training
train_inds = range_inds[test_size:]
range_train_inds = np.arange(len(train_inds))
rng.shuffle(range_train_inds)
# 20% of the training indices will be used for cv holdout
cv_size = int(0.2 * len(train_inds))
cv_inds = range_train_inds[:cv_size]
# get the rows and columns of the train indices

cv_inds_rows = list(inds_rows[train_inds][cv_inds])
cv_inds_cols = list(inds_cols[train_inds][cv_inds])
# get the rows and columns of the test indices
test_inds_rows = list(inds_rows[test_inds])
test_inds_cols = list(inds_cols[test_inds])

block = list(zip(cv_inds_rows, cv_inds_cols))
test_block = list(zip(test_inds_rows, test_inds_cols))

mask_test = mask.copy()
mask_test[test_inds_rows, test_inds_cols] = 0

if estimation_method == "usvt":
    logger.info("Using USVT estimation")
    # setup usvt imputation
    usvt_data = data.copy()
    usvt_mask = mask.copy()
    usvt_mask[test_inds_rows, test_inds_cols] = 0
    usvt_data[usvt_mask != 1] = np.nan
    # impute missing values simultaneously
    start_time = time()
    usvt_imputed = usvt(usvt_data)
    elapsed_time = time() - start_time
    imputations = usvt_imputed[test_inds_rows, test_inds_cols]
    # set the time to the average time per imputation
    imputation_times = [elapsed_time / len(test_block)] * len(test_block)
    fit_times = [0] * len(test_block)
else:
    if estimation_method == "dr":
        logger.info("Using doubly robust estimation")
        imputer = dr_nn()

        logger.info("Using doubly robust fit method")
        # Fit the imputer using leave-block-out validation
        fitter = DRLeaveBlockOutValidation(
            block,
            distance_threshold_range_row=(0, 4000000),
            distance_threshold_range_col=(0, 4000000),
            n_trials=200,
            data_type=data_type,
        )
    elif estimation_method == "row-row":
        logger.info("Using row-row estimation")
        imputer = row_row()

        logger.info("Using leave-block-out validation")
        fitter = LeaveBlockOutValidation(
            block,
            distance_threshold_range=(0, 4_000_000),
            n_trials=200,
            data_type=data_type,
        )
    elif estimation_method == "col-col":
        logger.info("Using col-col estimation")
        imputer = col_col()

        logger.info("Using leave-block-out validation")
        fitter = LeaveBlockOutValidation(
            block,
            distance_threshold_range=(0, 4_000_000),
            n_trials=200,
            data_type=data_type,
        )
    elif estimation_method == "ts":
        logger.info("Using two-sided estimation")
        estimator = TSEstimator()
        imputer = NearestNeighborImputer(estimator, data_type)

        logger.info("Using two-sided fit method")
        # Fit the imputer using leave-block-out validation
        fitter = TSLeaveBlockOutValidation(
            block,
            distance_threshold_range_row=(0, 4_000_000),
            distance_threshold_range_col=(0, 4_000_000),
            n_trials=200,
            data_type=data_type,
        )
    else:
        raise ValueError(
            f"Estimation method {estimation_method} and fit method {fit_method} not supported"
        )

    start_time = time()
    fitter.fit(data, mask_test, imputer)
    end_time = time()
    fit_times = [end_time - start_time] * len(test_block)

    # Impute missing values
    imputations = []
    imputation_times = []
    for row, col in tqdm(test_block, desc="Imputing missing values"):
        start_time = time()
        imputed_value = imputer.impute(row, col, data, mask_test)
        elapsed_time = time() - start_time
        imputation_times.append(elapsed_time)
        imputations.append(imputed_value)
    imputations = np.array(imputations)

ground_truth = data[test_inds_rows, test_inds_cols]
est_errors = np.abs(imputations - ground_truth)
logger.info(f"Mean absolute error: {np.mean(est_errors)}")

df = pd.DataFrame(
    data={
        "estimation_method": estimation_method,
        "fit_method": fit_method,
        "est_errors": est_errors,
        "row": test_inds_rows,
        "col": test_inds_cols,
        "time_impute": imputation_times,
        "time_fit": fit_times,
    }
)
print(df[["est_errors", "time_impute", "time_fit"]].describe())
logger.info(f"Saving est_errors to {save_path}...")
df.to_csv(save_path, index=False)
