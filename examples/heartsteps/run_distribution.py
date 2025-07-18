"""Script to run distributional nearest neighbors on the heartsteps dataset
using the bottom right corner (last 50 timesteps and bottom 10 users).

Example usage (from root of repo):
```bash
python run_distribution.py -od OUTPUT_DIR -em ESTIMATION_METHOD -fm FIT_METHOD
```
"""

# standard imports
import numpy as np
from tqdm import tqdm
import logging
import os
from time import time
import pandas as pd
from hyperopt import Trials

# import baseline methods

# import nearest neighbor methods
from nsquared.data_types import (
    DistributionKernelMMD,
    DistributionWassersteinSamples,
    Scalar
)
from nsquared import NearestNeighborImputer
from nsquared.fit_methods import (
    LeaveBlockOutValidation,
)
from nsquared.estimation_methods import RowRowEstimator, ColColEstimator
from nsquared.datasets.dataloader_factory import NNData

# from nsquared.dr_nn import dr_nn
from nsquared.utils.experiments import get_base_parser, setup_logging

parser = get_base_parser()
parser.add_argument(
    "--data_type",
    "-dt",
    type=str,
    default="kernel_mmd",
    choices=["kernel_mmd", "wasserstein_samples"],
    help="Data type to use",
)
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

rng = np.random.default_rng(seed=seed)

# Load the heartsteps dataset
# NOTE: the raw and processed data is cached in .joblib_cache
start_time = time()
hs_dataloader = NNData.create("heartsteps", freq="1min", num_measurements=60)
data, mask = hs_dataloader.process_data_distribution()
data = data[:, :200]  # only use the first 200 timesteps
mask = mask[:, :200]

elapsed_time = time() - start_time
logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")

logger.info("Using distribution data type")

if args.data_type == "kernel":
    data_type = DistributionKernelMMD(kernel="exponential")
elif args.data_type == "wasserstein_samples":
    data_type = DistributionWassersteinSamples(num_samples=data[0, 0].shape[0])
else:
    raise ValueError(f"Data type {args.data_type} not supported")

save_path = os.path.join(
    results_dir, f"est_errors-{estimation_method}-{fit_method}-{args.data_type}.csv"
)

if os.path.exists(save_path) and not args.force:
    logger.info(f"Results already exist at {save_path}. Use --force to overwrite.")
    exit()


holdout_inds = np.nonzero(mask == 1)
inds_rows = holdout_inds[0]
inds_cols = holdout_inds[1]
# range_inds = np.arange(len(inds_rows))

inds_rows_cv = inds_rows[np.logical_and(inds_rows < 27, inds_cols < 150)]
inds_cols_cv = inds_cols[np.logical_and(inds_rows < 27, inds_cols < 150)]
cv_range_inds = np.arange(len(inds_rows_cv))
# randomly shuffle indices for cv
rng.shuffle(cv_range_inds)
# 20% of the indices will be used for cv holdout
cv_size = int(0.2 * len(cv_range_inds))
cv_inds = cv_range_inds[:cv_size]

# est_inds = range_inds[:test_size]
# # # 80% of the indices will be used for training
# train_inds = range_inds[test_size:]
# range_train_inds = np.arange(len(train_inds))
# rng.shuffle(range_train_inds)
# # 20% of the training indices will be used for cv holdout
# cv_size = int(0.2 * len(train_inds))
# cv_inds = range_train_inds[:cv_size]
# # get the rows and columns of the train indices

cv_inds_rows = list(inds_rows_cv[cv_inds])
cv_inds_cols = list(inds_cols_cv[cv_inds])
# get the rows and columns of the test indices (last 50 timesteps and bottom 10 users)
test_inds_rows = list(
    inds_rows[np.logical_and(holdout_inds[0] >= 27, holdout_inds[1] >= 150)]
)
test_inds_cols = list(
    inds_cols[np.logical_and(holdout_inds[0] >= 27, holdout_inds[1] >= 150)]
)

block = list(zip(cv_inds_rows, cv_inds_cols))
test_block = list(zip(test_inds_rows, test_inds_cols))

mask_test = mask.copy()
mask_test[test_inds_rows, test_inds_cols] = 0
if estimation_method == "row-row":
    logger.info("Using row-row estimation")
    imputer = NearestNeighborImputer(
        estimation_method=RowRowEstimator(is_percentile=True), data_type=data_type
    )
    logger.info("Using leave-block-out validation")
    fitter = LeaveBlockOutValidation(
        block, distance_threshold_range=(0, 1), n_trials=100, data_type=data_type
    )
elif estimation_method == "col-col":
    logger.info("Using col-col estimation")
    imputer = NearestNeighborImputer(
        estimation_method=ColColEstimator(is_percentile=True), data_type=data_type
    )

    logger.info("Using leave-block-out validation")
    fitter = LeaveBlockOutValidation(
        block,
        distance_threshold_range=(0, 1),
        n_trials=100,
        data_type=data_type,
    )
else:
    raise ValueError(
        f"Estimation method {estimation_method} and fit method {fit_method} not supported"
    )


start_time = time()
trials = fitter.fit(data, mask_test, imputer, ret_trials=True, verbose=True)
end_time = time()
fit_times = [end_time - start_time] * len(test_block)

# CODE FOR EXTRACTING TRIAL METADATA
if (
    not isinstance(trials, float)
    and not isinstance(trials, int)
    and isinstance(trials[1], Trials)
):
    trials = trials[1]
    trial_data = []
    for trial in trials.trials:
        row = {}
        # get param vals
        params = trial["misc"]["vals"]
        for param_name, param_values in params.items():
            if param_values:
                row[param_name] = float(param_values[0])

        row["loss"] = float(trial["result"]["loss"])
        trial_data.append(row)

    df_trials = pd.DataFrame(trial_data)
    trials_save_path = os.path.join(
        results_dir, f"cvtrials-{estimation_method}-{fit_method}.csv"
    )
    logger.info(f"Saving trials data to {trials_save_path}...")
    df_trials.to_csv(trials_save_path, index=False)

# Impute missing values
imputations = []
imputation_times = []
for row, col in tqdm(test_block, desc="Imputing missing values"):
    start_time = time()
    imputed_value = imputer.impute(row, col, data, mask_test)
    elapsed_time = time() - start_time
    imputation_times.append(elapsed_time)
    imputations.append(imputed_value)
# imputations = np.array(imputations)

ground_truth = data[test_inds_rows, test_inds_cols]
est_errors = []
error_datatype = Scalar()
for i in range(len(imputations)):
    est_errors.append(error_datatype.distance(np.nanmean(imputations[i]), np.nanmean(ground_truth[i])))
# est_errors = np.abs(imputations - ground_truth)
logger.info(f"Mean absolute error: {np.mean(est_errors)}")
save_imputations = np.array(imputations, dtype=object)
ground_truth = np.array(ground_truth, dtype=object)
# save imputations to pkl
imputations_save_path = os.path.join(
    results_dir, f"imputations-{estimation_method}-{fit_method}-{args.data_type}.npy"
)
logger.info(f"Saving imputations to {imputations_save_path}...")
np.save(imputations_save_path, save_imputations)
# save ground truth to pkl
ground_truth_save_path = os.path.join(
    results_dir, f"ground_truth-{estimation_method}-{fit_method}.npy"
)
logger.info(f"Saving ground truth to {ground_truth_save_path}...")
np.save(ground_truth_save_path, ground_truth)

df = pd.DataFrame(
    data={
        "estimation_method": args.data_type,
        # "imputation": save_imputations,
        # "ground_truth": ground_truth,
        "data_type": args.data_type,
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
