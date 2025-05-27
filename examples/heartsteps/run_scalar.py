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
from hyperopt import Trials
import warnings

# import baseline methods
from baselines import usvt, softimpute

# import nearest neighbor methods
from nearest_neighbors.data_types import Scalar
from nearest_neighbors.estimation_methods import (
    StarNNEstimator,
    TSEstimator,
    AutoEstimator,
)
from nearest_neighbors import NearestNeighborImputer
from nearest_neighbors.fit_methods import (
    DRLeaveBlockOutValidation,
    TSLeaveBlockOutValidation,
    LeaveBlockOutValidation,
    AutoDRTSLeaveBlockOutValidation,
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
allow_self_neighbor = args.allow_self_neighbor

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
hs_dataloader = NNData.create("heartsteps", agg="mean")
data, mask = hs_dataloader.process_data_scalar()
data = data[:, :200]  # only use the first 200 timesteps
mask = mask[:, :200]
mask[mask == 2] = 0
elapsed_time = time() - start_time
logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")

logger.info("Using scalar data type")
# data_type_kernel = DistributionKernelMMD(kernel="exponential")
# data_type_wasserstein = DistributionWassersteinSamples()
data_type = Scalar()

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

# NOTE: different experiemnt setup
# data = data[:, :200]  # only use the first 200 timesteps
# mask = mask[:, :200]
# elapsed_time = time() - start_time
# logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")

# logger.info("Using scalar data type")
# data_type = Scalar()

# holdout_inds = np.nonzero(mask == 1)
# inds_rows = holdout_inds[0]
# inds_cols = holdout_inds[1]
# range_inds = np.arange(len(inds_rows))

# # randomly shuffle indices
# rng.shuffle(range_inds)
# # 20% of the indices will be used for testing
# test_size = int(0.2 * len(range_inds))
# test_inds = range_inds[:test_size]
# # 80% of the indices will be used for training
# train_inds = range_inds[test_size:]
# range_train_inds = np.arange(len(train_inds))
# rng.shuffle(range_train_inds)
# # 20% of the training indices will be used for cv holdout
# cv_size = int(0.2 * len(train_inds))
# cv_inds = range_train_inds[:cv_size]
# # get the rows and columns of the train indices

# cv_inds_rows = list(inds_rows[train_inds][cv_inds])
# cv_inds_cols = list(inds_cols[train_inds][cv_inds])
# # get the rows and columns of the test indices
# test_inds_rows = list(inds_rows[test_inds])
# test_inds_cols = list(inds_cols[test_inds])

# block = list(zip(cv_inds_rows, cv_inds_cols))
# test_block = list(zip(test_inds_rows, test_inds_cols))

# mask_test = mask.copy()
# mask_test[test_inds_rows, test_inds_cols] = 0

if estimation_method == "tabpfn":
    logger.info("Using TabPFN estimator")

    from tabpfn import TabPFNRegressor

    test_cols = set(test_inds_cols)

    imputations = []

    train_rows = np.unique(inds_rows_cv)
    test_rows = np.unique(test_inds_rows)

    fitting_time = 0
    imputing_time = 0

    for col in tqdm(list(test_cols)):
        model = TabPFNRegressor()
        # for each observed
        X = data[train_rows, :col]
        y = data[train_rows, col]

        # remove nan values in y
        X = X[~np.isnan(y)]
        y = y[~np.isnan(y)]

        X_test = data[test_rows, :col]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_time = time()
            # fit the model
            model.fit(X, y)
            end_time = time()
            fitting_time += end_time - start_time

            start_time = time()
            # make predictions
            y_pred = model.predict(X_test)
            y_pred = y_pred[mask[test_rows, col] == 1]
            end_time = time()
            imputing_time += end_time - start_time

            for val in y_pred:
                imputations.append(val)

    imputations = np.array(imputations)
    imputation_times = [imputing_time / len(test_block)] * len(test_block)
    fit_times = [fitting_time / len(test_block)] * len(test_block)

elif estimation_method == "usvt":
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
elif estimation_method == "softimpute":
    logger.info("Using SoftImpute estimation")
    # setup usvt imputation
    si_data = data.copy()
    si_mask = mask.copy()
    si_data = si_data[:, 1:]
    si_mask = si_mask[:, 1:]
    imputations = []
    imputation_times = []
    is_nan_array = np.isnan(data)
    print("\nIs NaN array:")
    print(is_nan_array)

    # Step 2: Check which columns are entirely NaN
    all_nan_columns_mask = is_nan_array.all(axis=0)

    # Step 3: Get the indices of these columns
    nan_column_indices = np.where(all_nan_columns_mask)[0]
    print(nan_column_indices)

    for row, col in test_block:
        col = col - 1
        si_mask[row, col] = 0
        si_data_test = si_data.copy()
        si_data_test[si_mask != 1] = np.nan
        # si_mask[test_inds_rows, test_inds_cols] = 0
        # si_data[si_mask != 1] = np.nan
        # impute missing values simultaneously
        start_time = time()
        si_imputed = softimpute(si_data_test)
        elapsed_time = time() - start_time
        si_mask[row, col] = 1
        imputations.append(si_imputed[row, col])
        # set the time to the average time per imputation
        imputation_times.append(elapsed_time / len(test_block))
    fit_times = [0] * len(test_block)
elif estimation_method == "star":
    logger.info("Using star estimation")
    estimator = StarNNEstimator()
    imputer = NearestNeighborImputer(estimator, data_type, distance_threshold=-1)
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
    fit_times = [0] * len(test_block)
else:
    if estimation_method == "dr":
        logger.info("Using doubly robust estimation")
        imputer = dr_nn()

        logger.info("Using doubly robust fit method")
        # Fit the imputer using leave-block-out validation
        fitter = DRLeaveBlockOutValidation(
            block,
            distance_threshold_range_row=(0, 1),
            distance_threshold_range_col=(0, 1),
            n_trials=100,
            data_type=data_type,
        )
    elif estimation_method == "row-row":
        logger.info("Using row-row estimation")
        imputer = row_row()

        logger.info("Using leave-block-out validation")
        fitter = LeaveBlockOutValidation(
            block,
            distance_threshold_range=(0, 1),
            n_trials=100,
            data_type=data_type,
        )
    elif estimation_method == "col-col":
        logger.info("Using col-col estimation")
        imputer = col_col()

        logger.info("Using leave-block-out validation")
        fitter = LeaveBlockOutValidation(
            block,
            distance_threshold_range=(0, 1),
            n_trials=100,
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
            distance_threshold_range_row=(0, 1),
            distance_threshold_range_col=(0, 1),
            n_trials=100,
            data_type=data_type,
            allow_self_neighbor=True,
        )
        # for TSNN, self neighbor is necessary (for now)
        allow_self_neighbor = True
    elif estimation_method == "auto":
        logger.info("Using AutoNN estimation")
        estimator = AutoEstimator(is_percentile=True)
        imputer = NearestNeighborImputer(estimator, data_type)
        logger.info("Using AutoNN fit method")
        # Fit the imputer using leave-block-out validation
        fitter = AutoDRTSLeaveBlockOutValidation(
            block,
            distance_threshold_range_row=(0, 1),
            distance_threshold_range_col=(0, 1),
            alpha_range=(0, 1),
            n_trials=200,
            data_type=data_type,
            allow_self_neighbor=args.allow_self_neighbor,
        )
    else:
        raise ValueError(
            f"Estimation method {estimation_method} and fit method {fit_method} not supported"
        )

    start_time = time()
    trials = fitter.fit(data, mask_test, imputer, ret_trials=True)
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
        imputed_value = imputer.impute(
            row, col, data, mask_test, allow_self_neighbor=allow_self_neighbor
        )
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
