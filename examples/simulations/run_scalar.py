"""Script to run NN imputers + USVT baseline on simulated data
using 20% of the observed indices as a test block. The main experiment involves increasing the size of the matrix.

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

# import baseline methods
from baselines import usvt, softimpute

# import nearest neighbor methods
from nearest_neighbors.data_types import Scalar
from nearest_neighbors.estimation_methods import TSEstimator#, AutoEstimator
from nearest_neighbors import NearestNeighborImputer
from nearest_neighbors.fit_methods import (
    DRLeaveBlockOutValidation,
    TSLeaveBlockOutValidation,
    LeaveBlockOutValidation,
    #AutoDRTSLeaveBlockOutValidation,
)
from nearest_neighbors.datasets.dataloader_factory import NNData
from nearest_neighbors.vanilla_nn import row_row, col_col
from nearest_neighbors.dr_nn import dr_nn

from nearest_neighbors.utils.experiments import get_base_parser, setup_logging
import argparse

parser = get_base_parser()
parser.add_argument("--allow_self_neighbor", action=argparse.BooleanOptionalAction, help="Allow self neighbor")
args = parser.parse_args()
output_dir = args.output_dir
estimation_method = args.estimation_method
fit_method = args.fit_method
seed = args.seed
log_level = args.log_level
allow_self_neighbor = args.allow_self_neighbor
setup_logging(log_level)
logger = logging.getLogger(__name__)
print(args.allow_self_neighbor)
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

# Load the simulated data dataset
# NOTE: the raw and processed data is cached in .joblib_cache
m_size = [2**4, 2**5, 2**6, 2**7]
num_trials = 30


def random_trial() -> None:
    """Run the random trial experiment (20% of observed is test set)."""
    sizes_data = []
    for i, size in enumerate(m_size):
        logger.info(f"Simulating data with size {size}x{size}")
        # Simulate data
        # NOTE: the raw and processed data is cached in .joblib_cache
        start_time = time()
        sim_dataloader = NNData.create(
            "synthetic_data", num_rows=size, num_cols=size, seed=i, miss_prob=0.0
        )
        data, mask = sim_dataloader.process_data_scalar()
        data_state = sim_dataloader.get_full_state_as_dict()
        data_true = data_state["full_data_true"]
        elapsed_time = time() - start_time
        logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")

        logger.info("Using scalar data type")
        data_type = Scalar()

        holdout_inds = np.nonzero(mask == 1)
        # cv_mask = mask.copy()
        # cv_mask[:, size - 1] = 0
        # cv_holdout_inds = np.nonzero(cv_mask == 1)
        # test_inds_rows = holdout_inds[0]
        # test_inds_cols = holdout_inds[1]
        inds_rows = holdout_inds[0]
        inds_cols = holdout_inds[1]
        range_inds = np.arange(len(inds_rows))

        # UNCOMMENT FOR 20% RANDOM TESTING
        # ----------------
        # randomly shuffle indices
        rng.shuffle(range_inds)
        # 20% of the indices will be used for testing
        test_size = int(0.2 * len(range_inds))
        test_inds = range_inds[:test_size]
        # 80% of the indices will be used for training
        train_inds = range_inds[test_size:]
        range_train_inds = np.arange(len(train_inds))
        rng.shuffle(range_train_inds)
        # 10% of the training indices will be used for cv holdout
        cv_size = int(0.2 * len(train_inds))
        cv_inds = range_train_inds[:cv_size]
        # get the rows and columns of the train indices

        cv_inds_rows = list(inds_rows[train_inds][cv_inds])
        cv_inds_cols = list(inds_cols[train_inds][cv_inds])
        # get the rows and columns of the test indices for random testing
        test_inds_rows = list(inds_rows[test_inds])
        test_inds_cols = list(inds_cols[test_inds])

        # ------------------

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
        elif estimation_method == "softimpute":
            logger.info("Using SoftImpute estimation")
            # setup usvt imputation
            si_data = data.copy()
            si_mask = mask.copy()
            si_mask[test_inds_rows, test_inds_cols] = 0
            si_data[si_mask != 1] = np.nan
            # impute missing values simultaneously
            start_time = time()
            si_imputed = softimpute(si_data)
            elapsed_time = time() - start_time
            imputations = si_imputed[test_inds_rows, test_inds_cols]
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
                    distance_threshold_range_row=(0, 50),
                    distance_threshold_range_col=(0, 50),
                    n_trials=200,
                    data_type=data_type,
                    allow_self_neighbor=args.allow_self_neighbor,
                )
            elif estimation_method == "row-row":
                logger.info("Using row-row estimation")
                imputer = row_row()

                logger.info("Using leave-block-out validation")
                fitter = LeaveBlockOutValidation(
                    block,
                    distance_threshold_range=(0, 50),
                    n_trials=200,
                    data_type=data_type,
                    allow_self_neighbor=args.allow_self_neighbor,
                )
            elif estimation_method == "col-col":
                logger.info("Using col-col estimation")
                imputer = col_col()

                logger.info("Using leave-block-out validation")
                fitter = LeaveBlockOutValidation(
                    block,
                    distance_threshold_range=(0, 50),
                    n_trials=200,
                    data_type=data_type,
                    allow_self_neighbor=args.allow_self_neighbor,
                )
            elif estimation_method == "ts":
                logger.info("Using two-sided estimation")
                estimator = TSEstimator()
                imputer = NearestNeighborImputer(estimator, data_type)

                logger.info("Using two-sided fit method")
                # Fit the imputer using leave-block-out validation
                fitter = TSLeaveBlockOutValidation(
                    block,
                    distance_threshold_range_row=(0, 50),
                    distance_threshold_range_col=(0, 50),
                    n_trials=200,
                    data_type=data_type,
                    allow_self_neighbor=args.allow_self_neighbor,
                )
            # elif estimation_method == "autonn":
            #     logger.info("AutoNN Incomplete")
                # logger.info("Using AutoNN estimation")
                # estimator = AutoEstimator()
                # imputer = NearestNeighborImputer(
                #     estimator, data_type
                # )

                # logger.info("Using AutoNN fit method")
                # # Fit the imputer using leave-block-out validation
                # fitter = AutoDRTSLeaveBlockOutValidation(
                #     block,
                #     distance_threshold_range_row=(0, 50),
                #     distance_threshold_range_col=(0, 50),
                #     gamma_range=(-1, 1),
                #     n_trials=200,
                #     data_type=data_type,
                #     allow_self_neighbor=args.allow_self_neighbor,
                # )
            else:
                raise ValueError(
                    f"Estimation method {estimation_method} and fit method {fit_method} not supported"
                )

            start_time = time()
            trials = fitter.fit(data, mask_test, imputer, ret_trials=False)
            end_time = time()
            fit_times = [end_time - start_time] * len(test_block)

            # CODE FOR EXTRACTING TRIAL METADATA
            # if (
            #     not isinstance(trials, float)
            #     and not isinstance(trials, int)
            #     and isinstance(trials[1], Trials)
            # ):
            #     trials = trials[1]
            #     trial_data = []
            #     for trial in trials.trials:
            #         row = {}
            #         # get param vals
            #         params = trial["misc"]["vals"]
            #         for param_name, param_values in params.items():
            #             if param_values:
            #                 row[param_name] = float(param_values[0])

            #         row["loss"] = float(trial["result"]["loss"])
            #         trial_data.append(row)

            #     df_trials = pd.DataFrame(trial_data)
            #     trials_save_path = os.path.join(
            #         results_dir, f"cvtrials-{estimation_method}-{fit_method}.csv"
            #     )
            #     logger.info(f"Saving trials data to {trials_save_path}...")
            #     df_trials.to_csv(trials_save_path, index=False)

            # Impute missing values
            imputations = []
            imputation_times = []
            for row, col in tqdm(test_block, desc="Imputing missing values"):
                #mask[row, col] = 0
                start_time = time()
                imputed_value = imputer.impute(row, col, data, mask, allow_self_neighbor=args.allow_self_neighbor)
                elapsed_time = time() - start_time
                imputation_times.append(elapsed_time)
                imputations.append(imputed_value)
                #mask[row, col] = 1
                # restore the mask for next ind
            imputations = np.array(imputations)

        ground_truth = data_true[test_inds_rows, test_inds_cols]
        est_errors = np.abs(imputations - ground_truth)
        logger.info(f"Mean absolute error: {np.mean(est_errors)}")

        df_size = pd.DataFrame(
            data={
                "estimation_method": estimation_method,
                "fit_method": fit_method,
                "est_errors": est_errors,
                "row": test_inds_rows,
                "col": test_inds_cols,
                "time_impute": imputation_times,
                "time_fit": fit_times,
                "size": size,
            }
        )
        sizes_data.append(df_size)
        # print(df[["est_errors", "time_impute", "time_fit"]].describe())
    df = pd.concat(sizes_data, ignore_index=True)
    logger.info(f"Saving est_errors to {save_path}...")
    df.to_csv(save_path, index=False)


def cantor(x: int, y: int) -> int:
    """Cantor pairing function to map two integers to a single integer."""
    return int(0.5 * (x + y) * (x + y + 1) + y)


def last_col_trial() -> None:
    """Run the last column trial experiment (last column is test set)."""
    all_data = []
    for i, size in enumerate(m_size):
        df_size = []
        for j in tqdm(range(num_trials), desc="Simulating data"):
            logger.info(f"Simulating data with size {size}x{size}")
            # Simulate data
            # NOTE: the raw and processed data is cached in .joblib_cache
            start_time = time()
            sim_dataloader = NNData.create(
                "synthetic_data",
                num_rows=size,
                num_cols=size,
                seed=cantor(i, j),
                miss_prob=0.5,
                stddev_noise=0.001
            )
            data, mask = sim_dataloader.process_data_scalar()
            data_state = sim_dataloader.get_full_state_as_dict()
            data_true = data_state["full_data_true"]
            # NOTE: this is denoised data
            #print("Max data: ", np.nanmax(data))
            #exit()
            #data = data_true.copy()
            elapsed_time = time() - start_time
            logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")
            logger.info("Using scalar data type")
            data_type = Scalar()

            holdout_inds = np.nonzero(mask == 1)
            cv_mask = mask.copy()
            cv_mask[:, size - 1] = 0
            cv_holdout_inds = np.nonzero(cv_mask == 1)

            test_inds_rows = holdout_inds[0]
            test_inds_cols = holdout_inds[1]
            inds_rows = cv_holdout_inds[0]
            inds_cols = cv_holdout_inds[1]
            range_inds = np.arange(len(inds_rows))

            # UNCOMMENT FOR 20% RANDOM TESTING
            # ----------------
            # randomly shuffle indices
            rng.shuffle(range_inds)
            cv_size = int(0.1 * len(range_inds))
            cv_inds = range_inds[:cv_size]
            # get the rows and columns of the train indices

            cv_inds_rows = list(inds_rows[cv_inds])
            cv_inds_cols = list(inds_cols[cv_inds])
            # get the rows and columns of the test indices for random testing
            # test_inds_rows = list(inds_rows[test_inds])
            # test_inds_cols = list(inds_cols[test_inds])

            # get the rows and columns of the test indices for last col testing
            test_inds_rows = list(test_inds_rows[test_inds_cols == size - 1])
            test_inds_cols = list(test_inds_cols[test_inds_cols == size - 1])
            # ------------------

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
            elif estimation_method == "softimpute":
                logger.info("Using SoftImpute estimation")
                # setup usvt imputation
                si_data = data.copy()
                si_mask = mask.copy()
                imputations = []
                imputation_times = []
                for row, col in test_block:
                    si_mask[row, col] = 0
                    si_data_test = si_data.copy()
                    si_data_test[si_mask != 1] = np.nan
                #si_mask[test_inds_rows, test_inds_cols] = 0
                #si_data[si_mask != 1] = np.nan
                # impute missing values simultaneously
                    start_time = time()
                    si_imputed = softimpute(si_data_test)
                    elapsed_time = time() - start_time
                    si_mask[row, col] = 1
                    imputations.append(si_imputed[row, col])
                # set the time to the average time per imputation
                    imputation_times.append([elapsed_time / len(test_block)])
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
            #     elif estimation_method == "autonn":
            #         logger.info("Using AutoNN estimation")
            #         estimator = AutoEstimator()
            #         imputer = NearestNeighborImputer(
            #             estimator, data_type
            #         )

            #         logger.info("Using AutoNN fit method")
            #         # Fit the imputer using leave-block-out validation
            #         fitter = AutoDRTSLeaveBlockOutValidation(
            #             block,
            #             distance_threshold_range_row=(0, 1),
            #             distance_threshold_range_col=(0, 1),
            #             gamma_range=(-1, 1),
            #             n_trials=200,
            #             data_type=data_type,
            #             allow_self_neighbor=args.allow_self_neighbor,
            #         )
                else:
                    raise ValueError(
                        f"Estimation method {estimation_method} and fit method {fit_method} not supported"
                    )

                start_time = time()
                # USE this to get trail metadata
                # trials = fitter.fit(data, mask_test, imputer, ret_trials=True)
                fitter.fit(data, mask_test, imputer, ret_trials=False)
                end_time = time()
                fit_times = [end_time - start_time] * len(test_block)

                # CODE FOR EXTRACTING TRIAL METADATA
                # if not isinstance(trials, float) and not isinstance(trials, int) and isinstance(trials[1], Trials):
                #     trials = trials[1]
                #     trial_data = []
                #     for trial in trials.trials:
                #         row = {}
                #         # get param vals
                #         params = trial['misc']['vals']
                #         for param_name, param_values in params.items():
                #             if param_values:
                #                 row[param_name] = float(param_values[0])

                #         row['loss'] = float(trial['result']['loss'])
                #         trial_data.append(row)

                #     df_trials = pd.DataFrame(trial_data)
                #     trials_save_path = os.path.join(
                #     results_dir, f"cvtrials-{estimation_method}-{fit_method}.csv"
                #     )
                #     logger.info(f"Saving trials data to {trials_save_path}...")
                #     df_trials.to_csv(trials_save_path, index=False)

                # Impute missing values
                imputations = []
                imputation_times = []
                for (
                    row,
                    col,
                ) in test_block:
                    mask[row, col] = 0
                    start_time = time()
                    imputed_value = imputer.impute(row, col, data, mask, allow_self_neighbor=args.allow_self_neighbor)
                    elapsed_time = time() - start_time
                    imputation_times.append(elapsed_time)
                    imputations.append(imputed_value)
                    # restore the mask for next ind
                    mask[row, col] = 1
                imputations = np.array(imputations)

            ground_truth = data_true[test_inds_rows, test_inds_cols]
            est_errors = np.abs(imputations - ground_truth)
            logger.info(f"Mean absolute error: {np.nanmean(est_errors)}")

            df_trial = pd.DataFrame(
                data={
                    "estimation_method": estimation_method,
                    "fit_method": fit_method,
                    "est_errors": est_errors,
                    "row": test_inds_rows,
                    "col": test_inds_cols,
                    "time_impute": imputation_times,
                    "time_fit": fit_times,
                    "size": size,
                    "sim_num": j
                }
            )
            df_size.append(df_trial)
        df_allsize = pd.concat(df_size, ignore_index=True)
        all_data.append(df_allsize)
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Saving est_errors to {save_path}...")
    df.to_csv(save_path, index=False)


# change this to change the experiment
last_col_trial()
