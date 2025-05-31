"""Script to compare row_row, aw_nn, and USVT baseline on simulated data
using 20% of the observed indices as a test block. The main experiment involves increasing the size of the matrix.

"""

# standard imports
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import logging
import os
from time import time
import pandas as pd
from hyperopt import Trials

# import baseline methods
from baselines import usvt

# import nearest neighbor methods
from nsquared.data_types import Scalar
from nsquared.fit_methods import (
    LeaveBlockOutValidation,
)
from nsquared.datasets.dataloader_factory import NNData
from nsquared.vanilla_nn import row_row
from nsquared.aw_nn import aw_nn, AWNNEstimator
from nsquared.utils.experiments import get_base_parser, setup_logging

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

train_times_path = os.path.join(
    results_dir, f"train_times-{estimation_method}-{fit_method}.csv"
)

if os.path.exists(save_path) and not args.force:
    logger.info(f"Results already exist at {save_path}. Use --force to overwrite.")
    # exit()

rng = np.random.default_rng(seed=seed)

# Load the simulated data dataset
# NOTE: the raw and processed data is cached in .joblib_cache
k = 4  # Number of repetitions for each size
m_size = np.repeat([2**4, 2**5, 2**6, 2**7], k)
# m_size = np.repeat([2**4, 2**5, 2**6, 2**7, 2**8, 2**9], k)


def AWNNvsRowRowVsUsvt(m_size: npt.NDArray) -> None:
    """Compare the performance of AWNN and RowRow on simulated data"""
    sizes_data = []
    train_times = []
    for i, size in zip(np.arange(len(m_size)), m_size):
        logger.info(f"Simulating data with size {size}x{size}")
        # Simulate data
        # NOTE: the raw and processed data is cached in .joblib_cache
        start_time = time()
        sim_dataloader = NNData.create(
            "synthetic_data",
            num_rows=size,
            num_cols=size,
            seed=3 * i,
            miss_prob=0.4,
            snr=4,
            latent_factor_combination_model="multiplicative",
            rho=1,
        )
        data, mask = sim_dataloader.process_data_scalar()
        data_state = sim_dataloader.get_full_state_as_dict(include_metadata=True)
        true_noise_variance = data_state["generation_metadata"]["stddev_noise"] ** 2
        data_true = data_state["full_data_true"]
        elapsed_time = time() - start_time
        logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")

        empirical_noise_variance = np.var(data[mask == 1] - data_true[mask == 1])
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
        mask_test_aw_nn = np.zeros_like(mask)
        for i, j in test_block:
            mask_test_aw_nn[i, j] = 1

        mask_test = mask.copy()
        mask_test[test_inds_rows, test_inds_cols] = 0

        _m_avg = np.average(mask)
        _m_test_avg = np.average(mask_test)
        _m_aw_nn_avg = np.average(mask_test_aw_nn)

        logger.info("Using USVT estimation")
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

        ground_truth = data_true[test_inds_rows, test_inds_cols]
        est_errors = np.abs(imputations - ground_truth)
        logger.info(f"Mean absolute error: {np.mean(est_errors)}")

        df_size = pd.DataFrame(
            data={
                "estimation_method": "usvt",
                "est_errors": est_errors,
                "row": test_inds_rows,
                "col": test_inds_cols,
                "time_impute": imputation_times,
                "time_fit": fit_times,
                "size": size,
                **data_state["generation_metadata"],
            }
        )
        sizes_data.append(df_size)

        logger.info("Using AWNN imputation")
        imputer = aw_nn()

        print(f"Empirical noise variance: {empirical_noise_variance}")
        print(f"True noise variance: {true_noise_variance}")

        # imputer.set_noise_variance(empirical_noise_variance)

        # Fit the imputer
        start_time = time()
        _imputed_train_data = imputer.impute_all(data, mask_test)
        end_time = time()
        fit_time = end_time - start_time
        est_method = imputer.estimation_method
        if not isinstance(est_method, AWNNEstimator):
            raise ValueError("The estimation method should be AWNNEstimator for AWNN.")
        noise_variance = est_method.noise_variance

        logger.info(
            f"Fitting completed in {fit_time:.2f} seconds with final noise variance: {noise_variance}"
        )

        # imputer.set_noise_variance(empirical_noise_variance)
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

        ground_truth = data_true[test_inds_rows, test_inds_cols]
        est_errors = np.abs(imputations - ground_truth)
        logger.info(f"Mean absolute error: {np.mean(est_errors)}")

        df_size = pd.DataFrame(
            data={
                "estimation_method": "aw",
                "est_errors": est_errors,
                "row": test_inds_rows,
                "col": test_inds_cols,
                "time_impute": imputation_times,
                "time_fit": [fit_time] * len(test_block),
                "size": size,
                **data_state["generation_metadata"],
            }
        )
        sizes_data.append(df_size)

        train_time = pd.DataFrame(
            data={
                "estimation_method": "aw",
                "time_fit": fit_time,
                "size": size**2,
                "empirical_noise_variance": np.var(
                    data[mask == 1] - data_true[mask == 1]
                ),
                "estimated_noise_variance": noise_variance,
            },
            index=pd.Index([0]),  # Add an index with a single row
        )
        train_times.append(train_time)

        logger.info("Using row-row estimation")
        imputer = row_row()

        logger.info("Using leave-block-out validation")
        fitter = LeaveBlockOutValidation(
            block,
            distance_threshold_range=(0, 50),
            n_trials=200,
            data_type=data_type,
        )

        start_time = time()
        trials = fitter.fit(data, mask_test, imputer, ret_trials=True)
        end_time = time()
        fit_times = [end_time - start_time] * len(test_block)
        fit_time = end_time - start_time

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
            imputations = np.array(imputations)

        train_time = pd.DataFrame(
            data={
                "estimation_method": "row_row",
                "time_fit": fit_time,
                "size": size**2,
                "empirical_noise_variance": np.var(
                    data[mask == 1] - data_true[mask == 1]
                ),
                "estimated_noise_variance": np.var(imputations - ground_truth),
            },
            index=pd.Index([0]),  # Add an index with a single row
        )
        train_times.append(train_time)

        ground_truth = data_true[test_inds_rows, test_inds_cols]
        est_errors = np.abs(imputations - data_true[test_inds_rows, test_inds_cols])
        logger.info(f"Mean absolute error: {np.mean(est_errors)}")

        df_size = pd.DataFrame(
            data={
                "estimation_method": "row_row",
                "est_errors": est_errors,
                "row": test_inds_rows,
                "col": test_inds_cols,
                "time_impute": imputation_times,
                "time_fit": fit_times,
                "size": size,
                **data_state["generation_metadata"],
            }
        )
        sizes_data.append(df_size)
        # print(df[["est_errors", "time_impute", "time_fit"]].describe())
    df = pd.concat(sizes_data, ignore_index=True)
    logger.info(f"Saving est_errors to {save_path}...")
    df.to_csv(save_path, index=False)
    train_times_df = pd.concat(train_times, ignore_index=True)
    logger.info(f"Saving train_times to {train_times_path}...")
    train_times_df.to_csv(train_times_path, index=False)


def AWNNvsUsvt(m_size: npt.NDArray) -> None:
    """Compare the performance of AWNN and RowRow on simulated data"""
    sizes_data = []
    train_times = []
    for i, size in zip(np.arange(len(m_size)), m_size):
        logger.info(f"Simulating data with size {size}x{size}")
        # Simulate data
        # NOTE: the raw and processed data is cached in .joblib_cache
        start_time = time()
        sim_dataloader = NNData.create(
            "synthetic_data",
            num_rows=size,
            num_cols=size,
            seed=3 * i,
            miss_prob=0.2,
            snr=3,
            latent_factor_combination_model="multiplicative",
            rho=1,
        )
        data, mask = sim_dataloader.process_data_scalar()
        data_state = sim_dataloader.get_full_state_as_dict(include_metadata=True)
        true_noise_variance = data_state["generation_metadata"]["stddev_noise"] ** 2
        data_true = data_state["full_data_true"]
        elapsed_time = time() - start_time
        logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")

        empirical_noise_variance = np.var(data[mask == 1] - data_true[mask == 1])
        logger.info("Using scalar data type")
        _data_type = Scalar()

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

        _block = list(zip(cv_inds_rows, cv_inds_cols))
        test_block = list(zip(test_inds_rows, test_inds_cols))
        mask_test_aw_nn = np.zeros_like(mask)
        for i, j in test_block:
            mask_test_aw_nn[i, j] = 1

        mask_test = mask.copy()
        mask_test[test_inds_rows, test_inds_cols] = 0

        _m_avg = np.average(mask)
        _m_test_avg = np.average(mask_test)
        _m_aw_nn_avg = np.average(mask_test_aw_nn)

        logger.info("Using USVT estimation")
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

        ground_truth = data_true[test_inds_rows, test_inds_cols]
        est_errors = np.abs(imputations - ground_truth)
        logger.info(f"Mean absolute error: {np.mean(est_errors)}")

        df_size = pd.DataFrame(
            data={
                "estimation_method": "usvt",
                "est_errors": est_errors,
                "row": test_inds_rows,
                "col": test_inds_cols,
                "time_impute": imputation_times,
                "time_fit": fit_times,
                "size": size,
                **data_state["generation_metadata"],
            }
        )
        sizes_data.append(df_size)

        logger.info("Using AWNN imputation")
        imputer = aw_nn()

        print(f"Empirical noise variance: {empirical_noise_variance}")
        print(f"True noise variance: {true_noise_variance}")

        # imputer.set_noise_variance(empirical_noise_variance)

        # Fit the imputer
        start_time = time()
        _imputed_train_data = imputer.impute_all(data, mask_test)
        end_time = time()
        fit_time = end_time - start_time
        est_method = imputer.estimation_method
        if not isinstance(est_method, AWNNEstimator):
            raise ValueError("The estimation method should be AWNNEstimator for AWNN.")
        noise_variance = est_method.noise_variance
        logger.info(
            f"Fitting completed in {fit_time:.2f} seconds with final noise variance: {noise_variance}"
        )

        # imputer.set_noise_variance(empirical_noise_variance)
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

        ground_truth = data_true[test_inds_rows, test_inds_cols]
        est_errors = np.abs(imputations - ground_truth)
        logger.info(f"Mean absolute error: {np.mean(est_errors)}")

        df_size = pd.DataFrame(
            data={
                "estimation_method": "aw",
                "est_errors": est_errors,
                "row": test_inds_rows,
                "col": test_inds_cols,
                "time_impute": imputation_times,
                "time_fit": [fit_time] * len(test_block),
                "size": size,
                **data_state["generation_metadata"],
            }
        )
        sizes_data.append(df_size)

        train_time = pd.DataFrame(
            data={
                "estimation_method": "aw",
                "time_fit": fit_time,
                "size": size**2,
                "empirical_noise_variance": np.var(
                    data[mask == 1] - data_true[mask == 1]
                ),
                "estimated_noise_variance": noise_variance,
            },
            index=pd.Index([0]),  # Add an index with a single row
        )
        train_times.append(train_time)

    df = pd.concat(sizes_data, ignore_index=True)
    logger.info(f"Saving est_errors to {save_path}...")
    df.to_csv(save_path, index=False)

    train_times_df = pd.concat(train_times, ignore_index=True)
    logger.info(f"Saving train_times to {train_times_path}...")
    train_times_df.to_csv(train_times_path, index=False)


AWNNvsRowRowVsUsvt(m_size)

# AWNNvsUsvt(m_size)
