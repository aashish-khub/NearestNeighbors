"""Script to run various Nearest Neighbor imputers (vanilla, DR, TS) and the USVT baseline on scalar-typed datasets.
Supports evaluation on both the synthetic and heartsteps datasets using leave-block-out validation.

Usage:
    Run ONE configuration:
        python run_all_scalar.py -od OUTPUT_DIR -em ESTIMATION_METHOD -fm FIT_METHOD -ds DATASET
    Run ALL configurations in parallel for ONE dataset:
        python run_all_scalar.py -od OUTPUT_DIR --all -ds heartsteps
    Run ALL configurations in parallel for ALL datasets:
        python run_all_scalar.py -od OUTPUT_DIR --all


Generates boxplots comparing absolute imputation errors across methods.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Tuple
from joblib import Memory, Parallel, delayed

from baselines import usvt
from nsquared.data_types import Scalar
from nsquared.estimation_methods import DREstimator, TSEstimator
from nsquared import NearestNeighborImputer
from nsquared.fit_methods import (
    DRLeaveBlockOutValidation,
    TSLeaveBlockOutValidation,
    LeaveBlockOutValidation,
)
from nsquared.datasets.dataloader_factory import NNData
from nsquared.vanilla_nn import row_row

SIMULATION_PARAMS = {
    "dimension_size": 100,
    "seed": 42,
    "miss_prob": 0.2,
    "holdout_size": 20,
    "test_size": 20,
    "max_threshold": 10,
    "num_trials": 200,
}

HEARTSTEPS_PARAMS = {
    "test_region_start_row": 21,
    "test_region_start_col": 159,
    "holdout_region_end_row": 9,
    "holdout_region_end_col": 25,
    "timesteps_to_consider": 200,  # only use the first 200 timesteps
    "max_threshold": 4_000_000,
    "num_trials": 200,
}
# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(
    dataset: str, memory: Memory
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load data for the specified dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load ('synthetic' or 'heartsteps').
    memory : Memory
        A joblib memory object for caching.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray | None]
        A tuple containing the data matrix, mask matrix, and ground truth matrix (if available).

    """
    if dataset == "synthetic":
        dataloader = NNData.create(
            "synthetic_data",
            num_rows=SIMULATION_PARAMS["dimension_size"],
            num_cols=SIMULATION_PARAMS["dimension_size"],
            seed=SIMULATION_PARAMS["seed"],
            miss_prob=SIMULATION_PARAMS["miss_prob"],
        )
        data, mask = dataloader.process_data_scalar()
        theta = dataloader.get_full_state_as_dict(include_metadata=True)[
            "full_data_true"
        ]
        return data, mask, theta
    elif dataset == "heartsteps":

        @memory.cache
        def get_heartsteps_data() -> Tuple[np.ndarray, np.ndarray]:
            return NNData.create("heartsteps").process_data_scalar()

        data, mask = get_heartsteps_data()
        timestep_cutoff = HEARTSTEPS_PARAMS["timesteps_to_consider"]
        return data[:, :timestep_cutoff], mask[:, :timestep_cutoff], None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def define_masks(
    dataset: str, data: np.ndarray, mask: np.ndarray
) -> Tuple[list, list, list, list]:
    """Define test and holdout masks for the given dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset ('synthetic' or 'heartsteps').
    data : np.ndarray
        The data matrix.
    mask : np.ndarray
        The mask matrix indicating missing values.

    Returns
    -------
    Tuple[list, list]
        A tuple containing the test block and holdout block as lists of index pairs.

    """
    inds_rows, inds_cols = np.nonzero(mask == 1)
    if dataset == "synthetic":
        test_mask_cutoff = SIMULATION_PARAMS["holdout_size"]
        holdout_mask_cutoff = (
            SIMULATION_PARAMS["dimension_size"] - SIMULATION_PARAMS["test_size"]
        )
        test_mask = (inds_rows > test_mask_cutoff) & (inds_cols > test_mask_cutoff)
        holdout_mask = (inds_rows < holdout_mask_cutoff) & (
            inds_cols < holdout_mask_cutoff
        )
    else:
        test_mask = (inds_rows > HEARTSTEPS_PARAMS["test_region_start_row"]) & (
            inds_cols > HEARTSTEPS_PARAMS["test_region_start_col"]
        )
        holdout_mask = (inds_rows < HEARTSTEPS_PARAMS["holdout_region_end_row"]) & (
            inds_cols < HEARTSTEPS_PARAMS["holdout_region_end_col"]
        )

    test_inds_rows = inds_rows[test_mask]
    test_inds_cols = inds_cols[test_mask]
    test_block = list(zip(test_inds_rows, test_inds_cols))
    test_inds_rows, test_inds_cols = zip(*test_block) if test_block else ([], [])

    # Build holdout block
    holdout_inds_rows = inds_rows[holdout_mask]
    holdout_inds_cols = inds_cols[holdout_mask]
    block = list(zip(holdout_inds_rows, holdout_inds_cols))
    block = [(i, j) for (i, j) in block if not np.isnan(data[i, j])]
    holdout_block = block
    return test_block, holdout_block, list(test_inds_rows), list(test_inds_cols)


def run_single_config(
    dataset: str, estimation_method: str, fit_method: str, memory: Memory
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Run a single configuration of the imputation process.

    Parameters
    ----------
    dataset : str
        The name of the dataset ('synthetic' or 'heartsteps').
    estimation_method : str
        The estimation method to use ('dr', 'ts', or 'vanilla').
    fit_method : str
        The fitting method to use ('dr', 'ts', or 'lbo').
    memory : Memory
        A joblib memory object for caching.

    Returns
    -------
    Tuple[str, np.ndarray, np.ndarray]
        A tuple containing the label, nearest neighbor errors, and USVT errors.

    """
    data_type = Scalar()
    data, mask, theta = load_data(dataset, memory)

    if estimation_method == "dr":
        imputer = NearestNeighborImputer(DREstimator(), data_type)
    elif estimation_method == "ts":
        imputer = NearestNeighborImputer(TSEstimator(), data_type)
    elif estimation_method == "vanilla":
        imputer = row_row()
    else:
        raise ValueError(f"Unsupported estimation method: {estimation_method}")

    test_block, holdout_block, test_inds_rows, test_inds_cols = define_masks(
        dataset, data, mask
    )

    if fit_method == "dr":
        fit_class = DRLeaveBlockOutValidation
    elif fit_method == "ts":
        fit_class = TSLeaveBlockOutValidation
    elif fit_method == "lbo":
        fit_class = LeaveBlockOutValidation
    else:
        raise ValueError(f"Unsupported fit method: {fit_method}")

    if dataset == "heartsteps":
        max_threshold = HEARTSTEPS_PARAMS["max_threshold"]
        num_trials = HEARTSTEPS_PARAMS["num_trials"]
    elif dataset == "synthetic":
        max_threshold = SIMULATION_PARAMS["max_threshold"]
        num_trials = SIMULATION_PARAMS["num_trials"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    fit_args = {
        "n_trials": num_trials,
        "data_type": data_type,
    }

    if fit_method in ["dr", "ts"]:
        fit_args["distance_threshold_range_row"] = (0, max_threshold)
        fit_args["distance_threshold_range_col"] = (0, max_threshold)
    else:
        fit_args["distance_threshold_range"] = (0, max_threshold)

    fitter = fit_class(holdout_block, **fit_args)
    fitter.fit(data, mask, imputer)

    if dataset == "synthetic":
        if theta is None:
            raise ValueError("Theta is None for the synthetic dataset.")
        ground_truth = theta[tuple(zip(*test_block))]
    else:
        ground_truth = data[test_inds_rows, test_inds_cols]

    imputations = []
    for row, col in tqdm(test_block, desc="Imputing missing values"):
        imputed_value = imputer.impute(row, col, data, mask)
        imputations.append(imputed_value)
    ground_truth = data[test_inds_rows, test_inds_cols]
    imputations = np.array(imputations)
    nn_errors = np.abs(imputations - ground_truth)

    usvt_data = data.copy()
    usvt_mask = mask.copy()
    usvt_mask[test_inds_rows, test_inds_cols] = 0
    usvt_data[mask != 1] = np.nan
    usvt_imputed = usvt(usvt_data)
    usvt_errs = np.array(
        list(np.abs(usvt_imputed[test_inds_rows, test_inds_cols] - ground_truth))
    )

    label = f"{estimation_method.upper()}-{fit_method.upper()}"
    return label, nn_errors, usvt_errs


def plot_all_errors(results: dict | tuple, output_dir: str, dataset: str) -> None:
    """Plot the imputation errors for all methods and datasets."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    colors = ["red", "lightgray", "blue", "green"]
    if isinstance(results, dict):
        num_subplots = len(results)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 10))
        for ax, (ds, res) in zip(axes, results.items()):
            labels = ["USVT"] + [label for label, _, _ in res]
            errors = [res[0][2]] + [r[1] for r in res]
            box = ax.boxplot(errors, patch_artist=True, widths=0.6, showfliers=False)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
            for median in box["medians"]:
                median.set_color("black")
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_title(
                f"Imputation Error Comparison - {ds.capitalize()} Dataset", fontsize=14
            )
            ax.set_ylabel("Absolute error")
            ax.grid(True, alpha=0.4)

        plt.tight_layout()
        save_path = os.path.join(figures_dir, f"ALL_{dataset}_error_boxplot.pdf")
    else:
        plt.figure()
        label, nn_errors, usvt_errors = results[0]
        labels = ["USVT"] + [label]
        errors = [usvt_errors] + [nn_errors]
        box = plt.boxplot(errors, patch_artist=True, widths=0.6, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
        for median in box["medians"]:
            median.set_color("black")
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.title(
            f"Imputation Error Comparison - {dataset.capitalize()} Dataset", fontsize=14
        )
        plt.ylabel("Absolute error", fontsize=15)
        plt.grid(True, alpha=0.4)
        save_path = os.path.join(figures_dir, f"{label}_{dataset}_error_boxplot.pdf")

    plt.savefig(save_path, bbox_inches="tight")
    logger.info(f"Saved combined plot to {save_path}")


def main() -> None:
    """Main function to execute the script.

    Parses command-line arguments, sets up configurations, and runs the
    imputation process for the specified datasets and methods.
    """
    parser = ArgumentParser()
    parser.add_argument("--output_dir", "-od", type=str, default="out")
    parser.add_argument(
        "--estimation_method", "-em", type=str, choices=["dr", "ts", "vanilla"]
    )
    parser.add_argument("--fit_method", "-fm", type=str, choices=["dr", "ts", "lbo"])
    parser.add_argument(
        "--dataset",
        "-ds",
        type=str,
        default="all",
        choices=["heartsteps", "synthetic", "all"],
    )
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    memory = Memory(".joblib_cache", verbose=1)

    configs = [("vanilla", "lbo"), ("dr", "dr"), ("ts", "ts")]
    datasets = [args.dataset] if args.dataset != "all" else ["heartsteps", "synthetic"]
    if args.all:
        scheduled_jobs = []
        if args.dataset == "all":
            # this is for when we are using ALL estimation methods/fit methods AND all datasets
            for dataset in datasets:
                for em, fm in configs:
                    scheduled_jobs.append((dataset, em, fm))
            print(f"Running the following {len(scheduled_jobs)} jobs:")
            for i in scheduled_jobs:
                print(i)
            results = Parallel(n_jobs=-1)(
                delayed(run_single_config)(ds, em, fm, memory)
                for ds, em, fm in scheduled_jobs
            )
            grouped = {ds: [] for ds in datasets}
            for (ds, _, _), res in zip(scheduled_jobs, results):
                grouped[ds].append(res)
            plot_all_errors(grouped, args.output_dir, "combined")
        else:
            # this is for when we are using ALL estimation methods/fit methods AND ONLY ONE dataset
            for em, fm in configs:
                scheduled_jobs.append((args.dataset, em, fm))
            print(f"Running the following {len(scheduled_jobs)} jobs:")
            for i in scheduled_jobs:
                print(i)
            results = Parallel(n_jobs=-1)(
                delayed(run_single_config)(ds, em, fm, memory)
                for ds, em, fm in scheduled_jobs
            )
            # results = tuple(results)
            grouped = {args.dataset: []}
            for (ds, _, _), res in zip(scheduled_jobs, results):
                grouped[ds].append(res)
            plot_all_errors(grouped, args.output_dir, args.dataset)
    else:
        # this is for when we are using ONE (estimation method, fit method) AND only ONE dataset
        assert args.estimation_method and args.fit_method, (
            "Specify both -em and -fm when not using --all"
        )
        results = run_single_config(
            args.dataset,
            args.estimation_method,
            args.fit_method,
            memory,
        )
        plot_all_errors(results, args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
