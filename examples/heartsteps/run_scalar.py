import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from time import time
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Tuple
from joblib import Memory

from baselines import usvt
from nearest_neighbors.data_types import Scalar
from nearest_neighbors.estimation_methods import DREstimator, TSEstimator
from nearest_neighbors import NearestNeighborImputer
from nearest_neighbors.fit_methods import (
    DRLeaveBlockOutValidation,
    TSLeaveBlockOutValidation,
    LeaveBlockOutValidation,
)
from nearest_neighbors.datasets.dataloader_factory import NNData
from nearest_neighbors.vanilla_nn import row_row

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(dataset: str, memory: Memory):
    if dataset == "synthetic":
        dataloader = NNData.create("synthetic_data", num_rows=100, num_cols=100, seed=42, miss_prob=0.2)
        data, mask = dataloader.process_data_scalar()
        theta = dataloader.get_full_state_as_dict(include_metadata=True)["full_data_true"]
        return data, mask, theta
    elif dataset == "heartsteps":
        @memory.cache
        def get_heartsteps_data() -> Tuple[np.ndarray, np.ndarray]:
            return NNData.create("heartsteps").process_data_scalar()

        data, mask = get_heartsteps_data()
        return data[:, :200], mask[:, :200], None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def select_imputer(estimation_method: str, data_type):
    if estimation_method == "dr":
        return NearestNeighborImputer(DREstimator(), data_type)
    elif estimation_method == "ts":
        return NearestNeighborImputer(TSEstimator(), data_type)
    elif estimation_method == "vanilla":
        return row_row()
    else:
        raise ValueError(f"Unsupported estimation method: {estimation_method}")

def define_masks(dataset: str, data, mask):
    inds_rows, inds_cols = np.nonzero(mask)
    if dataset == "synthetic":
        test_mask = (inds_rows > 80) & (inds_cols > 80)
        holdout_mask = (inds_rows < 20) & (inds_cols < 20)
    else:
        test_mask = (inds_rows > 21) & (inds_cols > 159)
        holdout_mask = (inds_rows < 9) & (inds_cols < 25)

    test_block = list(zip(inds_rows[test_mask], inds_cols[test_mask]))
    holdout_block = [(i, j) for i, j in zip(inds_rows[holdout_mask], inds_cols[holdout_mask]) if not np.isnan(data[i, j])]
    return test_block, holdout_block

def fit_imputer(fit_method: str, block, data, mask, imputer, data_type, dataset: str):
    max_threshold = 4_000_000 if dataset == "heartsteps" else 10
    fit_class = {
        "dr": DRLeaveBlockOutValidation,
        "ts": TSLeaveBlockOutValidation,
        "lbo": LeaveBlockOutValidation
    }[fit_method]

    fit_args = {
        "distance_threshold_range_row": (0, max_threshold),
        "distance_threshold_range_col": (0, max_threshold),
        "n_trials": 200,
        "data_type": data_type
    } if fit_method in ["dr", "ts"] else {
        "distance_threshold_range": (0, max_threshold),
        "n_trials": 200,
        "data_type": data_type
    }

    fitter = fit_class(block, **fit_args)
    start_time = time()
    fitter.fit(data, mask, imputer)
    logger.info(f"Fitting took {time() - start_time:.2f} seconds")

def evaluate_imputer(test_block, imputer, data, mask, ground_truth):
    imputations = [imputer.impute(i, j, data, mask) for i, j in tqdm(test_block, desc="Imputing")]
    return np.abs(np.array(imputations) - ground_truth)

def plot_errors(usvt_errors, nn_errors, output_dir, label):
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure()
    box = plt.boxplot([usvt_errors, nn_errors], patch_artist=True, widths=0.6, showfliers=False)
    for patch, color in zip(box["boxes"], ["red", "blue"]):
        patch.set_facecolor(color)
    for median in box["medians"]:
        median.set_color("black")
    plt.xticks([1, 2], ["USVT", label])
    plt.ylabel("Absolute error", fontsize=15)
    plt.grid(True, alpha=0.4)
    save_path = os.path.join(figures_dir, f"{label.replace(': ', '_')}_error_boxplot.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    logger.info(f"Saved plot to {save_path}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", "-od", type=str, default="out")
    parser.add_argument("--estimation_method", "-em", type=str, default="vanilla", choices=["dr", "ts", "vanilla"])
    parser.add_argument("--fit_method", "-fm", type=str, default="lbo", choices=["dr", "ts", "lbo"])
    parser.add_argument("--dataset", "-ds", type=str, default="synthetic", choices=["heartsteps", "synthetic"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    memory = Memory(".joblib_cache", verbose=1)
    data_type = Scalar()

    data, mask, theta = load_data(args.dataset, memory)
    imputer = select_imputer(args.estimation_method, data_type)
    test_block, holdout_block = define_masks(args.dataset, data, mask)
    fit_imputer(args.fit_method, holdout_block, data, mask, imputer, data_type, args.dataset)

    ground_truth = (theta if args.dataset == "synthetic" else data)[tuple(zip(*test_block))]
    nn_errors = evaluate_imputer(test_block, imputer, data, mask, ground_truth)

    usvt_data = data.copy()
    usvt_data[mask != 1] = np.nan
    usvt_data[tuple(zip(*test_block))] = np.nan
    usvt_imputed = usvt(usvt_data)
    usvt_errors = np.abs(usvt_imputed[tuple(zip(*test_block))] - ground_truth)

    label = f"{args.estimation_method.upper()}-{args.fit_method.upper()}"
    plot_errors(usvt_errors, nn_errors, args.output_dir, label)

if __name__ == "__main__":
    main()
