"""Script to run Star NN imputer on simulated data
using 20% of the observed indices as a test block. The main experiment involves increasing the size of the matrix. 

Example usage (from root of repo):
```bash
python run_star_nn.py -od OUTPUT_DIR
```
"""

# standard imports
import numpy as np
from tqdm import tqdm
import logging
import os
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import nearest neighbor methods
from nearest_neighbors.data_types import Scalar
from nearest_neighbors.datasets.dataloader_factory import NNData
from nearest_neighbors.star_nn import star_nn

from nearest_neighbors.utils.experiments import get_base_parser, setup_logging

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir
seed = args.seed
log_level = args.log_level

setup_logging(log_level)
logger = logging.getLogger(__name__)

os.makedirs(output_dir, exist_ok=True)
results_dir = os.path.join(output_dir, "results")
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, "star_nn_performance.csv")

# if os.path.exists(save_path) and not args.force:
#     logger.info(f"Results already exist at {save_path}. Use --force to overwrite.")
#     exit()

rng = np.random.default_rng(seed=seed)

# Load the simulated data dataset
# NOTE: the raw and processed data is cached in .joblib_cache
k = 3  # Number of repetitions for each size
m_size = np.repeat([2**4, 2**5, 2**6, 2**7], k)
sizes_data = []
train_times = []
for i, size in zip(np.arange(len(m_size)), m_size):
    logger.info(f"Simulating data with size {size}x{size}")
    # Simulate data
    # NOTE: the raw and processed data is cached in .joblib_cache
    start_time = time()
    sim_dataloader = NNData.create("synthetic_data", num_rows=size, num_cols=size, seed=i, miss_prob=0.2)
    data, mask = sim_dataloader.process_data_scalar()
    data_state = sim_dataloader.get_full_state_as_dict()
    data_true = data_state["full_data_true"]
    elapsed_time = time() - start_time
    logger.info(f"Time to load and process data: {elapsed_time:.2f} seconds")
    logger.info(f"Noise variance in the dataset: {np.var(data[mask == 1] - data_true[mask == 1])}")

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
    # get the rows and columns of the test indices
    test_inds_rows = list(inds_rows[test_inds])
    test_inds_cols = list(inds_cols[test_inds])

    # get the rows and columns of the train indices
    train_inds_rows = list(inds_rows[train_inds])
    train_inds_cols = list(inds_cols[train_inds])

    test_block = list(zip(test_inds_rows, test_inds_cols))

    mask_train = mask.copy()
    mask_train[test_inds_rows, test_inds_cols] = 0
    mask_test = mask.copy()
    mask_test[train_inds_rows, train_inds_cols] = 0

    # Initialize Star NN imputer
    logger.info("Using Star NN imputation")
    imputer = star_nn()
    
    
    
    # Fit the imputer
    start_time = time()
    imputed_train_data = imputer.fit(data, mask_train)
    end_time = time()
    fit_time = end_time - start_time
    logger.info(f"Fitting completed in {fit_time:.2f} seconds with final noise variance: {imputer.noise_variance}")
    
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
            "estimation_method": "star_nn",
            "est_errors": est_errors,
            "row": test_inds_rows,
            "col": test_inds_cols,
            "time_impute": imputation_times,
            "time_fit": [fit_time] * len(test_block),
            "size": size,
            "noise_variance": sim_dataloader.stddev_noise**2
        }
    )
    sizes_data.append(df_size)
    
    train_time = pd.DataFrame(
      data = {
          "estimation_method": "star_nn",
          "time_fit": fit_time,
          "size": size**2,
          "empirical_noise_variance": np.var(data[mask == 1] - data_true[mask == 1]),
          "estimated_noise_variance": imputer.noise_variance
      },
      index=[0]  # Add an index with a single row
    )
    train_times.append(train_time)

df = pd.concat(sizes_data, ignore_index=True)
logger.info(f"Saving Star NN performance metrics to {save_path}...")    
df.to_csv(save_path, index=False)

# Save train_times data to a separate CSV file
train_times_df = pd.concat(train_times, ignore_index=True)
train_times_path = os.path.join(results_dir, "star_nn_train_times.csv")
logger.info(f"Saving training times data to {train_times_path}...")
train_times_df.to_csv(train_times_path, index=False)

# Generate performance plots
plt.figure(figsize=(12, 8))
sns.boxplot(x="size", y="est_errors", data=df)
plt.title("Star NN Imputation Error by Matrix Size")
plt.xlabel("Matrix Size (2^n)")
plt.ylabel("Absolute Error")
plt.savefig(os.path.join(results_dir, "star_nn_error_by_size.png"))
plt.close()

plt.figure(figsize=(12, 8))
sns.boxplot(x="size", y="time_impute", data=df)
plt.title("Star NN Imputation Time by Matrix Size")
plt.xlabel("Matrix Size (2^n)")
plt.ylabel("Imputation Time (seconds)")
plt.savefig(os.path.join(results_dir, "star_nn_time_by_size.png"))
plt.close()

# Print summary statistics
logger.info("Summary statistics:")
logger.info(df.groupby("size")[["est_errors", "time_impute"]].describe()) 