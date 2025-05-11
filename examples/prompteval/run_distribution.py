"""Script to run distributional NN imputer on the PromptEval dataset.

Example usage (from root of repo):
```bash
python run_distribution.py -od OUTPUT_DIR -em ESTIMATION_METHOD -fm FIT_METHOD
```
"""

import os
import logging
import tqdm
import numpy as np
import pandas as pd

from nearest_neighbors.datasets.dataloader_factory import NNData
from nearest_neighbors.utils.experiments import get_base_parser, setup_logging
from nearest_neighbors.estimation_methods import RowRowEstimator, ColColEstimator
from nearest_neighbors.data_types import (
    DistributionKernelMMD,
    DistributionWassersteinQuantile,
)
from nearest_neighbors.nnimputer import NearestNeighborImputer
from nearest_neighbors.fit_methods import LeaveBlockOutValidation

parser = get_base_parser()
parser.add_argument(
    "--data_type",
    type=str,
    default="kernel_mmd",
    choices=["kernel_mmd", "wasserstein_quantile"],
)
args = parser.parse_args()
output_dir = args.output_dir
estimation_method = args.estimation_method
fit_method = args.fit_method
seed = args.seed
log_level = args.log_level
propensity = args.propensity
data_type = args.data_type

setup_logging(log_level)
logger = logging.getLogger(__name__)

os.makedirs(output_dir, exist_ok=True)
results_dir = os.path.join(output_dir, "results")
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(
    results_dir,
    f"est_errors-{estimation_method}-{fit_method}-{data_type}-p{propensity}.csv",
)

if os.path.exists(save_path) and not args.force:
    logger.info(f"Results already exist at {save_path}. Use --force to overwrite.")
    exit()


rng = np.random.default_rng(seed=seed)

hs_dataloader = NNData.create(
    "prompteval",
    # models=['meta_llama_llama_3_8b', 'meta_llama_llama_3_8b_instruct', 'meta_llama_llama_3_70b_instruct', 'codellama_codellama_34b_instruct', ],
    # tasks=['college_mathematics', 'miscellaneous', 'moral_disputes', 'jurisprudence', 'moral_scenarios', 'college_chemistry'],
    propensity=propensity,
    seed=seed,
)
data, mask = hs_dataloader.process_data_distribution()

holdout_inds = np.nonzero(mask == 1)
inds_rows = holdout_inds[0]
inds_cols = holdout_inds[1]
range_inds = np.arange(len(inds_rows))

# randomly shuffle indices
rng.shuffle(range_inds)
# 20% of the indices will be used for testing
test_size = int(0.2 * len(range_inds))
logger.info(f"Test size: {test_size}")
test_inds = range_inds[:test_size]
# 80% of the indices will be used for training
train_inds = range_inds[test_size:]
range_train_inds = np.arange(len(train_inds))
rng.shuffle(range_train_inds)
# 20% of the training indices will be used for cv holdout
cv_size = int(0.2 * len(train_inds))
logger.info(f"CV size: {cv_size}")
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

match data_type:
    case "kernel_mmd":
        data_type = DistributionKernelMMD(kernel="exponential")
    case "wasserstein_quantile":
        data_type = DistributionWassersteinQuantile()
    case _:
        raise ValueError(f"Data type {data_type} not supported")

match estimation_method:
    case "row-row":
        estimator = RowRowEstimator()
    case "col-col":
        estimator = ColColEstimator()
    case _:
        raise ValueError(f"Estimation method {estimation_method} not supported")

fit_method = LeaveBlockOutValidation(
    block, distance_threshold_range=(0, 1.0), n_trials=10, data_type=data_type
)
imputer = NearestNeighborImputer(
    estimator,
    data_type,
    #  distance_threshold=0.5
)
eta = fit_method.fit(data, mask_test, imputer)
logger.info(f"Distance threshold: {eta}")

# Impute the missing values using the fitted model
imputations = []
for i, j in tqdm.tqdm(test_block):
    imputations.append(imputer.impute(i, j, data, mask_test))

ground_truth = data[test_inds_rows, test_inds_cols]

test_len = len(test_block)

est_errors = []
for t in range(test_len):
    if len(imputations[t]) > 0 and len(ground_truth[t]) > 0:
        err = data_type.distance(imputations[t], ground_truth[t])
    else:
        err = np.nan
    est_errors.append(err)

# Create DataFrame with list entries
df = pd.DataFrame(
    {
        "imputation": [
            x.tolist() if isinstance(x, np.ndarray) else x for x in imputations
        ],
        "ground_truth": [
            x.tolist() if isinstance(x, np.ndarray) else x for x in ground_truth
        ],
        "estimation_method": [args.estimation_method] * len(imputations),
        "fit_method": [args.fit_method] * len(imputations),
        "est_errors": est_errors,
        "row": test_inds_rows,
        "col": test_inds_cols,
        "data_type": [args.data_type] * len(imputations),
        "propensity": [args.propensity] * len(imputations),
        "seed": [args.seed] * len(imputations),
    }
)

print(df[["est_errors"]].describe())
logger.info(f"Saving est_errors to {save_path}...")
df.to_csv(save_path, index=False)
