"""Script for scalar matrix completion.

Example usage:
```bash
python run_scalar.py --dataset heartsteps -od OUTPUT_DIR --distance_threshold 0.5
```
"""

import logging
from argparse import ArgumentParser

from nearest_neighbors import NNData
from nearest_neighbors.vanilla_nn import row_row


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser = ArgumentParser()
parser.add_argument(
    "--dataset", "-d", type=str, default="heartsteps", help="Name of the dataset to use"
)
parser.add_argument(
    "--output_dir", "-od", type=str, default="out", help="Output directory"
)
parser.add_argument(
    "--distance_threshold",
    "-dt",
    type=float,
    default=0.5,
    help="Distance threshold to use",
)
args = parser.parse_args()
dataset = args.dataset
output_dir = args.output_dir
distance_threshold = args.distance_threshold

# Example: Load the HeartSteps dataset with default parameters.
logger.info("Loading HeartSteps dataset with default params in scalar form...")
data_generator = NNData.create(dataset, download=True, save_processed=True)
data, mask = data_generator.process_data_scalar(cached=True)
logger.debug(f"Data shape: {data.shape}")
logger.debug(f"Mask shape: {mask.shape}")

# Load NNImputer
nn_imputer = row_row(
    distance_threshold=distance_threshold,
)

# TODO: perform fitting procedure

# perform imputation
row, col = 0, 1
imputed_value = nn_imputer.impute(row, col, data, mask)
logger.info(f"True value at row {row} and column {col}: {data[row, col]}")
logger.info(f"Imputed value at row {row} and column {col}: {imputed_value}")

# TODO: save results
