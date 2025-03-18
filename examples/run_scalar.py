"""Script to recreate the experiment corresponding to Figure X from the following paper:

Dwivedi, R., Tian, K., Tomkins, S., Klasnja, P., Murphy, S., & Shah, D. (2022).
Counterfactual inference for sequential experiments. arXiv preprint arXiv:2202.06891.

Todo:
- [ ] Determine which experiment to recreate

Example usage:
```bash
python run_scalar.py -od OUTPUT_DIR --distance_threshold 0.5
```

"""

import logging
import os
from argparse import ArgumentParser
import numpy as np
from nearest_neighbors import NNData
from nearest_neighbors.vanilla_nn import row_row


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser = ArgumentParser()
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
output_dir = args.output_dir
distance_threshold = args.distance_threshold
DATASET = "heartsteps"

# Example: Load the HeartSteps dataset with default parameters.
logger.info("Loading HeartSteps dataset with default params in scalar form...")
data_generator = NNData.create(DATASET, download=True, save_processed=True)
# NOTE: data is a 2D array, where nan values indicate missing values
# NOTE: mask is a 2D array, where 0 indicates
data, mask = data_generator.process_data_scalar(cached=True)
logger.debug(f"Data shape: {data.shape}")
logger.debug(f"Mask shape: {mask.shape}")
logger.debug(f"Data: {data}")
logger.debug(f"Mask: {mask}")

# Load NNImputer
nn_imputer = row_row(
    distance_threshold=distance_threshold,
)

# TODO: perform fitting procedure

# perform imputation
imputed_data = np.zeros_like(data)
for r, c in np.ndindex(data.shape):
    if mask[r, c] == 0:
        imputed_value = nn_imputer.impute(r, c, data, mask)
        imputed_data[r, c] = imputed_value
logger.info(f"Imputed data: {imputed_data}")

# save imputed data
save_dir = os.path.join(output_dir, "scores")
os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, f"imputed_{dataset}-eta{distance_threshold}.npy")
# logger.info(f"Saving imputed data to {save_path}...")
# np.save(save_path, imputed_data)
