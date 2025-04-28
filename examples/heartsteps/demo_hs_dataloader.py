"""The following script demonstrates the usage of the HeartSteps dataloader.

The HeartSteps V1 dataset comes from https://github.com/klasnja/HeartStepsV1 and is data from a study on a mobile fitness intevention.
It includes step data on 37 participants over a ~52 day period. The default data configuration contains 1 hour of step data after each notification in 5min intervals.
The data can be processed into the distributional setting (4d tensor) or the scalar setting (2d matrix). The dimensions of the tensor are N x T x n x 1, where N is the number of participants,
T is the number of time points (5 notification per day for 52 days = 270), and n is the number of step samples (by default 12)/

To run this script, ensure you're in the root directory of the package and run:
```
    python examples/demo_hs_dataloader.py
```
"""

from nearest_neighbors.datasets.dataloader_factory import NNData
import numpy as np

print("====== HeartSteps Data Demo ======")
print("\nHeartSteps Help Information:")
NNData.help("heartsteps")

# Example: Load the HeartSteps dataset with default parameters.
print("===== Loading HeartSteps dataset with default params in distributional form ===")
data_generator = NNData.create("heartsteps")
data, mask = data_generator.process_data_distribution()

print("\nData matrix shape:")
print(data.shape)
print("\nFull data matrix shape:")
array_4d = []
# Fill the 4D array
for i in range(data.shape[0]):
    inner_array = []
    for j in range(data.shape[1]):
        # Try to convert the object to a numpy array if it isn't already
        inner_array.append(np.array(data[i, j], dtype=float))
    array_4d.append(inner_array)
full_data = np.array(array_4d)
print(full_data.shape)

print("\nAvailability Mask shape:")
print(mask.shape)
print("\nAvailability Mask (1 if observed, 0 if missing, 2 if unavailable):")
print(mask)

print(
    "\n===== Loading HeartSteps dataset with custom params in distributional form ==="
)
print("\n Custom params: freq='1min', participants=25, num_measurements=60")
custom_data_generator = NNData.create(
    "heartsteps", freq="1min", participants=25, num_measurements=60
)
data, mask = custom_data_generator.process_data_distribution()

print("\nData matrix shape:")
print(data.shape)
print("\nFull data matrix shape:")
array_4d = []
# Fill the 4D array
for i in range(data.shape[0]):
    inner_array = []
    for j in range(data.shape[1]):
        # Try to convert the object to a numpy array if it isn't already
        inner_array.append(np.array(data[i, j], dtype=float))
    array_4d.append(inner_array)
full_data = np.array(array_4d)
print(full_data.shape)
print("\nAvailability Mask shape:")
print(mask.shape)
print("\nAvailability Mask (1 if observed, 0 if missing, 2 if unavailable):")
print(mask)
