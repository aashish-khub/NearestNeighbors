"""The following script demonstrates how to generate synthetic data using the nearest_neighbors package.
It shows how to explore the results of the synthetic data generation, and various ways in which the
data generation parameters can be customized. It also documents how parameters can be accessed from the
instance of the data generator, and how the full state dictionary can be accessed to see the true data
before noise, the noisy data, and the availability mask, etc.

To run this script, ensure you're in the root directory of the package and run:
```
    python examples/demo_synthetic_data_generation.py
```
"""

from nearest_neighbors.dataloader_factory import NNData

print("====== Synthetic Data Demo: Exploring Parameters and Full State Keys ======")
print("\nDataset Help Information:")
NNData.help("synthetic_data")

# Example 1: Multiplicative model (default) with custom parameters.
print("\n====== Example 1: Multiplicative Model (Default) ======")
data_generator = NNData.create(
    "synthetic_data", num_rows=10, num_cols=10, seed=29, miss_prob=0.2
)
data, mask = data_generator.process_data_scalar()

print("\nWe can see the produced data (multiplicative model) below:")
print("Observed Data:")
print(data)
print("\nAvailability Mask (1 if observed, 0 if missing):")
print(mask)

# Example 2: Additive model with lower noise and higher missing probability.
print("\n====== Example 2: Additive Model with Lower Noise ======")
data_generator2 = NNData.create(
    "synthetic_data",
    num_rows=10,
    num_cols=10,
    seed=29,
    miss_prob=0.3,
    latent_factor_combination_model="additive",
    stddev_noise=0.5,
)
data2, mask2 = data_generator2.process_data_scalar()

print("\nWe can see the produced data (additive model with lower noise) below:")
print("Observed Data:")
print(data2)
print("\nAvailability Mask (1 if observed, 0 if missing):")
print(mask2)

# Explore the keys of the full state dictionary.
print("\n====== Full State Dictionary Keys ======")
full_state = data_generator.get_full_state_as_dict(include_metadata=True)
print("Keys in the full state dictionary:")
print(list(full_state.keys()))
print(
    "For instance, we may want to see the true noisy data and the true data before noise:"
)
print("\nFirst row from 'full_data_noisy', with no missing values:")
print(full_state["full_data_noisy"][0])
print("\nFirst row from 'full_data_true', before the noise was added:")
print(full_state["full_data_true"][0])


print("\nFor debugging, we can also access the full generation metadata:")
gen_metadata = full_state["generation_metadata"]
for key, value in gen_metadata.items():
    print(f"\t{key}: {value}")
