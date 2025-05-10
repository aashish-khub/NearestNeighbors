"""Demo: Loading and inspecting the PromptEvaldataset using the nearest_neighbors package.

Run this script from the root of the repository with:
    python examples/demo_prompteval_dataloader.py
"""

from nearest_neighbors.datasets.dataloader_factory import NNData

print("====== PromptEval Data Demo ======")

# Show dataset help info
print("\nDataset Help Information:")
NNData.help("prompteval")

# Load with default settings
print("\n====== Example 1: Default Parameters ======")
loader = NNData.create("prompteval", seed=42)
data, mask = loader.process_data_scalar()

print("Observed Ratings Matrix (with NaNs):")
print(data)
print("\nMask Matrix (True = rating present, False = missing):")
print(mask)

# Load with custom sampling
print("\n====== Example 2: Custom Parameters ======")
loader_custom = NNData.create("movielens", seed=42, sample_users=100, sample_movies=50)
data_custom, mask_custom = loader_custom.process_data_scalar()

print("Custom-Sized Ratings Matrix:")
print(data_custom)
print("\nCustom Mask Matrix:")
print(mask_custom)

# View saved internal state
print("\n====== Full State Dictionary Keys ======")
state = loader.get_full_state_as_dict(include_metadata=True)
print("Top-level keys:")
print(list(state.keys()))
print("\nCustom parameters used:")
print(state.get("custom_params", {}))
