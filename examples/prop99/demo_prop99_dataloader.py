"""Demo: Loading and inspecting the Tax Burden on Tobacco dataset using the nearest_neighbors package.

Run this script from the root of the repository with:
    python examples/demo_prop99_data_loading.py
"""

from nearest_neighbors.datasets.dataloader_factory import NNData
import numpy as np

print("====== Tax Burden on Tobacco Data Demo ======")

# Show dataset help info
print("\nDataset Help Information:")
NNData.help("prop99")

# Load with default settings
print("\n====== Example 1: Default Parameters ======")
loader = NNData.create("prop99", seed=42)
data, mask = loader.process_data_scalar()

print("Observed Cigarette Consumption Matrix (with NaNs):")
print(data[:5, :5])  # Show only the first 5 states and 5 years to keep output manageable
print("\nMask Matrix:")
print(mask[:5, :5])

# Load with custom parameters
print("\n====== Example 2: Custom Parameters ======")
loader_custom = NNData.create("prop99", seed=42, 
                            start_year=1980, end_year=2000, 
                            sample_states=10)
data_custom, mask_custom = loader_custom.process_data_scalar()

print("Custom-Sized Consumption Matrix:")
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

# If metadata is available, print state names
if "metadata" in state and "state_names" in state["metadata"]:
    print("\nFirst 5 states in the dataset:")
    print(state["metadata"]["state_names"][:5])
    print("\nYears covered:")
    years = state["metadata"]["years"]
    print(f"{years[0]} to {years[-1]}")