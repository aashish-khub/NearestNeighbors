"""The following script demonstrates the usage of the PromptEval dataloader.

The PromptEval dataset comes from https://huggingface.co/PromptEval. The data consists of performance (correctness) evaluations of 15 LLMs on
57 tasks (like abstract algebra, professional law, etc.) using 100 different prompt templates. There are a variable number of questions asked to each LLM across tasks. 

To run this script, ensure you're in the root directory of the package and run:
```
    python examples/datasets/prompteval/demo_prompteval_dataloader.py
```
"""

from nearest_neighbors.dataloader_factory import NNData
import numpy as np

print("====== PromptEval Data Demo ======")
print("\PromptEval Help Information:")
NNData.help("prompteval")

# Example: Load the PromptEval dataset with default parameters.
print("===== Loading PromptEval dataset with default params in distributional form ===")
data_generator = NNData.create("prompteval")
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
    "\n===== Loading PromptEval dataset with default params in scalar form ==="
)
data, mask = data_generator.process_data_scalar()

print("\nData matrix shape:")
print(data.shape)
print("\nAvailability Mask shape:")
print(mask.shape)
print("\nMCAR Availability Mask (1 if observed, 0 if missing):")
print(mask)
