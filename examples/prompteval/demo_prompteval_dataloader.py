"""The following script demonstrates the usage of the PromptEval dataloader.

The PromptEval dataset comes from https://github.com/kyuseongchoi5/EfficientEval_BayesOpt and is data from a study on the efficient evaluation of LLMs across a variety of benchmarks, models and prompt templates.
The data can be processed into the distributional setting (4d tensor) or the scalar setting (2d matrix). The dimensions of the tensor are N x T x n x 1, where N is the number of benchmarks,
T is the number of models(15), and n is the number of prompt templates(100). Each entries are the average correctness of the model (across all examples) on the benchmark and prompt template.

To run this script, ensure you're in the root directory of the package and run:
```
    python examples/prompteval/demo_prompteval_dataloader.py
```
"""

from nearest_neighbors.datasets.dataloader_factory import NNData
# import numpy as np

print("====== PromptEval Data Demo ======")
print("\nPromptEval Help Information:")
NNData.help("prompteval")

# Example: Load the PromptEval dataset with default parameters.
print("===== Loading PromptEval dataset with default params in distributional form ===")
# Define the models and tasks
models = ['meta_llama_llama_3_8b', 'meta_llama_llama_3_8b_instruct', 'meta_llama_llama_3_70b_instruct', 'codellama_codellama_34b_instruct', 
          'google_flan_t5_xl', 'google_flan_t5_xxl', 'google_flan_ul2', 'ibm_mistralai_merlinite_7b', 'mistralai_mixtral_8x7b_instruct_v01', 
          'mistralai_mistral_7b_instruct_v0_2', 'google_gemma_7b', 'google_gemma_7b_it', 'tiiuae_falcon_40b', 'mistralai_mistral_7b_v0_1', 'tiiuae_falcon_180b']
tasks = ['college_mathematics', 'miscellaneous', 'moral_disputes', 'jurisprudence', 'moral_scenarios', 'college_chemistry',
         'professional_medicine', 'clinical_knowledge', 'abstract_algebra', 'nutrition', 'professional_psychology', 'high_school_government_and_politics',
         'high_school_us_history', 'high_school_chemistry', 'high_school_macroeconomics', 'management', 'conceptual_physics', 'philosophy',
         'electrical_engineering', 'high_school_psychology', 'medical_genetics', 'high_school_geography',
         'high_school_statistics', 'international_law', 'elementary_mathematics', 'high_school_physics', 'world_religions',
         'high_school_european_history', 'formal_logic', 'security_studies', 'sociology', 'high_school_biology', 'us_foreign_policy',
         'high_school_microeconomics', 'college_medicine', 'college_computer_science', 'logical_fallacies', 'high_school_computer_science',
         'anatomy', 'econometrics', 'astronomy', 'college_biology', 'virology', 'professional_accounting',
         'college_physics', 'high_school_world_history', 'business_ethics', 'global_facts', 'public_relations',
         'marketing', 'human_aging', 'professional_law', 'high_school_mathematics', 'prehistory', 'machine_learning',
         'computer_security', 'human_sexuality']

data_generator = NNData.create("prompteval", models=models, tasks=tasks)
data, mask = data_generator.process_data_distribution()

model = models[0]
task = tasks[0]

print(
    "\n===== Loading PromptEval dataset with specific model and task in scalar form ==="
)

custom_data_generator = NNData.create(
    "prompteval", models=[model], tasks=[task], propensity=0.5, seed = 42
)
data, mask = custom_data_generator.process_data_scalar()

print("\nData matrix shape:")
print(data.shape)
print("\nAvailability Mask shape:")
print(mask.shape)



# print(
#     "\n===== Loading PromptEval dataset with default params in distributional form ==="
# )
# # print("\n Custom params: freq='1min', participants=25, num_measurements=60")
# custom_data_generator = NNData.create(
#     "prompteval", models=models, tasks=tasks
# )
# data, mask = custom_data_generator.process_data_distribution()

# print("\nData matrix shape:")
# print(data.shape)
# print("\nAvailability Mask shape:")
# print(mask.shape)
