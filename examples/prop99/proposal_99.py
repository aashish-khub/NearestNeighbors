"""Script to run the SyntheticControl package on the California smoking data.

TODO: add paper reference

Adapted from: https://github.com/OscarEngelbrektson/SyntheticControlMethods/blob/master/examples/proposal_99.py
"""

# Import packages
import pandas as pd
import os

from SyntheticControlMethods import Synth

from nearest_neighbors.utils.experiments import get_base_parser

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir

# Import data
data_dir = "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/"
df = pd.read_csv(data_dir + "smoking_data" + ".csv")

# Fit Differenced Synthetic Control
sc = Synth(df, "cigsale", "state", "year", 1989, "California", n_optim=10, pen="auto")  # type: ignore

print(sc.original_data.weight_df)
print(sc.original_data.comparison_df)
print(sc.original_data.pen)


california_dir = os.path.join(output_dir, "california")
os.makedirs(california_dir, exist_ok=True)
california_save_path = os.path.join(california_dir, "california-sc-auto.csv")
df = pd.DataFrame(
    data={
        "control": None
        if sc.original_data.synth_outcome is None
        else sc.original_data.synth_outcome.flatten(),
        "obs": None
        if sc.original_data.treated_outcome_all is None
        else sc.original_data.treated_outcome_all.flatten(),
        "estimation_method": "sc",
        "fit_method": "auto",
    },
    index=range(1970, 2001),
)
print(f"Saving df_california to {california_save_path}...")
df.to_csv(california_save_path)

# Visualize
sc.plot(
    ["original", "pointwise", "cumulative"],
    treated_label="California",
    synth_label="Synthetic California",
    treatment_label="Proposal 99",
)


# In-time placebo
# Placebo treatment period is 1982, 8 years earlier
sc.in_time_placebo(1982)

# Visualize
sc.plot(
    ["in-time placebo"], treated_label="California", synth_label="Synthetic California"
)

# Compute in-space placebos
sc.in_space_placebo(1)

sc.original_data.rmspe_df.to_csv("rmspe_df.csv")  # type: ignore

# Visualize
sc.plot(["rmspe ratio"], treated_label="California", synth_label="Synthetic California")
sc.plot(
    ["in-space placebo"],
    in_space_exclusion_multiple=5,
    treated_label="California",
    synth_label="Synthetic California",
)
