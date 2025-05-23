"""Script to run the SyntheticControl package on the California smoking data.

TODO: add paper reference

Adapted from: https://github.com/OscarEngelbrektson/SyntheticControlMethods/blob/master/examples/proposal_99.py
"""

# Import packages
import pandas as pd
import os
import numpy as np
import logging

from SyntheticControlMethods import Synth

from nsquared.utils.experiments import get_base_parser, setup_logging

parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir
log_level = args.log_level

setup_logging(log_level)
logger = logging.getLogger(__name__)

# Import data
data_dir = "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/"
df = pd.read_csv(data_dir + "smoking_data" + ".csv")

STATE_CODES = {
    "Alabama": "AL",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Georgia": "GA",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Mexico": "NM",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

for state in df["state"].unique():
    # Fit Differenced Synthetic Control
    sc = Synth(df, "cigsale", "state", "year", 1989, state, n_optim=10, pen="auto")  # type: ignore

    logger.info(sc.original_data.weight_df)
    logger.info(sc.original_data.comparison_df)
    logger.info(sc.original_data.pen)

    # save using the state code to ensure compatibility with the rest of the codebase
    state_dir = os.path.join(output_dir, "sc", STATE_CODES[state])
    os.makedirs(state_dir, exist_ok=True)
    state_save_path = os.path.join(state_dir, f"sc-{STATE_CODES[state]}-auto.csv")
    control = (
        sc.original_data.synth_outcome.flatten()
        if sc.original_data.synth_outcome is not None
        else None
    )
    obs = (
        sc.original_data.treated_outcome_all.flatten()
        if sc.original_data.treated_outcome_all is not None
        else None
    )
    df_synthetic_control = pd.DataFrame(
        data={
            "control": control,
            "obs": obs,
            "est_errors": np.abs(control - obs)
            if control is not None and obs is not None
            else None,
            "estimation_method": "sc",
            "fit_method": "auto",
            "state": STATE_CODES[state],
        },
        index=range(1970, 2001),
    )
    logger.info(f"Saving to {state_save_path}...")
    df_synthetic_control.to_csv(state_save_path)
