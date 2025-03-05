"""Script to structure the HeartSteps data.
Ref: https://github.com/calebchin/DistributionalNearestNeighbors/blob/main/experiments/structure_hs_data.ipynb
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import argparse
from typing import Any
import warnings

def _transform_dnn(
    df : Any,
    users: int = 37,
    max_study_day: int = 52,
    day_dec: int = 5,
    num_measurements: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively transform the processed HeartSteps data into a 4d tensor"""
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    final_M = np.zeros((users, max_study_day, day_dec, num_measurements))
    final_A = np.zeros((users, max_study_day, day_dec))

    for user in range(1, users + 1):
        for day in range(1, max_study_day + 1):
            for slot in range(1, day_dec + 1):
                try:
                    df_uds = df.loc[pd.IndexSlice[(user, day, slot)]]
                    ind = df.index.get_indexer_for(df_uds.index)[0]
                    df_rng = df.iloc[
                        np.arange(ind, min(ind + num_measurements, len(df)))
                    ]
                    if df.iloc[ind]["avail"]:
                        # only take send.sedentary as the treatment indicator, could use send.active later
                        val = df.iloc[ind]["send.sedentary"]
                        conv_val = val if ~np.isnan(val) else df.iloc[ind]["send"]
                        final_A[user - 1, day - 1, slot - 1] = int(conv_val)
                    else:
                        final_A[user - 1, day - 1, slot - 1] = 2

                    measurements = df_rng["steps"].to_numpy()
                    if len(measurements) == num_measurements:
                        final_M[user - 1, day - 1, slot - 1] = measurements
                    else:
                        m_pad = np.pad(
                            measurements,
                            (0, num_measurements - len(measurements)),
                            constant_values=np.nan,
                        )
                        final_M[user - 1, day - 1, slot - 1] = m_pad
                except KeyError as e:

                    final_A[user - 1, day - 1, slot - 1] = 0
                    final_M[user - 1, day - 1, slot - 1] = np.full(
                        num_measurements, np.nan
                    )
    final_M = final_M.reshape((users, max_study_day * day_dec, num_measurements))
    final_A = final_A.reshape((users, max_study_day * day_dec))
    return final_M, final_A


def _get_mode(x: pd.Series) -> pd.Series:
    if len(pd.Series.mode(x) > 1):
        return pd.Series.mode(x, dropna=False)[0]
    else:
        return pd.Series.mode(x, dropna=False)


def _reind_id(df_u: pd.DataFrame) -> pd.DataFrame:
    """Function to reindex the data to include all time points for each user."""
    rng = pd.date_range(
        min(df_u.index.astype("datetime64[ns]")),
        max(df_u.index.astype("datetime64[ns]")) + timedelta(days=1),
        normalize=True,
        inclusive="both",
        freq="5min",
    )
    rng = rng[rng.indexer_between_time("00:00", "23:55")]
    # print(rng)
    df_reind = df_u.reindex(rng)
    df_reind["user.index"] = df_reind["user.index"].ffill().bfill()
    df_reind["study.day.nogap"] = df_reind["study.day.nogap"].bfill().ffill()
    df_reind["steps"] = df_reind["steps"].fillna(0)
    return df_reind


def _take_range(df: pd.DataFrame, range: int) -> pd.DataFrame:
    idx = df.index.get_indexer_for(
        df[pd.notna(df["sugg.select.slot"])].index
    )
    ranges = [np.arange(i, min(i + range + 1, len(df))) for i in idx]
    return df.iloc[np.concatenate(ranges)]


def _create_slots(df: pd.DataFrame) -> pd.DataFrame:
    most_rec_slot = 0.0
    for ind, row in df.iterrows():
        curr_slot = row["sugg.select.slot"]
        if not np.isnan(curr_slot):
            if (
                most_rec_slot != 5.0
                and curr_slot != most_rec_slot + 1
                and most_rec_slot != 0.0
            ):
                df.at[ind, "new_slot"] = most_rec_slot + 1
                most_rec_slot += 1
            elif most_rec_slot == 5.0 and curr_slot != 1:
                df.at[ind, "new_slot"] = 1
                most_rec_slot = 1
            else:
                df.at[ind, "new_slot"] = curr_slot
                most_rec_slot = curr_slot
    return df


def _study_day(df: pd.DataFrame) -> pd.DataFrame:
    most_rec_slot = 1.0
    curr_study_day = 1
    for ind, row in df.iterrows():
        curr_slot = row["new_slot"]
        if not np.isnan(curr_slot):
            if most_rec_slot == 5.0 and curr_slot == 1:
                curr_study_day += 1
            most_rec_slot = curr_slot
            df.at[ind, "study_day"] = curr_study_day
    return df


def preprocess_heartsteps(
    data_dir: str = "", output_dir: str = "", download: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the HeartSteps V1 data. The data is returned in a N x T x n x d ndarray where the N is the number of participants, T is the number of decision/time points, n is the number of measurements per entry, and d is the dimension of each measurement. ]
    Each entry is an aggregated step count of one hour after a notification decision period.
    All participants are aligned to have the same start date and each have 5 decision points per day. The mask is returned as a N x T ndarray, where 1 indicates a notifcation was sent at that decision point, 0 indicates no notification was sent, and 2 indicates that the participant was unavilable.

    Args:
    ----
    data_dir: str
        Directory containing the raw HeartSteps data (jbsteps.csv and suggestions.csv). If None, download must be True.
    output_dir: str
        Directory to save the processed data. If None, the processed data is not saved.
    download: bool
        Indicates if the csv files necessary for the data processing should be downloaded from a separate repo (True) or are already locally downloaded (False).
        If True, data_dir must be None.

    Returns:
    -------
    Data : ndarray TODO: sort out whether to return ndarray of objects or floats
        N x T x n x d ndarray
    Data2d : ndarray
        N x T ndarray of objects
    Mask : ndarray
        Binary N x T ndarray, where 1 indicates a notifcation was sent at that decision point and 0 indicates no notification was sent.

    """
    # TODO: address deprecated warnings
    # print(data_dir != "" and not download)
    assert (data_dir == "" and download) or (data_dir != "" and not download), (
        "data_dir must be specified if download is False"
    )
    assert os.path.exists(data_dir), "Provided data_dir does not exist"
    assert os.path.exists(os.path.join(data_dir, "jbsteps.csv")), (
        "jbsteps.csv not found in data_dir"
    )
    assert os.path.exists(os.path.join(data_dir, "suggestions.csv")), (
        "suggestions.csv not found in data_dir"
    )
    assert output_dir == "" or os.path.exists(output_dir), (
        "Provided output_dir does not exist"
    )

    if download:
        # TODO: add download code
        df_steps = pd.DataFrame()
        df_suggestions = pd.DataFrame()
        pass
    else:
        df_steps = pd.read_csv(os.path.join(data_dir, "jbsteps.csv"), low_memory=False)
        df_suggestions = pd.read_csv(os.path.join(data_dir, "suggestions.csv"), low_memory=False)

    print("Processing HeartSteps V1...")
    # get relevant cols and reformatting
    df_steps = df_steps[["user.index", "steps.utime", "steps", "study.day.nogap"]]
    df_steps["steps.utime"] = pd.to_datetime(df_steps["steps.utime"])
    # create multi-index
    df_steps = df_steps.set_index(["user.index", "steps.utime"])

    # get relevant cols and reformatting
    df_sugg_sel = df_suggestions[
        [
            "user.index",
            "decision.index.nogap",
            "sugg.select.utime",
            "sugg.decision.utime",
            "sugg.select.slot",
            "avail",
            "send",
            "send.active",
            "send.sedentary",
        ]
    ]
    df_sugg_sel = df_sugg_sel.copy()
    df_sugg_sel["sugg.decision.utime"] = pd.to_datetime(
        df_sugg_sel["sugg.decision.utime"]
    )
    df_sugg_sel = df_sugg_sel.dropna(
        subset=["sugg.decision.utime", "sugg.select.utime", "user.index"]
    )

    # group the step data by five minute intervals
    df_5min = (
        df_steps.groupby(
            [
                pd.Grouper(freq="5min", level="steps.utime", label="right"),
                pd.Grouper(level="user.index"),
            ],
            sort=False,
        )
        .agg({"steps": "sum", "study.day.nogap": lambda x: _get_mode(x)})
        .reset_index()
    )

    df_5min_ind = df_5min.set_index("steps.utime")

    # expand the step data to include all time points
    # df_expand5min = df_5min_ind.groupby("user.index", group_keys=False).apply(
    #     lambda df_u: _reind_id(df_u)
    # )

    result_dfs = []
    user_indices = df_5min_ind["user.index"].unique()
    # process each user 
    for user_idx in user_indices:
        # filter for just this user
        df_u = df_5min_ind[df_5min_ind["user.index"] == user_idx].copy()
        reindexed_df = _reind_id(df_u)
        result_dfs.append(reindexed_df)
    df_expand5min = pd.concat(result_dfs)

    df_expand5min = df_expand5min.reset_index(names="steps.utime")
    df_expand5min["user.index"] = df_expand5min["user.index"].astype("int64")

    # merge the step data with the notification data
    df_merged = (
        pd.merge_asof(
            df_expand5min.sort_values(by="steps.utime"),
            df_sugg_sel.sort_values(by="sugg.decision.utime"),
            left_on="steps.utime",
            right_on="sugg.decision.utime",
            by="user.index",
            tolerance=pd.Timedelta("5min"),
            allow_exact_matches=False,
            direction="backward",
        )
        .sort_values(by=["user.index", "steps.utime"])
        .reset_index(drop=True)
    )
    df_merged["sugg.select.slot"] = np.where(
        df_merged["decision.index.nogap"].isna(), np.nan, df_merged["sugg.select.slot"]
    )

    # get 12 rows after each notification period (1 hour of observations)
    unique_users = df_merged["user.index"].unique()
    result_dfs = []

    for user_idx in unique_users:
        df_user = df_merged[df_merged["user.index"] == user_idx]
        df_user_range = _take_range(df_user, 12)
        result_dfs.append(df_user_range)

    df_merged_cut = pd.concat(result_dfs).reset_index(drop=True)
    df_merged_cut_nd = df_merged_cut.drop_duplicates()

    # set up column for study day
    df_merged_cut_nd = df_merged_cut_nd.copy()
    df_merged_cut_nd["study_day"] = np.nan
    df_merged_cut_nd["new_slot"] = np.nan

    # align decision points
    unique_users = df_merged_cut_nd["user.index"].unique()
    slot_results = []

    for user_idx in unique_users:
        df_user = df_merged_cut_nd[df_merged_cut_nd["user.index"] == user_idx]
        user_slots = _create_slots(df_user)
        slot_results.append(user_slots)

    df_slot = pd.concat(slot_results)

    # Second groupby operation: study_day
    unique_users_slot = df_slot["user.index"].unique()
    study_day_results = []

    for user_idx in unique_users_slot:
        df_user = df_slot[df_slot["user.index"] == user_idx]
        user_study_day = _study_day(df_user)
        study_day_results.append(user_study_day)

    df_study_day = pd.concat(study_day_results)

    # create unique index
    df_final = df_study_day.set_index(["user.index", "study_day", "new_slot"])

    # transform into 4d tensor + mask
    Data, Mask = _transform_dnn(df_final)
    N, T = Mask.shape
    Data2d = np.empty([N, T], dtype=object)

    # to align with 4d structure
    Data = Data[:, :, :, np.newaxis]

    for i in range(N):
        for j in range(T):
            Data2d[i, j] = Data[i, j]
    # write data to output_dir
    if output_dir != "":
        np.save(os.path.join(output_dir, "data.npy"), Data2d, allow_pickle=True)
        np.save(os.path.join(output_dir, "mask.npy"), Mask, allow_pickle=True)

    return Data, Data2d, Mask


def preprocess_heartsteps_scalar(
    data_dir: str = "", output_dir: str = "", agg: str = "mean", download: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess the HeartSteps V1 data. The data is returned in a N x T ndarray where the rows represent participants and the columns represent time points and each entry is an aggregated step count of one hour after a notification decision period.
    All participants are aligned to have the same start date and each have 5 decision points per day. The mask is returned as a binary N x T ndarray, where 1 indicates a notifcation was sent at that decision point and 0 indicates no notification was sent.

    Args:
    ----
    data_dir: str
        Directory containing the raw HeartSteps data (jbsteps.csv and suggestions.csv). If None, download must be True.
    output_dir: str
        Directory to save the processed data. If empty, the processed data is not saved.
    agg: str
        Aggregation method to use. Options are "mean", "sum", "median", "std", and "variance".
    download: bool
        Indicates if the csv files necessary for the data processing should be downloaded from a separate repo (True) or are already locally downloaded (False).
        If True, data_dir must be None.

    Returns:
    -------
    Data : ndarray
        N x T ndarray where the rows represent participants and the columns represent time points and each entry is an aggregated step count of one hour after a notification decision period.
    Mask : ndarray
        Binary N x T ndarray, where 1 indicates a notifcation was sent at that decision point and 0 indicates no notification was sent.

    """
    Data, Data2d, Mask = preprocess_heartsteps(
        data_dir, output_dir="", download=download
    )
    if agg == "mean":
        Data = Data.mean(axis=2)
    elif agg == "sum":
        Data = Data.sum(axis=2)
    elif agg == "median":
        Data = np.median(Data, axis=2)
    elif agg == "std":
        Data = np.std(Data, axis=2)
    elif agg == "variance":
        Data = np.var(Data, axis=2)
    else:
        raise ValueError(
            "agg must be one of 'mean', 'sum', 'median', 'std', or 'variance'"
        )

    Data = np.squeeze(Data)
    Data = Data.astype(object)
    # write data to output_dir
    if output_dir != "":
        np.save(os.path.join(output_dir, "data.npy"), Data, allow_pickle=True)
        np.save(os.path.join(output_dir, "mask.npy"), Mask, allow_pickle=True)

    return Data, Mask


def main(args: argparse.Namespace) -> None:
    """Main function to preprocess the HeartSteps data."""
    assert args.type in ["scalar", "distributional"], (
        "type must be one of 'scalar' or 'distributional'"
    )
    assert args.agg in ["mean", "sum", "median", "std", "variance"], (
        "agg must be one of 'mean', 'sum', 'median', 'std', or 'variance'"
    )
    if args.type == "scalar":
        preprocess_heartsteps_scalar(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            agg=args.agg,
            download=args.download,
        )
    elif args.type == "distributional":
        preprocess_heartsteps(
            data_dir=args.data_dir, output_dir=args.output_dir, download=args.download
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--type", default="scalar", type=str, help="Data Type: distributional or scalar"
    )
    parser.add_argument(
        "--agg",
        default="mean",
        type=str,
        help="Aggregation method for scalar data: mean, sum, median, std, or variance",
    )
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="Directory containing the raw HeartSteps data (jbsteps.csv and suggestions.csv). If empty, --download must be True.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Directory to save the processed data. If empty, the processed data is not saved.",
    )
    parser.add_argument(
        "--download",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use --download if no local csvs, otherwise use --no-download",
    )

    args = parser.parse_args()

    main(args)
