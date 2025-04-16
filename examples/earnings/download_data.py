import wrds
from tqdm import tqdm
import pickle
import pandas as pd
import os
import numpy as np
from scipy.optimize import linear_sum_assignment


print("Connecting to WRDS...")
# Connect to WRDS
conn = wrds.Connection(wrds_username="jacobf1")

print("Reading US tickers...")
# Read the list of US tickers from the us_tickers.txt file
with open("us_tickers.txt", "r") as f:
    tickers = f.read().splitlines()

print("Getting I/B/E/S tickers...")

# Check if ibes_tickers.pkl already exists
if os.path.exists("ibes_tickers.pkl"):
    # Load the ibes_tickers.pkl file
    with open("ibes_tickers.pkl", "rb") as f:
        ibes_tickers = pickle.load(f)
else:
    # Get the earnings estimate data from the I/B/E/S database
    ibes_tickers = []
    for t in tqdm(tickers):
        try:
            data = conn.raw_sql(f"""SELECT ticker, oftic, cname, estimator, analys, FPI, MEASURE, VALUE, FPEDATS, ANNDATS, ANNTIMS, ACTUAL, ANNDATS_ACT, ANNTIMS_ACT
                                FROM tr_ibes.det_epsus
                                WHERE oftic = '{t}'
                                and anndats >= '01/01/2010'
                                and anndats <= '01/01/2025'
                                and fpi = '6'
                                """)
            if data.shape[0] == 0:
                continue
            ibes_ticker = data["ticker"].iloc[0]
            ibes_tickers.append((t, ibes_ticker))
        except Exception as _:
            continue

    # Save the ibes_tickers to a pkl file
    with open("ibes_tickers.pkl", "wb") as f:
        pickle.dump(ibes_tickers, f)

ibes_data = dict()
# Check if data folder exists
if not os.path.exists("data"):
    # Create the data folder
    os.makedirs("data")

print("Downloading I/B/E/S data...")
for oftic, ibes in tqdm(ibes_tickers):
    # Check if the data file already exists
    if os.path.exists(f"data/{oftic}.csv"):
        ibes_data[oftic] = pd.read_csv(f"data/{oftic}.csv")
    else:
        # Download the data
        ibes_data[oftic] = (
            conn.raw_sql(f"""SELECT ticker, oftic, cname, estimator, analys, FPI, MEASURE, VALUE, FPEDATS, ANNDATS, ANNTIMS, ACTUAL, ANNDATS_ACT, ANNTIMS_ACT
                            FROM tr_ibes.det_epsus
                            WHERE ticker = '{ibes}'
                            and anndats >= '01/01/2010'
                            and anndats <= '01/01/2025'
                            and fpi = '6'
                            """)
        )

        # Save data to csv
        ibes_data[oftic].to_csv(f"data/{oftic}.csv", index=False)

print("Calculating unique dates...")
unique_dates = dict()
max_length = 60
for t, data in ibes_data.items():
    dates = pd.to_datetime(data["anndats_act"].dropna().unique(), format="%Y-%m-%d")
    dates = np.array(list(dates) + [pd.NaT] * (max_length - len(dates)))
    unique_dates[t] = dates

print("Aligning dates...")
cost_matrices = []  # cost matrix for 'AAPL' to all other tickers

base_date = unique_dates["AAPL"]

aligned_dates = dict()

for t, d in unique_dates.items():
    cost_matrix = []
    for d1 in base_date:
        row = []
        for d2 in d:
            row.append(
                abs((d1 - d2).days) if pd.notna(d1) and pd.notna(d2) else 365 * 5 * 2
            )
        cost_matrix.append(row)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
    aligned_dates[t] = d[col_ind]

print("Creating dictionary of dataframes...")
if os.path.exists("quarterly_actual.pkl") and os.path.exists("quarterly_data.pkl"):
    with open("quarterly_actual.pkl", "rb") as f:
        quarterly_actual = pickle.load(f)

    with open("quarterly_data.pkl", "rb") as f:
        quarterly_data = pickle.load(f)

else:
    # since we start at January 1, 2010, we can assume that the first date is the first quarter of 2010
    quarterly_data = dict()
    quarterly_actual = dict()  # actual value, announcement date, announcement time
    for oftic, data in tqdm(ibes_data.items()):
        data = data.dropna(axis=0, subset=["anndats_act"])
        starting_year = 2010  # we start at January 1, 2010
        for i, date in enumerate(aligned_dates[oftic]):
            quarter_num = (i % 4) + 1  # 1,2,3,4
            year = starting_year + (i // 4)
            if date is pd.NaT:
                quarterly_actual[oftic, year, quarter_num] = None
                quarterly_data[oftic, year, quarter_num] = None
                continue
            date = date.strftime("%Y-%m-%d")
            subdata = data[data["anndats_act"] == date].copy()

            quarterly_actual[oftic, year, quarter_num] = (
                subdata["actual"].iloc[0],
                subdata["anndats_act"].iloc[0],
                subdata["anntims_act"].iloc[0],
            )
            subdata["ann_datetime"] = pd.to_datetime(
                subdata["anndats"] + " " + subdata["anntims"],
                format="%Y-%m-%d %H:%M:%S",
            )

            quarterly_data[oftic, year, quarter_num] = subdata[
                ["value", "ann_datetime"]
            ]

    with open("quarterly_actual.pkl", "wb") as f:
        pickle.dump(quarterly_actual, f)

    with open("quarterly_data.pkl", "wb") as f:
        pickle.dump(quarterly_data, f)
