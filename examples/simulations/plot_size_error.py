import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Define T values and corresponding error rates for each method
T_values = np.array([2**4, 2**5, 2**6, 2**7])


# Define T values and corresponding error rates for each method
T_values = np.array([2**4, 2**5, 2**6, 2**7])


parser = argparse.ArgumentParser(description="Plot estimation errors")
parser.add_argument("--num_sims", type=int, default=30, help="Number of simulations")
parser.add_argument("--output_dir", type=str, help="Output directory")
args = parser.parse_args()
# Define T values (sizes)
T_values = np.array([2**4, 2**5, 2**6, 2**7])
num_sims = args.num_sims
output_dir = args.output_dir


# Function to process a single CSV file and extract errors by size
def _process_csv(filepath: str) -> np.ndarray:
    # Read the CSV
    df = pd.read_csv(filepath)

    # Create an array to store the averaged errors for each size and simulation
    # Shape: (4 sizes, 20 simulations)
    averaged_errors = np.zeros((len(T_values), num_sims))

    for i, size in enumerate(T_values):
        for j in range(num_sims):
            # Filter rows for this simulation
            size_rows = df[(df["sim_num"] == j) & (df["size"] == size)]
            size_data = size_rows.copy()
            averaged_errors[i, j] = np.nanmean(size_data["est_errors"])

    return averaged_errors


# Process each CSV file
col_err = _process_csv(f"{output_dir}/results/est_errors-col-col-lbo.csv")
row_err = _process_csv(f"{output_dir}/results/est_errors-row-row-lbo.csv")
# usvt_err = process_csv(f"{output_dir}/results/est_errors-usvt-lbo.csv")
drnn_err = _process_csv(f"{output_dir}/results/est_errors-dr-lbo.csv")
tsnn_err = _process_csv(f"{output_dir}/results/est_errors-ts-lbo.csv")
auto_err = _process_csv(f"{output_dir}/results/est_errors-auto-lbo.csv")
#softimpute_err = process_csv("cvrange_30sims/results/est_errors-softimpute-lbo.csv")

# Extract errors for each method as a numpy array 4 x 30

# USVT_errors = 2**-3 * T_values**-0.10
UserNN_errors = np.nanmean(row_err, axis=1)
TimeNN_errors = np.nanmean(col_err, axis=1)
DRNN_errors = np.nanmean(drnn_err, axis=1)
# USVT_errors = np.nanmean(usvt_err, axis = 1)
TSNN_errors = np.nanmean(tsnn_err, axis=1)
# Softimpute_errors = np.nanmean(softimpute_err, axis = 1)
Auto_errors = np.nanmean(auto_err, axis=1)


unn_stderr = np.nanstd(row_err, axis=1) / np.sqrt(row_err.shape[1])
tnn_stderr = np.nanstd(col_err, axis=1) / np.sqrt(col_err.shape[1])
drnn_stderr = np.nanstd(drnn_err, axis=1) / np.sqrt(drnn_err.shape[1])
# usvt_stderr = np.nanstd(usvt_err, axis = 1) / np.sqrt(usvt_err.shape[1])
tsnn_stderr = np.nanstd(tsnn_err, axis=1) / np.sqrt(tsnn_err.shape[1])
auto_stderr = np.nanstd(auto_err, axis=1) / np.sqrt(auto_err.shape[1])
# softimpute_stderr = np.nanstd(softimpute_err, axis = 1) / np.sqrt(softimpute_err.shape[1])


# Create the plot
plt.figure()
# plt.plot(T_values, USVT_errors, 'r', linestyle='-', marker='D', markersize=8, linewidth=2, label=r'USVT: $T^{-0.46}$')


def _add_regression_line(
    x: np.ndarray, y: np.ndarray, color: str, label: str, linestyle: str
) -> float:
    # Fit a line to log-transformed data
    slope, intercept = np.polyfit(np.log2(x), np.log2(y), 1)
    plt.plot(
        x, 2 ** (intercept) * x**slope, color=color, linestyle=linestyle, linewidth=2
    )
    # label=f"{label} (slope: {slope:.2f})")
    return slope


# usvt_slope = add_regression_line(T_values, USVT_errors, 'red', 'USVT', '--')
unn_slope = _add_regression_line(T_values, UserNN_errors, "green", "User-NN", ":")
tnn_slope = _add_regression_line(T_values, TimeNN_errors, "orange", "Time-NN", ":")
drnn_slope = _add_regression_line(T_values, DRNN_errors, "blue", "DR-NN", "--")
tsnn_slope = _add_regression_line(T_values, TSNN_errors, "purple", "Time-NN", ":")
auto_slope = _add_regression_line(T_values, Auto_errors, "orange", "Auto", ":")
# softimpute_slope = add_regression_line(T_values, Softimpute_errors, 'black', 'SoftImpute', ':')

# Plot each method with corresponding markers, colors, and line styles
# plt.errorbar(T_values, USVT_errors, yerr=usvt_stderr, fmt = 'r', marker='>', markersize=12, linestyle='None', barsabove=True, label=rf'USVT: $T^{{{usvt_slope:.2f}}}$')
plt.errorbar(
    T_values,
    UserNN_errors,
    yerr=unn_stderr,
    fmt="g",
    marker="o",
    markersize=12,
    linestyle="None",
    label=rf"Row-NN: $T^{{{unn_slope:.2f}}}$",
)
plt.errorbar(
    T_values,
    TimeNN_errors,
    yerr=tnn_stderr,
    fmt="s",
    color="orange",
    marker="s",
    linestyle="None",
    markersize=12,
    label=rf"Col-NN: $T^{{{tnn_slope:.2f}}}$",
)
plt.errorbar(
    T_values,
    DRNN_errors,
    fmt="bD",
    yerr=drnn_stderr,
    markersize=12,
    linestyle="None",
    label=rf"DR-NN: $T^{{{drnn_slope:.2f}}}$",
)
plt.errorbar(
    T_values,
    TSNN_errors,
    fmt="p",
    yerr=tsnn_stderr,
    markersize=12,
    linestyle="None",
    label=rf"TS-NN: $T^{{{tsnn_slope:.2f}}}$",
)
plt.errorbar(
    T_values,
    Auto_errors,
    fmt="o",
    yerr=auto_stderr,
    markersize=12,
    linestyle="None",
    label=rf"Auto-NN: $T^{{{auto_slope:.2f}}}$",
)
# plt.errorbar(T_values, Softimpute_errors, fmt='k^', yerr=softimpute_stderr, markersize=12, linestyle="None", label=rf'SoftImpute: $T^{{{softimpute_slope:.2f}}}$')
# Logarithmic scale for both axes
plt.xscale("log", base=2)
plt.yscale("log", base=2)

# Axis labels
plt.xlabel(r"T", fontsize=15)
plt.ylabel(r"Error for t = T, a = 1", fontsize=15)

# Title
# plt.title(r'Decay of avg. error across users (N = T, 30 trials)', fontsize=16)

# Add legend
plt.legend(fontsize=12)

# Grid for better readability
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
ax1 = plt.gca()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.grid(True, alpha=0.4)

# Show the plot
plt.savefig(f"{output_dir}/sims_plot.pdf", bbox_inches="tight")
