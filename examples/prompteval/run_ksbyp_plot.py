import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from nsquared.utils import plotting_utils

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
df_mmd_col_03 = pd.read_csv(f"{output_dir}/results/est_errors-col-col-lbo-kernel_mmd-p0.3-tp4.0.csv")
df_mmd_col_05 = pd.read_csv(f"{output_dir}/results/est_errors-col-col-lbo-kernel_mmd-p0.5-tp4.0.csv")
df_mmd_col_07 = pd.read_csv(f"{output_dir}/results/est_errors-col-col-lbo-kernel_mmd-p0.7-tp4.0.csv")

df_wasserstein_03 = pd.read_csv(f"{output_dir}/results/est_errors-col-col-lbo-wasserstein_samples-p0.3-tp4.0.csv")
df_wasserstein_05 = pd.read_csv(f"{output_dir}/results/est_errors-col-col-lbo-wasserstein_samples-p0.5-tp4.0.csv")
df_wasserstein_07 = pd.read_csv(f"{output_dir}/results/est_errors-col-col-lbo-wasserstein_samples-p0.7-tp4.0.csv")

df_mmd_row_03 = pd.read_csv(f"{output_dir}/results/est_errors-row-row-lbo-kernel_mmd-p0.3-tp4.0.csv")
df_mmd_row_05 = pd.read_csv(f"{output_dir}/results/est_errors-row-row-lbo-kernel_mmd-p0.5-tp4.0.csv")
df_mmd_row_07 = pd.read_csv(f"{output_dir}/results/est_errors-row-row-lbo-kernel_mmd-p0.7-tp4.0.csv")

df_wasserstein_row_03 = pd.read_csv(f"{output_dir}/results/est_errors-row-row-lbo-wasserstein_samples-p0.3-tp4.0.csv")
df_wasserstein_row_05 = pd.read_csv(f"{output_dir}/results/est_errors-row-row-lbo-wasserstein_samples-p0.5-tp4.0.csv")
df_wasserstein_row_07 = pd.read_csv(f"{output_dir}/results/est_errors-row-row-lbo-wasserstein_samples-p0.7-tp4.0.csv")

df_mmd_col = pd.concat([df_mmd_col_03, df_mmd_col_05, df_mmd_col_07], ignore_index=True)
df_wasserstein_col = pd.concat([df_wasserstein_03, df_wasserstein_05, df_wasserstein_07], ignore_index=True)

propensity = np.array([0.3, 0.5, 0.7])

mmd_col_error_03 = np.array(df_mmd_col_03['est_errors'].values)
mmd_col_error_05 = np.array(df_mmd_col_05['est_errors'].values)
mmd_col_error_07 = np.array(df_mmd_col_07['est_errors'].values)

mmd_row_error_03 = np.array(df_mmd_row_03['est_errors'].values)
mmd_row_error_05 = np.array(df_mmd_row_05['est_errors'].values)
mmd_row_error_07 = np.array(df_mmd_row_07['est_errors'].values)

wasserstein_col_error_03 = np.array(df_wasserstein_03['est_errors'].values)
wasserstein_col_error_05 = np.array(df_wasserstein_05['est_errors'].values)
wasserstein_col_error_07 = np.array(df_wasserstein_07['est_errors'].values)

wasserstein_row_error_03 = np.array(df_wasserstein_row_03['est_errors'].values)
wasserstein_row_error_05 = np.array(df_wasserstein_row_05['est_errors'].values)
wasserstein_row_error_07 = np.array(df_wasserstein_row_07['est_errors'].values)

mmd_row_errors = np.array([np.nanmean(mmd_row_error_03), np.nanmean(mmd_row_error_05), np.nanmean(mmd_row_error_07)])
mmd_col_errors = np.array([np.nanmean(mmd_col_error_03), np.nanmean(mmd_col_error_05), np.nanmean(mmd_col_error_07)])
wasserstein_row_errors = np.array([np.nanmean(wasserstein_row_error_03), np.nanmean(wasserstein_row_error_05), np.nanmean(wasserstein_row_error_07)])
wasserstein_col_errors = np.array([np.nanmean(wasserstein_col_error_03), np.nanmean(wasserstein_col_error_05), np.nanmean(wasserstein_col_error_07)])

mmd_row_stderr = np.array([np.nanstd(mmd_row_error_03) / np.sqrt(mmd_row_error_03.shape), np.nanstd(mmd_row_error_05) / np.sqrt(mmd_row_error_05.shape), np.nanstd(mmd_row_error_07) / np.sqrt(mmd_row_error_07.shape)])
mmd_col_stderr = np.array([np.nanstd(mmd_col_error_03) / np.sqrt(mmd_col_error_03.shape), np.nanstd(mmd_col_error_05) / np.sqrt(mmd_col_error_05.shape), np.nanstd(mmd_col_error_07) / np.sqrt(mmd_col_error_07.shape)])
wasserstein_row_stderr = np.array([np.nanstd(wasserstein_row_error_03) / np.sqrt(wasserstein_row_error_03.shape), np.nanstd(wasserstein_row_error_05) / np.sqrt(wasserstein_row_error_05.shape), np.nanstd(wasserstein_row_error_07) / np.sqrt(wasserstein_row_error_07.shape)])
wasserstein_col_stderr = np.array([np.nanstd(wasserstein_col_error_03) / np.sqrt(wasserstein_col_error_03.shape), np.nanstd(wasserstein_col_error_05) / np.sqrt(wasserstein_col_error_05.shape), np.nanstd(wasserstein_col_error_07) / np.sqrt(wasserstein_col_error_07.shape)])

# Create the plot
plt.figure(figsize=(3.25, 2.5))
#figsize=(plotting_utils.NEURIPS_TEXTWIDTH / 2, 2.5)
# plt.plot(T_values, USVT_errors, 'r', linestyle='-', marker='D', markersize=8, linewidth=2, label=r'USVT: $T^{-0.46}$')


def _add_regression_line(
    x: np.ndarray, y: np.ndarray, color: str, label: str, linestyle: str
) -> float:
    # Fit a line to log-transformed data
    slope, intercept = np.polyfit(x, y, 1)
    plt.plot(
        x, slope*x + intercept, color=color, linestyle=linestyle, linewidth=2
    )
    # label=f"{label} (slope: {slope:.2f})")
    return slope


col_mmd_slope = _add_regression_line(propensity, mmd_col_errors, "blue", "", ":")
row_mmd_slope = _add_regression_line(propensity, mmd_row_errors, "magenta", "", ":")
col_wasserstein_slope = _add_regression_line(propensity, wasserstein_col_errors, "red", "", "--")
row_wasserstein_slope = _add_regression_line(propensity, wasserstein_row_errors, "green", "", "--")
# softimpute_slope = add_regression_line(T_values, Softimpute_errors, 'black', 'SoftImpute', ':')

def _format_lbl(method:str) -> str:
    return str(plotting_utils.METHOD_ALIASES_SINGLE_LINE.get(method, method))

plt.errorbar(
    propensity,
    mmd_col_errors,
    fmt="Db",
    yerr=mmd_col_stderr.squeeze(),
    markersize=5,
    linestyle="None",
    label=rf"{_format_lbl('kernel')} (col)",
)
plt.errorbar(
    propensity,
    mmd_row_errors,
    fmt="mo",
    yerr=mmd_row_stderr.squeeze(),
    markersize=5,
    linestyle="None",
    label=rf"{_format_lbl('kernel')} (row)",
)
plt.errorbar(
    propensity,
    wasserstein_col_errors,
    fmt="vr",
    yerr=wasserstein_col_stderr.squeeze(),
    markersize=5,
    linestyle="None",
    label=rf"{_format_lbl('wasserstein_samples')} (col)",
)
plt.errorbar(
    propensity,
    wasserstein_row_errors,
    fmt="^g",
    yerr=wasserstein_row_stderr.squeeze(),
    markersize=5,
    linestyle="None",
    label=rf"{_format_lbl('wasserstein_samples')} (row)",
)

# Axis labels
plt.xlabel(r"Propensity (p)", fontsize=plotting_utils.LABEL_FONT_SIZE)
plt.ylabel(r"Kolmogorov-Smirnov distance", fontsize=plotting_utils.LABEL_FONT_SIZE)

# Title
# plt.title(r'Decay of avg. error across users (N = T, 30 trials)', fontsize=16)

# Add legend
plt.legend(fontsize=plotting_utils.LEGEND_FONT_SIZE, loc="upper right")

# set tick font size
#print(plotting_utils.TICK_FONT_SIZE)
plt.tick_params(axis="both", which="major", labelsize=plotting_utils.LABEL_FONT_SIZE)

# set exact x ticks
plt.xticks(
    [0.3, 0.5, 0.7],
    fontsize=plotting_utils.LABEL_FONT_SIZE,
)
# Grid for better readability
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
ax1 = plt.gca()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.grid(True, alpha=0.4)
plt.tight_layout()
# Show the plot
plt.savefig(f"{output_dir}/ksbyp_prompteval_plot.pdf", bbox_inches="tight")
