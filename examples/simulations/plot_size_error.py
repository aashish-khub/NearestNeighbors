import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Define T values and corresponding error rates for each method
T_values = np.array([2**4, 2**5, 2**6, 2**7])


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define T values (sizes)
T_values = np.array([2**4, 2**5, 2**6, 2**7])

# Function to process a single CSV file and extract errors by size
def process_csv(filepath):
    # Read the CSV
    df = pd.read_csv(filepath)
    
    # Create an array to store the averaged errors for each size and simulation
    # Shape: (4 sizes, 20 simulations)
    averaged_errors = np.zeros((len(T_values), 30))
    
    for i, size in enumerate(T_values):
        # Filter rows for this size
        size_rows = df[df['size'] == size]
        
        # Group by simulation run (we need to identify which rows belong to the same simulation)
        # We'll use a combination of features to identify unique simulation runs
        # This assumes each simulation has the same number of rows (equal to 'size')
        
        # First, get all rows for this size
        size_data = size_rows.copy()
        
        # Calculate number of rows per simulation (should be equal to size)
        expected_rows_per_sim = size
        
        # Total number of rows for this size
        total_rows = len(size_data)
        
        # Expected number of simulations
        expected_sims = total_rows / expected_rows_per_sim
        
        if expected_sims != 30:
            print(f"Warning: Expected 30 simulations for size {size}, got approximately {expected_sims}")
        
        # Now we need to split the data into simulation groups
        # This assumes the data is already ordered by simulation
        # and that each simulation has exactly 'size' number of rows
        
        for sim_idx in range(min(30, int(np.ceil(expected_sims)))):
            start_idx = sim_idx * expected_rows_per_sim
            end_idx = min(start_idx + expected_rows_per_sim, total_rows)
            
            if start_idx >= total_rows:
                # No more data for this simulation, fill with NaN
                averaged_errors[i, sim_idx] = np.nan
                continue
            
            # Get data for this simulation
            sim_data = size_data.iloc[start_idx:end_idx]
            
            # Average the errors for this simulation
            if not sim_data.empty:
                averaged_errors[i, sim_idx] = np.nanmean(sim_data['est_errors'].values)
            else:
                averaged_errors[i, sim_idx] = np.nan
    
    # Calculate mean and standard error across simulations for each size
    #mean_errors = np.nanmean(averaged_errors, axis=1)
    #std_errors = np.nanstd(averaged_errors, axis=1) / np.sqrt(np.sum(~np.isnan(averaged_errors), axis=1))
    
    return averaged_errors

# df_col_errors = pd.read_csv("last_col_exper/results/est_errors-col-col-lbo.csv")
# df_row_errors = pd.read_csv("last_col_exper/results/est_errors-row-row-lbo.csv")
# df_usvt_errors = pd.read_csv("last_col_exper/results/est_errors-usvt-lbo.csv")
# df_drnn_errors = pd.read_csv("last_col_exper/results/est_errors-dr-lbo.csv")
# df_tsnn_errors = pd.read_csv("last_col_exper/results/est_errors-ts-lbo.csv")

# Process each CSV file
col_err = process_csv("all_exper_errdec_30/results/est_errors-col-col-lbo.csv")
row_err = process_csv("all_exper_errdec_30/results/est_errors-row-row-lbo.csv")
usvt_err = process_csv("all_exper_errdec_30/results/est_errors-usvt-lbo.csv")
drnn_err = process_csv("all_exper_errdec_30/results/est_errors-dr-lbo.csv")
tsnn_err = process_csv("all_exper_errdec_30/results/est_errors-ts-lbo.csv")

# Extract errors for each method as a numpy array 4 x 30


# USVT_errors = 2**-3 * T_values**-0.10
UserNN_errors = np.nanmean(row_err, axis = 1)
TimeNN_errors = np.nanmean(col_err, axis = 1)
DRNN_errors = np.nanmean(drnn_err, axis = 1)
USVT_errors = np.nanmean(usvt_err, axis = 1)
TSNN_errors = np.nanmean(tsnn_err, axis = 1)



unn_stderr = np.nanstd(row_err, axis = 1) / np.sqrt(row_err.shape[1])
tnn_stderr = np.nanstd(col_err, axis = 1) / np.sqrt(col_err.shape[1])
drnn_stderr = np.nanstd(drnn_err, axis = 1) / np.sqrt(drnn_err.shape[1])
usvt_stderr = np.nanstd(usvt_err, axis = 1) / np.sqrt(usvt_err.shape[1])
tsnn_stderr = np.nanstd(tsnn_err, axis = 1) / np.sqrt(tsnn_err.shape[1])



# Create the plot
plt.figure()
#plt.plot(T_values, USVT_errors, 'r', linestyle='-', marker='D', markersize=8, linewidth=2, label=r'USVT: $T^{-0.46}$')

def add_regression_line(x, y, color, label, linestyle):
    # Fit a line to log-transformed data
    slope, intercept = np.polyfit(np.log2(x), np.log2(y), 1)
    plt.plot(x, 2**(intercept) * x**slope, color=color, linestyle=linestyle, linewidth=2) 
            #label=f"{label} (slope: {slope:.2f})")
    return slope

usvt_slope = add_regression_line(T_values, USVT_errors, 'red', 'USVT', '--')
unn_slope = add_regression_line(T_values, UserNN_errors, 'green', 'User-NN', ':')
tnn_slope = add_regression_line(T_values, TimeNN_errors, 'orange', 'Time-NN', ':')
drnn_slope = add_regression_line(T_values, DRNN_errors, 'blue', 'DR-NN', '--')
tsnn_slope = add_regression_line(T_values, TSNN_errors, 'purple', 'Time-NN', ':')

# Plot each method with corresponding markers, colors, and line styles
plt.errorbar(T_values, USVT_errors, yerr=usvt_stderr, fmt = 'r', marker='>', markersize=12, linestyle='None', barsabove=True, label=rf'USVT: $T^{{{usvt_slope:.2f}}}$')
plt.errorbar(T_values, UserNN_errors, yerr=unn_stderr, fmt = 'g', marker='o', markersize=12, linestyle='None', label=rf'Row-NN: $T^{{{unn_slope:.2f}}}$')
plt.errorbar(T_values, TimeNN_errors, yerr=tnn_stderr, fmt='s', color = "orange", marker='s', linestyle='None', markersize=12, label=rf'Col-NN: $T^{{{tnn_slope:.2f}}}$')
plt.errorbar(T_values, DRNN_errors, fmt='bD', yerr=drnn_stderr, markersize=12, linestyle="None", label=rf'DR-NN: $T^{{{drnn_slope:.2f}}}$')
plt.errorbar(T_values, TSNN_errors, fmt='p', yerr=tsnn_stderr, markersize=12, linestyle="None", label=rf'TS-NN: $T^{{{tsnn_slope:.2f}}}$')
# Logarithmic scale for both axes
plt.xscale('log', base=2)
plt.yscale('log', base=2)

# Axis labels
plt.xlabel(r'T', fontsize=15)
plt.ylabel(r'Error for t = T, a = 1', fontsize=15)

# Title
#plt.title(r'Decay of avg. error across users (N = T, 30 trials)', fontsize=16)

# Add legend
plt.legend(fontsize=12)

# Grid for better readability
#plt.grid(True, which="both", linestyle='--', linewidth=0.5)
ax1 = plt.gca()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.grid(True, alpha=0.4)


# Show the plot#
plt.savefig("all_exper_errdec_30/sims_plot.pdf", bbox_inches = "tight")


# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# # --- Configuration ---

# # 1. Set the directory where your CSV files are located
# CSV_DIRECTORY = './size_expers_nonoise/results' # Use '.' for the current directory, or '/path/to/your/csvs/'

# # 2. List the exact method names used in your filenames
# #    Example: if you have 'est_errors-MyMethod1-lbo.csv', add 'MyMethod1'
# METHOD_NAMES = ['row-row', 'col-col', 'dr', 'ts', 'usvt'] # <-- REPLACE WITH YOUR ACTUAL METHOD NAMES

# # 3. Define the sizes for which you want separate plots
# SIZES_TO_PLOT = [16, 32, 64, 128] # Corresponds to 2**4, 2**5, 2**6, 2**7

# # --- Load Data ---
# all_data = []

# for method in METHOD_NAMES:
#     # Construct the expected filename
#     filename = f'est_errors-{method}-lbo.csv'
#     file_path = os.path.join(CSV_DIRECTORY, filename)

#     try:
#         # Read the CSV
#         df_temp = pd.read_csv(file_path)

#         # Add the method name as a column (critical step!)
#         df_temp['estimation_method'] = method

#         # Select only necessary columns (optional, but good practice)
#         df_temp = df_temp[['estimation_method', 'est_errors', 'size']]

#         all_data.append(df_temp)
#         print(f"  - Loaded data for method: '{method}' from {filename}")

#     except FileNotFoundError:
#         print(f"  - WARNING: File not found for method '{method}': {file_path}. Skipping.")
#     except Exception as e:
#         print(f"  - WARNING: Error reading file for method '{method}' ({file_path}): {e}. Skipping.")

# # --- Combine Data ---
# if not all_data:
#     print("\nError: No data was successfully loaded. Cannot create plots.")
# else:
#     combined_df = pd.concat(all_data, ignore_index=True)
#     print(f"\nData combined. Total rows: {len(combined_df)}")

#     # --- Create Plots using Matplotlib ---
#     print("\nGenerating plots...")
#     for size_val in SIZES_TO_PLOT:
#         print(f"  - Plotting for size = {size_val}...")

#         # Filter data for the current size
#         df_size_filtered = combined_df[combined_df['size'].astype(float) == float(size_val)]

#         if df_size_filtered.empty:
#             print(f"    -> No data found for size {size_val}. Skipping this plot.")
#             continue

#         # Prepare data specifically for plt.boxplot
#         # It needs a list where each element is the data for one box
#         data_to_plot = []
#         labels_for_plot = []
#         for method in METHOD_NAMES:
#             # Get errors for this specific method AND this specific size
#             errors_for_method = df_size_filtered[(df_size_filtered['estimation_method'] == method) & (~np.isnan(df_size_filtered['est_errors']))]['est_errors'].tolist()
            

#             # Only add if there's data, otherwise boxplot might behave unexpectedly or give warnings
#             if errors_for_method:
#                  if method == "ts":
#                     print(f"errors_for_method {method}")
#                  data_to_plot.append(errors_for_method)
#                  labels_for_plot.append(method)
#             else:
#                  # Optionally, add an empty list if you want a gap or placeholder
#                  # data_to_plot.append([])
#                  # labels_for_plot.append(method)
#                  print(f"    -> No data for method '{method}' at size {size_val}.")


#         if not data_to_plot:
#              print(f"    -> No data found for *any* method at size {size_val}. Skipping plot.")
#              continue

#         # Create the plot using Matplotlib
#         fig, ax = plt.subplots(figsize=(10, 6)) # Get figure and axes objects
#         ax.boxplot(data_to_plot, tick_labels=labels_for_plot) # Pass the prepared data and labels

#         # Customize the plot
#         ax.set_title(f'Estimation Errors for Size = {size_val}')
#         ax.set_xlabel('Estimation Method')
#         ax.set_ylabel('Estimation Error (est_errors)')
#         plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if they overlap
#         plt.tight_layout() # Adjust layout to prevent labels overlapping

#         # Show the plot
#         plt.savefig(f'plot_size_{size_val}.png', dpi=300, bbox_inches='tight')

#     print("\nPlotting finished.")