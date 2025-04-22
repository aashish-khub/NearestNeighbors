"""Script to generate plots from Star NN simulation results.

This script creates two plots:
1. Mean training time vs matrix size with error bars
2. Empirical vs estimated noise variance by matrix size with connecting lines
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
from nearest_neighbors.utils.experiments import setup_logging

# Setup logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate plots from Star NN simulation results')
parser.add_argument('-od', '--output_dir', type=str, default='out', help='Output directory')
args = parser.parse_args()

# Define the output directory
output_dir = args.output_dir
results_dir = os.path.join(output_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Load the train_times data from CSV
train_times_path = os.path.join(results_dir, "star_nn_train_times.csv")
if not os.path.exists(train_times_path):
    logger.error(f"Train times data file not found at {train_times_path}")
    exit(1)

train_times_df = pd.read_csv(train_times_path)
logger.info(f"Loaded train times data with {len(train_times_df)} rows")

# Plot 1: Mean training time vs size with error bars
plt.figure(figsize=(10, 6))

# Calculate mean and standard deviation of training time for each size
mean_times = train_times_df.groupby('size')['time_fit'].mean()
std_times = train_times_df.groupby('size')['time_fit'].std()
sizes = mean_times.index

# Create error bar plot
plt.errorbar(sizes, mean_times, yerr=std_times, fmt='o-', capsize=5, linewidth=2, markersize=8)
plt.xlabel('Matrix Size (n*m)', fontsize=12)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.title('Mean Training Time vs Matrix Size', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'star_nn_training_time.png'), dpi=300)
plt.close()

# Plot 2: Empirical vs estimated noise variance by matrix size
plt.figure(figsize=(10, 6))

# Define a slight shift for better visibility
shift = 0.1

# Get unique sizes and sort them
unique_sizes = sorted(train_times_df['size'].unique())

# Create a mapping from size to x-coordinate for even spacing
size_to_x = {size: i for i, size in enumerate(unique_sizes)}

# Plot for each size
for size in unique_sizes:
    # Filter data for this size
    size_data = train_times_df[train_times_df['size'] == size]
    
    # Get x-coordinate for this size
    x_pos = size_to_x[size]
    
    # Plot empirical noise variance points (shifted slightly to the right)
    plt.scatter(
        [x_pos + shift] * len(size_data), 
        size_data['empirical_noise_variance'], 
        color='blue', 
        label='Empirical' if size == unique_sizes[0] else "",
        s=100,
        alpha=0.7
    )
    
    # Plot estimated noise variance points (shifted slightly to the left)
    plt.scatter(
        [x_pos - shift] * len(size_data), 
        size_data['estimated_noise_variance'], 
        color='red', 
        label='Estimated' if size == unique_sizes[0] else "",
        s=100,
        alpha=0.7
    )
    
    # Connect points from the same replication with thin lines
    for _, row in size_data.iterrows():
        plt.plot(
            [x_pos - shift, x_pos + shift], 
            [row['estimated_noise_variance'], row['empirical_noise_variance']], 
            color='gray', 
            linestyle='-', 
            linewidth=0.5,
            alpha=0.5
        )

# Set x-axis ticks to the size values
plt.xticks(range(len(unique_sizes)), unique_sizes)

plt.xlabel('Matrix Size (n*m)', fontsize=12)
plt.ylabel('Noise Variance', fontsize=12)
plt.title('Empirical vs Estimated Noise Variance by Matrix Size', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'star_nn_noise_variance.png'), dpi=300)
plt.close()

logger.info(f"Plots saved to {results_dir}") 