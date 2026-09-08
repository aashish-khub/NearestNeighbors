# Replicating simulations experiments

Figure 1 ([`example_plots/sims_plot_highsnr.pdf`](example_plots/sims_plot_highsnr.pdf)):
AutoNN recovers DRNN performance with high SNR. The low-SNR counterpart is
[`example_plots/sims_plot_lowsnr.pdf`](example_plots/sims_plot_lowsnr.pdf).

To replicate Figure 1, run the following commands from the `examples/simulations` directory.

```bash
# Run simulated experiments with high SNR (noise std = 0.001)
./slurm_scripts/run_accuracy.sh -o OUTPUT_DIR -l ERROR -n 0.001
# Plot the figure
python plot_size_error.py --output_dir OUTPUT_DIR
```

To replicate the low SNR case, run the same commands with `-n 1.0` instead.
