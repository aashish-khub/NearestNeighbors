# Replicating simulations experiments
<div style="text-align: center;">
  <img src="example_plots/sims_plot_highsnr.png" width="350">

  Figure 1: AutoNN recovers DRNN performance with High SNR
</div>




To replicate Figure 1, run the following commands while in the `examples/simulations` directory.

```bash
# Run simulated experiments with high SNR (noise std = 0.001)
./slurm_scripts run_acccuracy.sh -o OUTPUT_DIR -l ERROR -n 0.001
# Plot the figure
python plot_size_error.py --output_dir OUTPUT_DIR
```
To replicate the low SNR case, run the same commands with option `-n 1.0` instead.