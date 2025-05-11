# Counterfactual Inference: Proposition 99

To compute the estimation error for control states, please run:

```bash
./slurm_scripts/run_accuracy.sh OUTPUT_DIR
```

To generate box plots of the absolute error across control states, please run:

```bash
python plot_sc_error.py -od OUTPUT_DIR
```

To generate the synthetic controls for California in particular, please run:

```bash
./slurm_scripts/run_california.sh OUTPUT_DIR
```

To generate a line plot of the synthetic control versus observed values, please run:

```bash
python plot_synthetic_control.py -od OUTPUT_DIR
```
