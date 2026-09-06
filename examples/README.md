# Examples

Runnable scripts showing how to use $N^2$, one directory per benchmark dataset plus
`simulations/` for synthetic data. Each directory has `demo_*_dataloader.py` (load and
inspect the data), `run_scalar.py` / `run_distribution.py` (evaluate every estimator on a
held-out block; see `bench/README.md`), and `plot_*.py` (figures from those results).

Install the extras first: `pip install "nsquared[examples]"`. Run the scripts from their
own directory, e.g. `cd examples/heartsteps && python demo_hs_dataloader.py`.

Nothing here is part of the package itself; treat it as reference code.
