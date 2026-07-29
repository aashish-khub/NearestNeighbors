# Contributing to $N^2$

Thanks for your interest in $N^2$! This project is developed in the open and we welcome
bug reports, questions, documentation improvements, new nearest neighbor variants, new
data types, and new benchmark datasets.

Everyone participating in this project is expected to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## Getting support

If you have a question about how to use the package, please **do not** email the authors
individually — asking in public means the next person with the same question can find the
answer.

1. Check the [documentation](docs/index.md) and the [examples](examples/) directory.
2. Search [existing issues](https://github.com/aashish-khub/NearestNeighbors/issues) —
   your question may already be answered.
3. Open a new issue using the **Question / support** template. We aim to respond within a
   week.

## Reporting a bug

Open an issue using the **Bug report** template and include:

- what you expected to happen and what actually happened,
- a minimal, self-contained code snippet that reproduces the problem,
- the full traceback, if there is one,
- your output of `python -c "import nsquared, sys, numpy; print(sys.version, numpy.__version__)"`
  and your operating system.

Reports that include a runnable reproduction get fixed much faster than reports that
don't.

## Requesting a feature

Open an issue using the **Feature request** template. Describe the problem you are trying
to solve rather than only the solution you have in mind — often there is an existing way
to do it, or a more general change that helps more users.

## Security issues

If you believe you have found a security problem, please report it privately by emailing
`dwivedi@cornell.edu` rather than opening a public issue.

---

## Setting up a development environment

$N^2$ requires Python 3.10 or later.

```bash
git clone https://github.com/aashish-khub/NearestNeighbors.git
cd NearestNeighbors
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
```

`pip install -e ".[dev]"` installs the package in editable mode along with every optional
extra (`data`, `plots`, `examples`) plus `pytest`, `ruff`, and `pre-commit`. The test
suite assumes all of them, so use `[dev]` rather than a narrower extra when developing.

The runtime install is deliberately much smaller — `pip install nsquared` pulls only
`numpy`, `hyperopt`, and `tqdm`. Please keep it that way: if you add a dependency, put it
behind the extra that needs it rather than in `[project.dependencies]`, and make the
import site fail with a message naming that extra. See the missing-dependency handling in
`nsquared/datasets/dataloader_factory.py` for the pattern. For a self-contained numerical
algorithm, prefer implementing it on NumPy over adding a dependency — that is why
`nsquared/baselines/_softimpute.py` exists.

## Running the checks

Run the linter and formatter over the whole tree:

```bash
pre-commit run --all-files
```

This runs `ruff` (lint + format) and `pyright` (static type checking). Both must pass —
`tests/test_precommit.py` runs `pre-commit` as part of the test suite, so a lint failure
is a test failure.

Run the test suite:

```bash
pytest
```

Both commands are run on every push and pull request by
[CI](.github/workflows/ci.yml), across Python 3.10–3.12.

### Style conventions

- Public functions, classes, and modules need docstrings (`ruff`'s `D` rules are enabled).
- Type annotations are required on function signatures (`ruff`'s `ANN` rules are enabled)
  and checked by `pyright`.
- Formatting is handled by `ruff format`; do not hand-format.

## Submitting a pull request

1. Fork the repository and create a branch off `main` with a descriptive name.
2. Make your change, adding tests for any new behavior and updating the docs if you
   change a public interface.
3. Make sure `pre-commit run --all-files` and `pytest` both pass locally.
4. Open a pull request describing what changed and why, and linking any related issue.
5. A maintainer will review it. We may ask for changes; this is normal and not a
   judgment about your work.

Please keep pull requests focused — one logical change per PR is much easier to review
than several unrelated ones.

---

## Extending $N^2$

The package is built around two abstractions that you can extend independently. See
[docs/user_guide.md](docs/user_guide.md) for the full explanation, and
[bench/README.md](bench/README.md) for the benchmark-specific details.

### Adding a nearest neighbor variant

Implement a new `EstimationMethod` subclass in `src/nsquared/estimation_methods.py`
(defining `impute`), add a `FitMethod` in `src/nsquared/fit_methods.py` if your method
needs a different cross-validation strategy, and register an alias for it in
`src/nsquared/utils/experiments.py` and `src/nsquared/utils/plotting_utils.py` so it can
be used from the benchmark scripts. Because estimators are written against the abstract
`DataType` interface, your method will work for scalars *and* distributions automatically.

### Adding a data type

Implement a new `DataType` subclass in `src/nsquared/data_types.py`, defining `distance`
and `average` for your entry space. Every existing estimator (with the exception of the
doubly robust one, which needs a well-defined subtraction) then works with it out of
the box.

### Adding a benchmark dataset

Create `src/nsquared/datasets/<name>/loader.py` with an `NNDataLoader` subclass decorated
with `@register_dataset("<Name>", params)`, plus an `__init__.py` re-exporting it. The
loader is then available through `NNData.create("<Name>")`. See
[bench/README.md](bench/README.md#adding-new-datasets) for the required methods.

### Please add tests

New estimators and data types should come with tests in `tests/`. The existing tests are
good templates: they check invariants that must hold regardless of tuning (for example,
that imputing a constant matrix returns that constant, and that imputation error
decreases as the matrix grows).

---

## Releasing (maintainers)

See [RELEASE.md](RELEASE.md) for the versioning and tagging procedure.
