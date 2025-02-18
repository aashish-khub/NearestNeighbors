# NearestNeighbors
## Overview
This repository aims to consolidate code implementations for various Nearest Neighbor (NN) methods.

## Setup
This package requires Python 3.10.4 as specified in the pyproject.toml. Please verify your Python version by running `python --version` in your terminal. If you’re not running Python 3.10.4, please adjust your environment accordingly (for example, if you use pyenv: `pyenv local 3.10.4`).


Dependencies are managed in pyproject.toml. To install the dependencies, run the following commands, based on your Operating System:

**POSIX Systems (MacOS/Linux):**
```bash
python --version   # Ensure this outputs Python 3.10.4
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .

```
**Windows Systems:**
```powershell
python --version   # Ensure this outputs Python 3.10.4
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install .
```
## Submitting Changes
### Linting
Before submitting changes, please run pre-commit hooks to ensure that the code is formatted correctly. To do so, run the following command:
```bash
pre-commit run --a
```
The linter should run without any errors and autofix any issues that it can. If there are any issues that the linter cannot fix, please fix them manually before committing your changes.


### Tests
Please ensure that all tests pass before submitting your changes. To run the tests, run the following command:
```bash
pytest
```
Once all tests pass, you can submit your changes.

## Methods to Be Consolidated
1. **Vanilla Nearest Neighbors**
   Paper: [arXiv:2202.06891](https://arxiv.org/pdf/2202.06891)

2. **TS (Two Sided) Nearest Neighbors**
   Paper: [arXiv:2411.12965](https://arxiv.org/pdf/2411.12965)

3. **DR (Doubly Robust) Nearest Neighbors**
   Paper: [arXiv:2211.14297](https://arxiv.org/pdf/2211.14297)

4. **DNN (Wasserstein Distance)**
   Paper: [arXiv:2410.13112](https://arxiv.org/pdf/2410.13112)

5. **DNN (Kernel-based)**
   Paper: [arXiv:2410.13381](https://arxiv.org/pdf/2410.13381)

6. **Syn-NN (Synthetic Nearest Neighbors)**
   Paper: [arXiv:2109.15154](https://arxiv.org/pdf/2109.15154)
   Code: [Syn-NN implementation](https://github.com/AbdullahO/What-If/blob/main/algorithms/snn_biclustering.py)

7. **Nadaraya-Watson**
   Code: [GitHub repository](https://github.com/ag2435/npr/tree/main/npr/nw)

