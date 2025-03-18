# NearestNeighbors

<!-- Given an incomplete matrix, where the entries in the matrix could correpsond to either scalars or distributions, the goal is to fill in the rest of the matrix. See [examples](./examples/) on how matrix completion can be applied to problems in personalized healthcare, LLM evaluation, and more.  -->

<!-- We leverage nearest neighbor methods due to their simplicity and scalability. These algorithms estimate a missing entry by finding "similar" rows or columns and then use their average as the estimate for a missing entry. -->

## Setup
This package requires Python 3.10.4. Please verify your Python version by running `python --version` in your terminal. If you’re not running Python 3.10.4, please adjust your environment accordingly (for example, if you use pyenv: `pyenv local 3.10.4`).

> [!NOTE]
> To install pyenv, follow the instructions here: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation.

Dependencies are managed in `pyproject.toml`. To install the dependencies, run the following commands, based on your Operating System:

**POSIX Systems (MacOS/Linux):**
```bash
python --version   # Ensure this outputs Python 3.10.4
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . # Install in editable mode
```
**Windows Systems:**
```powershell
python --version   # Ensure this outputs Python 3.10.4
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e . # Install in editable mode
```

> [!NOTE]
> If using VSCode, make sure to set the interpreter to the .venv environment using `Cmd + Shift + P` -> `Python: Select Interpreter`.

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

## Examples

This repository implements the methods from the following papers:

[Vanilla Nearest Neighbors](https://arxiv.org/abs/2202.06891)
```bibtex
@article{dwivedi2022counterfactual,
  title={Counterfactual inference for sequential experiments},
  author={Dwivedi, Raaz and Tian, Katherine and Tomkins, Sabina and Klasnja, Predrag and Murphy, Susan and Shah, Devavrat},
  journal={arXiv preprint arXiv:2202.06891},
  year={2022}
}
```

[Two Sided Nearest Neighbors (TS-NN)](https://arxiv.org/abs/2411.12965)
```bibtex
@article{sadhukhan2024adaptivity,
  title={On adaptivity and minimax optimality of two-sided nearest neighbors},
  author={Sadhukhan, Tathagata and Paul, Manit and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2411.12965},
  year={2024}
}
```

[Doubly Robust Nearest Neighbors (DRNN)](https://arxiv.org/abs/2211.14297)
```bibtex
@article{dwivedi2022doubly,
  title={Doubly robust nearest neighbors in factor models},
  author={Dwivedi, Raaz and Tian, Katherine and Tomkins, Sabina and Klasnja, Predrag and Murphy, Susan and Shah, Devavrat},
  journal={arXiv preprint arXiv:2211.14297},
  year={2022}
}
```

[Distributional (Wasserstein) Nearest Neighbors (WassersteinNN)](https://arxiv.org/abs/2410.13112)
```bibtex
@article{feitelberg2024distributional,
  title={Distributional Matrix Completion via Nearest Neighbors in the Wasserstein Space},
  author={Feitelberg, Jacob and Choi, Kyuseong and Agarwal, Anish and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2410.13112},
  year={2024}
}
```

[Distributional (Kernel) Nearest Neighbors](https://arxiv.org/abs/2410.13381)
```bibtex
@article{choi2024learning,
  title={Learning Counterfactual Distributions via Kernel Nearest Neighbors},
  author={Choi, Kyuseong and Feitelberg, Jacob and Chin, Caleb and Agarwal, Anish and Dwivedi, Raaz},
  journal={arXiv preprint arXiv:2410.13381},
  year={2024}
}
```

[Synthetic Nearest Neighbors (Syn-NN)](https://arxiv.org/abs/2109.15154)
```bibtex
@inproceedings{agarwal2023causal,
  title={Causal matrix completion},
  author={Agarwal, Anish and Dahleh, Munther and Shah, Devavrat and Shen, Dennis},
  booktitle={The thirty sixth annual conference on learning theory},
  pages={3821--3826},
  year={2023},
  organization={PMLR}
}
```
Code: [Syn-NN implementation](https://github.com/AbdullahO/What-If/blob/main/algorithms/snn_biclustering.py)
