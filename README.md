# $N^2$

<!-- Given an incomplete matrix, where the entries in the matrix could correpsond to either scalars or distributions, the goal is to fill in the rest of the matrix. See [examples](./examples/) on how matrix completion can be applied to problems in personalized healthcare, LLM evaluation, and more.  -->

<!-- We leverage nearest neighbor methods due to their simplicity and scalability. These algorithms estimate a missing entry by finding "similar" rows or columns and then use their average as the estimate for a missing entry. -->

This package requires Python 3.10 or later. Please verify your Python version by running `python --version` in your terminal. If you're not running Python 3.10+, please adjust your environment accordingly (for example, if you use pyenv: `pyenv local 3.10` or any later version like `pyenv local 3.11`).

> [!NOTE]
> To install pyenv, follow the instructions here: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation, then run `eval "$(pyenv init -)"`.

## Setup for production usage

To install the package in production, run the following command:
```bash
pip install nsquared
```
This will install the package and all its dependencies. You can then use the package with:
```python
import nsquared as nsq
```

## Setup for Development

### If using last release
If you want to use the last release with development functionality enabled, you can install the package with the following command:
```bash
pip install nsquared[dev]
```
This will install the package and all its dependencies, including the development dependencies. You can then use the package with:
```python
import nsquared as nsq
```

### If using latest code (not yet tagged and released)
Dependencies are managed in `pyproject.toml`. To install the dependencies, run the following commands, based on your Operating System:

**POSIX Systems (MacOS/Linux):**

```bash
python --version   # Ensure this outputs Python 3.10 or later
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . # Install in editable mode
```
**Windows Systems:**
```powershell
python --version   # Ensure this outputs Python 3.10 or later
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e . # Install in editable mode
```

Once installed, you can use the package with:
```python
import nsquared as nsq
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
Once all tests pass, you may submit your changes.

### Tagging and Releasing a New Version
Detailed instructions for releasing a new version of the package can be found in the [Release Instructions](RELEASE.md) file. This includes instructions for updating the version number, creating a new release branch, and tagging the release.


## Contributing
We welcome contributions to this package! If you have any ideas for new features, bug fixes, or improvements, please feel free to open an issue or a pull request. We appreciate your help in making this package better!

## License
This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

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