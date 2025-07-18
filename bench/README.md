# $N^2$ Bench
In this directory, we provide resources for replicating our experimental results, adding and testing new methods, and adding new datasets.
## Replicating experimental results

### For Windows users
We run our experiments with shell scripts, which requires additional setup for Windows users. A recommended option is Git Bash.

#### Using Git Bash
1. Install [Git for Windows](https://gitforwindows.org/)
2. Open Git Bash
3. Navigate to project bench directory

### For Mac / Linux Users
No additional setup should be necessary. 

### Experiments:
To replicate the experimental results in [our paper](https://arxiv.org/pdf/2506.04166), run `bench/experiments.sh` as follows:

```bash
# from the bench directory
./experiments.sh -o OUTPUT_DIR -l LOG_LEVEL -e EXPERIMENT_NAME
```
If `EXPERIMENT_NAME` is `all` (default), experiments on all four datasets (heartsteps, movielens, prompteval, and prop99) will be run and stored in `OUTPUT_DIR`. You can also run the experiments for a specific dataset setting `EXPERIMENT_NAME` to `heartsteps`, `movielens`, `prompteval`, and `prop99`. 

The experiments for `movielens` are very memory-intensive due to the size of the dataset and could terminate unexpectedly. 

For plotting utilities, see the `examples` directory for files with names prefixed by `plot`.  


## Adding and testing new methods

### Nearest neighbor variants:
To add a new nearest neighbor variant:

1. Define an `EstimationMethod` in `nsquared/estimation_methods.py`. 
   
   ```python
   class MyNewEstimator(EstimationMethod):
    """Estimate things"""
        def __init__(self, is_percentile: bool = True):
            super().__init__(is_percentile)
            ...

        def impute(
            self,
            row: int,
            column: int,
            data_array: npt.NDArray,
            mask_array: npt.NDArray,
            distance_threshold: Union[float, Tuple[float, float]],
            data_type: DataType,
            allow_self_neighbor: bool = False,
            **kwargs: Any,
        ) -> npt.NDArray:
        """
        Impute the missing value at
        the given row and column using 
        XYZ method
        """
        ...
   ```

2. Define a `FitMethod` (if needed). 
    ```python
    class MyNewFitMethod(FitMethod):
        def __init__(
            self,
            block: list[tuple[int, int]],
            distance_threshold_range_row: tuple[float, float],
            distance_threshold_range_col: tuple[float, float],
            alpha_range: tuple[float, float],
            n_trials: int,
            data_type: DataType,
            allow_self_neighbor: bool = False,
        ):
            ...

    def fit(
            self,
            data_array: npt.NDArray,
            mask_array: npt.NDArray,
            imputer: NearestNeighborImputer,
            ret_trials: bool = False,
        ) -> Union[tuple[float, float], tuple[tuple[float, float], Trials]]:
            ...

    ```

3. To ensure compatability with the experiments and plotting functions, add an alias for your method in [`nsquared/utils/experiments.py`](https://github.com/aashish-khub/NearestNeighbors/blob/441a382efba2cf68c22ac379bd5750f91d9e03ee/src/nsquared/utils/experiments.py#L19) under the parser argument for `--estimation_method` and fill out [`nsquared/utils/plotting_utils.py`](https://github.com/aashish-khub/NearestNeighbors/blob/441a382efba2cf68c22ac379bd5750f91d9e03ee/src/nsquared/utils/plotting_utils.py#L19) with your desired plot settings. 

4. To test on a given dataset (for example: `heartsteps`), navigate to `examples/heartsteps/run_scalar.py` and add your method following the existing template. In general, all that is required is adding a block to [this conditional structure](https://github.com/aashish-khub/NearestNeighbors/blob/441a382efba2cf68c22ac379bd5750f91d9e03ee/examples/heartsteps/run_scalar.py#L157), but your method is not required to use the exact format of the `run_scalar.py` script. For distributional methods, do the same, but in `run_distribution.py`. 
   
5. Adjust the corresponding `slurm_scripts/*.sh` files in the `examples` directory to add your new methods alias under `METHODS`. 

### Non-nearest neigbor methods
If you wish to test methods on the benchmark that do not follow the $N^2$ framework, there are two options:

#### Option 1:
Use downloaded data hosted [here](https://github.com/calebchin/nsquared_bench_data). 

This contains the masked (missingness included) matrix used in our experiments, the corresponding unmasked (no missingness) matrix, and the masking matrix for each dataset. 

#### Option 2:
Use the dataloader in `nsquared.datasets`. 

For examples on how the dataloader works, check out the use of the loader [here](https://github.com/aashish-khub/NearestNeighbors/blob/main/examples/heartsteps/run_scalar.py) or [here](https://github.com/aashish-khub/NearestNeighbors/blob/main/examples/prop99/run_scalar.py). 


## Adding new datasets

To add a new dataset:

1. Create a new directory inside the `nsquared/datasets` directory with two files: `loader.py` and `__init__.py`.
2. The `loader.py` file should look something like:
   ```python
    from nsquared.datasets.dataloader_base import NNDataLoader
    from nsquared.datasets.dataloader_factory import register_dataset
    # imports
    ...
    #
    memory = Memory(".joblib_cache", verbose=2)
    logger = logging.getLogger(__name__)
    params = {
    # specific parameters for the dataset
    }

    @register_dataset("MyNewDataset", params)
    class MyNewDatasetLoader(NNDataLoader):
        """
        ...
        """
        def __init__(
            self,
            # specific params
            agg: str = "mean",
            **kwargs: Any,
            ):
            """
            Initialize data loader
            """
            super().__init__(
                agg=agg,
                **kwargs,
            )
            ...

        def process_data_scalar(self, agg: str = "mean") -> tuple[np.ndarray, np.ndarray]:
            """
            Process new dataset in the scalar setting (# of entries is 1)
            """      
            ...      

        def process_data_distribution(self, data_type: DataType | None = None) -> tuple[np.ndarray, np.ndarray]:
            """
            Process new dataset in distribuional setting (# of entries is > 1)
            """
            ...

        def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
            """
            Returns the full state as a dictionary (including data matrix, mask, and specific params).
            """
            ...
   ```
3. The `__init__.py` should contain:
   ```python
   from .loader import MyNewDatasetLoader  # noqa: F401
   ```
4. To run the existing methods on the dataset, follow the template provided in any of the example `run_scalar.py` or `run_distribution.py` files. It is possible that all you will need to change is the data loading step (e.g. `my_new_dataloader = NNData.create("MyNewDataset")`). 