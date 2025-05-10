"""Dataset loader for the PromptEval (MMLU) dataset.

Source1: https://huggingface.co/datasets/PromptEval
Source2: https://github.com/kyuseongchoi5/EfficientEval_BayesOpt
Paper Reference for transformation implementation:
    Felipe Maia Polo et al.
    "Efficient multi-prompt evaluation of LLMs."
    Neurips, 2024.
    https://arxiv.org/pdf/2405.17202
"""

from nearest_neighbors.datasets.dataloader_base import NNDataLoader
from nearest_neighbors.datasets.dataloader_factory import register_dataset
import numpy as np
import pandas as pd
from typing import Any
import logging
from joblib import Memory
from datasets import load_dataset
from datasets import DatasetDict


memory = Memory(".joblib_cache", verbose=2)
logger = logging.getLogger(__name__)

params = {
    "tasks": (
        list[str],
        None,
        "List of tasks to evaluate on. By default, returns all.",
    ),
    "models": (
        list[str],
        None,
        "List of models to evaluate by. By default, returns all.",
    ),
    "seed": (int, None, "Random seed for reproducibility"),
    "propensity": (float, None, "Proportion of data to keep"),
}


@register_dataset("prompteval", params)
class PromptEvalDataLoader(NNDataLoader):
    """Data from the PromptEval study formatted into a matrix or tensor.
    To initialize with default settings, use: NNData.create("prompteval").

    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        models: list[str] | None = None,
        seed: int | None = None,
        propensity: float = 1.0,  # Default to 1.0 (keeping all data)
        **kwargs: Any,
    ):
        """Initializes the PromptEval data loader.

        Args:
        ----
            tasks: benchmark tasks to evaluate on. Default: None (use all tasks).
            models: models that are evaluated on for each tasks. Default: None (use all models).
            seed: Random seed for reproducibility. Default: None
            propensity: Proportion of data to keep. Default: 1.0
            kwargs: Additional keyword arguments.

        """
        super().__init__(
            **kwargs,
        )
        self.tasks = tasks
        self.models = models
        self.propensity = propensity
        if seed is not None:
            np.random.seed(
                seed=seed
            )  # instantiate random seed if provided but do it only once here

    def load_config_data(self, config_name: str) -> pd.DataFrame | None:
        """Load and process data for a specific configuration.

        Args:
            config_name: Name of the configuration/task to load.

        Returns:
            DataFrame containing the processed data or None if loading fails.

        """
        try:
            # Load the dataset with the specific config
            dataset = load_dataset(
                "PromptEval/PromptEval_MMLU_correctness",
                config_name,
                #    download_mode='force_redownload'
            )
            list_dfs = []
            for key in dataset:
                print(f"Keys in the dataset: {key}")
                if isinstance(dataset, DatasetDict):
                    df = dataset[key].to_pandas()
                    if isinstance(df, pd.DataFrame):
                        df = pd.DataFrame(df)
                        df_reset = df.reset_index()
                        df_long = pd.melt(
                            df_reset,
                            id_vars=["index"],
                            var_name="example",
                            value_name="correctness",
                        )
                        df_long["model"] = key
                        df_long["task"] = config_name
                        df_long = df_long.rename(columns={"index": "format"})

                        list_dfs.append(df_long)
                    else:
                        print(f"Data for key '{key}' is not a DataFrame")
                else:
                    raise ValueError("Loaded dataset is not a DatasetDict")
            df_final = pd.concat(list_dfs, ignore_index=True)
            return df_final
        except Exception as e:
            print(f"Error loading config '{config_name}': {e}")
            return None

    def process_data_scalar(self) -> tuple[np.ndarray, np.ndarray]:
        """Processes the data into scalar setting. This implementation is applicable when generating (template * example) matrix, while fixing model and task.

        Returns
        -------
            data: 2d processed data matrix of floats (in this case, each entry is boolean as the metric is correctness)
            mask: Mask for processed data

        """
        if not self.tasks or not self.models:
            raise ValueError("Tasks and models must be specified")

        assert len(self.models) == 1, "Only one model is supported in scalar mode"
        assert len(self.tasks) == 1, "Only one task is supported in scalar mode"

        model = self.models[0]
        task = self.tasks[0]
        propensity = self.propensity

        dataset = load_dataset("PromptEval/PromptEval_MMLU_correctness", task)
        df = pd.DataFrame(dataset[model])
        N, T = df.shape[0], df.shape[1]
        Masking: np.ndarray = np.zeros((N, T))

        Masking = np.random.binomial(1, propensity, size=(N, T))

        data = df.to_numpy(dtype=float)
        data[Masking == 0] = np.nan
        mask = Masking
        self.data = data
        self.mask = mask
        return data, mask

    def process_data_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """Process the data into distributional setting.

        Returns
        -------
            data: task * model * template matrix of floats (in this case, each entry average of correctness across examples)
            mask: Mask for processed data

        """
        if not self.tasks or not self.models:
            raise ValueError("Tasks and models must be specified")

        models = self.models
        tasks = self.tasks
        propensity = self.propensity

        # Load the data for the multiple config and model
        final_list = []
        for config in tasks:
            df = self.load_config_data(config)
            if df is not None:
                final_list.append(df)
            else:
                print(f"Failed to load data for config: {config}")

        df_final = pd.concat(final_list, ignore_index=True)

        formats = df_final["format"].unique()
        models = df_final["model"].unique()
        tasks = df_final["task"].unique()

        # data_array = np.empty((len(formats), len(models), len(tasks)))
        data_array = np.empty((len(tasks), len(models), len(formats)))
        for id_model, model in enumerate(models):
            for id_task, task in enumerate(tasks):
                df_sub = df_final[
                    (df_final["task"] == task) & (df_final["model"] == model)
                ]
                format_examples = []
                for id_format, format in enumerate(formats):
                    df_sub_task = df_sub[df_sub["format"] == format]
                    format_examples = df_sub_task["correctness"].to_list()
                    data_array[id_task, id_model, id_format] = np.mean(format_examples)

        N, T = len(tasks), len(models)
        Masking: np.ndarray = np.zeros((N, T))
        Masking = np.reshape(np.random.binomial(1, propensity, (N * T)), (N, T))

        data = df_final.to_numpy()
        mask = Masking
        self.data = data
        self.mask = mask
        return data, mask

    def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
        """Returns the full state as a dictionary.

        Args:
            include_metadata (bool): Whether to include metadata in the dictionary. Default: False.

        Returns:
            dict: Dictionary containing the data and mask

        """
        full_state = {
            "data": self.data,
            "mask": self.mask,
        }
        return full_state
