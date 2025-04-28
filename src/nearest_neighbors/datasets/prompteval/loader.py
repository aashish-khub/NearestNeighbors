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
import pickle
from typing import Any
import logging
from joblib import Memory


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

    urls = {
        "full_data": "https://github.com/kyuseongchoi5/EfficientEval_BayesOpt/tree/main/data/MMLU/data_all.pkl"
    }

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

    def process_data_scalar(self) -> tuple[np.ndarray, np.ndarray]:
        """Processes the data into scalar setting. This implementation is applicable when generating (template * example) matrix, while fixing model and task.

        Returns
        -------
            data: 2d processed data matrix of floats (in this case, each entry is boolean as the metric is correctness)
            mask: Mask for processed data

        """
        if not self.tasks or not self.models:
            raise ValueError("Tasks and models must be specified")

        model = self.models[0]  # Use the first model
        task = self.tasks[0]  # Use the first task
        propensity = self.propensity

        assert len(self.models) == 1, "Only one model is supported in scalar mode"
        assert len(self.tasks) == 1, "Only one task is supported in scalar mode"

        full_data = self._load_data()

        df = pd.DataFrame(full_data[0][model])
        df_subject = df[df["subject"] == task]
        num_examples = len(
            df_subject["correctness"]
        )  # decide the column dimension of the data matrix

        temp_example = np.zeros(
            (len(full_data), num_examples)
        )  # num_templates * num_examples

        for j in range(len(full_data)):
            # j : per prompt template
            df = pd.DataFrame(full_data[j][model])
            print(f"Loading {model} for {j}th prompt template with {task} subject")
            df_subject = df[df["subject"] == task]
            temp_example[j, :] = df_subject["correctness"]

        temp_example_df = pd.DataFrame(temp_example)

        # Create a mask of the original data according to the propensity
        original_mask = (
            temp_example_df.notna()
        )  # This just tells us what's naturally present

        n_rows = temp_example_df.shape[0]
        n_cols = temp_example_df.shape[1]
        n_rows_keep = int(n_rows * propensity)
        n_cols_keep = int(n_cols * propensity)

        # Randomly select which rows/columns to keep
        rows_keep_indices = np.random.choice(n_rows, n_rows_keep, replace=False)
        cols_keep_indices = np.random.choice(n_cols, n_cols_keep, replace=False)

        # Create a new mask starting with all False
        propensity_mask = pd.DataFrame(
            False, index=original_mask.index, columns=original_mask.columns
        )

        # Set the randomly selected rows/columns to True
        for i in rows_keep_indices:
            for j in cols_keep_indices:
                propensity_mask.iloc[i, j] = True

        data = temp_example_df.to_numpy()
        mask = propensity_mask.to_numpy()
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

        full_data = self._load_data()

        task_model_temp = np.zeros((len(tasks), len(models), len(full_data)))

        for j in range(len(full_data)):
            # j : per prompt template
            for k, model in enumerate(models):
                # model : per model
                df = pd.DataFrame(full_data[j][model])
                for l, subject in enumerate(tasks):
                    print(
                        f"Loading {model} for {j}th prompt template with {subject} subject"
                    )
                    df_subject = df[df["subject"] == subject]
                    task_model_temp[l, k, j] = np.mean(df_subject["correctness"])

        task_model_temp_df = pd.DataFrame(task_model_temp)

        # Create a mask of the original data according to the propensity
        original_mask = (
            task_model_temp_df.notna()
        )  # This just tells us what's naturally present

        n_rows = task_model_temp_df.shape[0]
        n_cols = task_model_temp_df.shape[1]
        n_rows_keep = int(n_rows * propensity)
        n_cols_keep = int(n_cols * propensity)

        # Randomly select which rows/columns to keep
        rows_keep_indices = np.random.choice(n_rows, n_rows_keep, replace=False)
        cols_keep_indices = np.random.choice(n_cols, n_cols_keep, replace=False)

        # Create a new mask starting with all False
        propensity_mask = pd.DataFrame(
            False, index=original_mask.index, columns=original_mask.columns
        )

        # Set the randomly selected rows/columns to True
        for i in rows_keep_indices:
            for j in cols_keep_indices:
                propensity_mask.iloc[i, j] = True

        data = task_model_temp_df.to_numpy()
        mask = propensity_mask.to_numpy()
        self.data = data
        self.mask = mask
        return data, mask

    def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
        """Returns the full state as a dictionary. For HeartSteps, this includes the data, masking matrix, and the custom parameters (if include_metadata == True

        If the data and mask are None, then the data has not been processed yet. Call process_data_scalar() or process_data_distribution() to process the data first.

        Args:
            include_metadata (bool): Whether to include metadata in the dictionary. Default: False. The metadata for HeartSteps is currently empty.

        """
        full_state = {
            "data": self.data,
            "mask": self.mask,
        }
        return full_state

    @classmethod
    @memory.cache
    def _load_data(cls) -> Any:
        """Load the MMLU full dataset.

        Returns:
            full_data: Any Python object stored in the pickle file

        """
        logger.info("Retrieving MMLU full dataset from url...")
        full_data_path = cls.urls["full_data"]

        # GitHub URLs can't be directly downloaded; you would need to use raw content
        # or download from releases. Here we're assuming the file is downloaded locally.
        try:
            # Using standard pickle module instead of pandas
            with open(full_data_path, "rb") as f:
                full_data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            raise ValueError(
                f"Could not load data from {full_data_path}. Make sure the file exists."
            )

        return full_data
