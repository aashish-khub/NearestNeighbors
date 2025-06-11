"""Dataset loader for the PromptEval (MMLU) dataset.

Source1: https://huggingface.co/datasets/PromptEval
Source2: https://github.com/kyuseongchoi5/EfficientEval_BayesOpt
Paper Reference for transformation implementation:
    Felipe Maia Polo et al.
    "Efficient multi-prompt evaluation of LLMs."
    Neurips, 2024.
    https://arxiv.org/pdf/2405.17202
"""

from nsquared.datasets.dataloader_base import NNDataLoader
from nsquared.datasets.dataloader_factory import register_dataset
from nsquared.data_types import (
    DataType,
    DistributionKernelMMD,
    DistributionWassersteinSamples,
    DistributionWassersteinQuantile,
)
import numpy as np
import pandas as pd
from typing import Any, cast
import logging
from joblib import Memory
from datasets import load_dataset, Dataset
from tqdm import tqdm


memory = Memory(".joblib_cache", verbose=0)
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

    MODELS = [
        "meta_llama_llama_3_8b",
        "meta_llama_llama_3_8b_instruct",
        "meta_llama_llama_3_70b_instruct",
        "codellama_codellama_34b_instruct",
        "google_flan_t5_xl",
        "google_flan_t5_xxl",
        "google_flan_ul2",
        "ibm_mistralai_merlinite_7b",
        "mistralai_mixtral_8x7b_instruct_v01",
        "mistralai_mistral_7b_instruct_v0_2",
        "google_gemma_7b",
        "google_gemma_7b_it",
        "tiiuae_falcon_40b",
        "mistralai_mistral_7b_v0_1",
        "tiiuae_falcon_180b",
    ]

    TASKS = [
        "college_mathematics",
        "miscellaneous",
        "moral_disputes",
        "jurisprudence",
        "moral_scenarios",
        "college_chemistry",
        "professional_medicine",
        "clinical_knowledge",
        "abstract_algebra",
        "nutrition",
        "professional_psychology",
        "high_school_government_and_politics",
        "high_school_us_history",
        "high_school_chemistry",
        "high_school_macroeconomics",
        "management",
        "conceptual_physics",
        "philosophy",
        "electrical_engineering",
        "high_school_psychology",
        "medical_genetics",
        "high_school_geography",
        "high_school_statistics",
        "international_law",
        "elementary_mathematics",
        "high_school_physics",
        "world_religions",
        "high_school_european_history",
        "formal_logic",
        "security_studies",
        "sociology",
        "high_school_biology",
        "us_foreign_policy",
        "high_school_microeconomics",
        "college_medicine",
        "college_computer_science",
        "logical_fallacies",
        "high_school_computer_science",
        "anatomy",
        "econometrics",
        "astronomy",
        "college_biology",
        "virology",
        "professional_accounting",
        "college_physics",
        "high_school_world_history",
        "business_ethics",
        "global_facts",
        "public_relations",
        "marketing",
        "human_aging",
        "professional_law",
        "high_school_mathematics",
        "prehistory",
        "machine_learning",
        "computer_security",
        "human_sexuality",
    ]

    def __init__(
        self,
        tasks: list[str] | None = None,
        models: list[str] | None = None,
        seed: int | None = None,
        propensity: float = 1.0,  # Default to 1.0 (keeping all data)
        n_examples_per_task: int = 100,
        **kwargs: Any,
    ):
        """Initializes the PromptEval data loader.

        Args:
        ----
            tasks: benchmark tasks to evaluate on. Default: None (use all tasks).
            models: models that are evaluated on for each tasks. Default: None (use all models).
            seed: Random seed for reproducibility. Default: None
            propensity: Proportion of data to keep. Default: 1.0
            n_examples_per_task: Number of examples to sample (with replacement) from each task
                in the distributional setting. Default: 100.
            kwargs: Additional keyword arguments.

        """
        super().__init__(
            **kwargs,
        )
        self.tasks = self.TASKS if tasks is None else tasks
        self.models = self.MODELS if models is None else models
        self.propensity = propensity
        self.seed = seed
        if self.seed is not None:
            np.random.seed(
                seed=self.seed
            )  # instantiate random seed if provided but do it only once here
        self.n_examples_per_task = n_examples_per_task

        # print('Loading dataset')
        # for task in tqdm(self.tasks):
        #     # Download the dataset to cache
        #     load_dataset("PromptEval/PromptEval_MMLU_correctness", name=task)

    @staticmethod
    @memory.cache
    def load_config_data(
        task: str, model: str, n_examples_per_task: int = 100, seed: int | None = None
    ) -> pd.DataFrame | None:
        """Load and process data for a specific configuration.

        Args:
            task: Name of the task to load.
            model: Name of the model to load.
            n_examples_per_task: Number of examples to sample (with replacement) from each task. Default: 100.
            seed: Random seed for reproducibility. Default: None.

        Returns:
            DataFrame containing the processed data or None if loading fails.

        """
        # Load the dataset with the specific config
        # NOTE: dataset is a Dataset
        # NOTE: HF automatically downloads the dataset if it's not already downloaded
        # and caches it in ~/.cache/huggingface/hub/
        dataset = cast(
            Dataset,
            load_dataset(
                "PromptEval/PromptEval_MMLU_correctness",
                name=task,
                split=model,
                num_proc=12,
                keep_in_memory=True,
            ),
        )
        # rows are format templates, columns are examples
        df_table: pd.DataFrame = cast(pd.DataFrame, dataset.to_pandas())
        # sample columns with replacement
        df_table = (
            df_table.transpose()
            .sample(n=n_examples_per_task, replace=True, random_state=seed)
            .transpose()
            .reset_index()
        )
        # melt the table into rows for every format-example pair
        df_long = pd.melt(
            df_table,
            id_vars=["index"],
            var_name="example",
            value_name="correctness",
        )
        df_long["model"] = model
        df_long["task"] = task
        df_long = df_long.rename(columns={"index": "format"})
        return df_long

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

        ds = cast(
            Dataset,
            load_dataset(
                "PromptEval/PromptEval_MMLU_correctness",
                name=task,
                split=model,
                keep_in_memory=True,
                num_proc=8,
            ),
        )
        df = cast(pd.DataFrame, ds.to_pandas())
        mask = np.random.binomial(1, propensity, size=df.shape)

        data = df.to_numpy(dtype=float)
        #data[mask == 0] = np.nan
        self.data = data
        self.mask = mask
        return data, mask

    def process_data_distribution(
        self, data_type: DataType | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process the data into distributional setting.

        Args:
        ----
            data_type: Data type to use for the data. Default: None.

        Returns:
        -------
            data: numpy array of shape (n_tasks, n_models, n_templates, 1)
                if data_type is None or DistributionKernelMMD.
                numpy array of shape (n_tasks, n_models, n_templates)
                if data_type is DistributionWassersteinSamples or DistributionWassersteinQuantile.
                Entry is the average correctness across examples for a given model-task-template combination
            mask: numpy array of shape (n_tasks, n_models, n_templates, 1)
                Entry is 1 if the example is kept, 0 otherwise

        """
        if not self.tasks or not self.models:
            raise ValueError("Tasks and models must be specified")

        models = self.models
        tasks = self.tasks
        propensity = self.propensity

        # Load the data for the multiple config and model
        df_list = []
        for task in tqdm(tasks):
            for model in models:
                df = self.load_config_data(
                    task, model, self.n_examples_per_task, self.seed
                )
                df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)

        # compute the mean correctness across examples for each model-task-format combination
        df = df.pivot_table(
            index=["model", "task", "format"], values="correctness", aggfunc="mean"
        )
        # aggregate the correctness values into a list for each model-task combination
        df = df.groupby(["model", "task"]).agg(list)
        # unstack the task and model index into separate columns
        df = df.unstack()
        # drop the multi-index correctness columns
        df = df.droplevel(0, axis=1)
        # convert lists to numpy arrays of type float
        if data_type is None or isinstance(data_type, DistributionKernelMMD):
            entry_shape = (-1, 1)
        elif isinstance(data_type, DistributionWassersteinSamples) or isinstance(
            data_type, DistributionWassersteinQuantile
        ):
            entry_shape = (-1,)
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        data = np.array(
            [np.array(x, dtype=float) for x in df.values.flatten()]  # type: ignore
        ).reshape(df.shape + entry_shape)

        # Simulate MCAR missingness
        mask = np.random.binomial(1, propensity, size=data.shape[:2])
        #TODO: change this back later
        #data[mask == 0] = np.nan

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
