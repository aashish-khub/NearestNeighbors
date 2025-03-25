from nearest_neighbors.dataloader_base import NNDataLoader
from nearest_neighbors.dataloader_factory import register_dataset
import os
import numpy as np
import pandas as pd
from typing import Any, cast
from datasets import load_dataset, Dataset, DatasetDict

params = {
    # in the future, add custom params here
}


@register_dataset("prompteval", params)
class PromptEvalDataLoader(NNDataLoader):
    """Data from the PromptEval dataset formatted into a matrix or tensor.
    To initialize with default settings, use: NNData.create("prompteval").

    """
    
    urls = {"prompteval": ""}
    all_tasks = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]
    mask_strategies = ["mcar"]

    def __init__(
        self,
        download: bool = False,
        save_dir: str = "./",
        save_processed: bool = False,
        tasks: list = all_tasks,
        mask_strategy: str = "mcar",
        p: float = 0.2,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """Initializes the PromptEval data loader.

        Args:
        ----
            download (bool): whether to download the data locally. Default: False. If True, data is downloaded at save_dir
            save_dir (str): directory to download the data to or where it already exists. Also the directory where the processed data will be.
                Default: "./" (current directory).
            save_processed (bool): whether to save the processed data. Default: False.
            tasks (list): list of tasks to load from PromptEval. Default: all tasks.
            mask_strategy (str): strategy for generating the mask for missing values. Default: "mcar".
            p (float): proportion of missing data for MCAR strategy. Default: 0.2.
            seed (int | None): random seed for reproducibility. Default: None (random)
            **kwargs: additional arguments to be passed to the base class.

        """
        super().__init__(download=download, save_dir=save_dir, **kwargs)
        self.task_list = tasks
        self.data = None
        self.mask = None
        self.seed = seed
        if mask_strategy not in self.mask_strategies:
            raise ValueError(
                f"Mask strategy {mask_strategy} not supported. Supported strategies: {self.mask_strategies}"
            )
        self.mask_strategy = mask_strategy
        if mask_strategy == "mcar":
            self.generate_mask = self._generate_mask_mcar
            self.p = p  # proportion of missing data for MCAR
        else:
            raise NotImplementedError(
                f"Mask strategy {mask_strategy} not implemented yet."
            )

    def download_data(self) -> None:
        """Download the PromptEval correctness data to self.save_dir."""
        pe_data = load_dataset(
            "PromptEval/PromptEval_MMLU_Correctness", "abstract_algebra"
        )
        pe_data = cast(Dataset, pe_data)
        pe_data.to_csv(
            os.path.join(self.save_dir, "prompteval_correctness_data.csv"), index=False
        )

    # TODO: switch to proper caching
    def process_data_scalar(
        self, cached: bool = False, agg: str = "mean"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Processes the data into scalar matrix setting. Returns a tuple of ndarrays representing 2d data matrix and mask.

        Args:
            cached (bool): whether to use cached data. If False, then directly use urls, otherwise use self.save_dir. Default: False.
            agg (str): aggregation method to use to create scalar dataset. Default: "mean".

        """
        if not cached:
            df_final = self._load_data()
        else:
            df_final = pd.read_csv(
                os.path.join(self.save_dir, "prompteval_correctness_data.csv")
            )
        data3d = self._form_data(df_final)
        mask = self.generate_mask(data3d, self.p)
        if agg == "mean":
            data = np.nanmean(data3d, axis=2)
        elif agg == "sum":
            data = np.nansum(data3d, axis=2)
        elif agg == "median":
            data = np.nanmedian(data3d, axis=2)
        elif agg == "std":
            data = np.nanstd(data3d, axis=2)
        elif agg == "variance":
            data = np.nanvar(data3d, axis=2)
        else:
            raise ValueError(
                "agg must be one of 'mean', 'sum', 'median', 'std', or 'variance'"
            )
        self.data = data
        self.mask = mask
        if self.save_processed:
            np.save(
                os.path.join(self.save_dir, "prompteval_correctness_data.npy"), data
            )
            np.save(
                os.path.join(self.save_dir, "prompteval_correctness_mask.npy"), mask
            )
        return self.data, self.mask

    def process_data_distribution(
        self, cached: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Processes the data into distribution matrix setting. Returns a tuple of ndarrays representing 2d data matrix and mask."

        Args:
        ----
            cached (bool): whether to use cached data. If False, then directly use urls, otherwise use self.save_dir. Default: False.

        """
        if not cached:
            df_final = self._load_data()
        else:
            df_final = pd.read_csv(
                os.path.join(self.save_dir, "prompteval_correctness_data.csv")
            )

        data = self._form_data(df_final)
        self.mask = self.generate_mask(data, self.p)
        data2d = np.empty((data.shape[0], data.shape[1]), dtype=object)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data2d[i, j] = data[i, j]
        self.data = data2d
        if self.save_processed:
            np.save(
                os.path.join(
                    self.save_dir, "prompteval_correctness_data_distribution.npy"
                ),
                data2d,
                allow_pickle=True,
            )
            np.save(
                os.path.join(
                    self.save_dir, "prompteval_correctness_mask_distribution.npy"
                ),
                self.mask,
                allow_pickle=True,
            )
        return self.data, self.mask
    
    def get_full_state_as_dict(self, include_metadata: bool = False) -> dict:
        """Returns the full state of the data loader as a dictionary.

        Args:
            include_metadata (bool): whether to include metadata in the returned dictionary. Default: False.

        """
        state_dict = {
            "data": self.data,
            "mask": self.mask,
            "task_list": self.task_list,
            "seed": self.seed,
            "mask_strategy": self.mask_strategy,
        }
        if include_metadata:
            state_dict["metadata"] = {
                "download": self.download,
                "save_dir": self.save_dir,
                "save_processed": self.save_processed,
            }
        return state_dict

    def _form_data(self, df_final: pd.DataFrame) -> np.ndarray:
        """Forms the data and mask from the loaded DataFrame.

        Args:
            df_final (pd.DataFrame): The final DataFrame containing the loaded PromptEval data.

        """
        formats = df_final["format"].unique()
        models = df_final["model"].unique()
        tasks = df_final["task"].unique()
        data = np.empty((len(formats), len(models), len(tasks)))
        for id_model, model in enumerate(models):
            for id_format, format in enumerate(formats):
                df_sub = df_final[
                    (df_final["format"] == format) & (df_final["model"] == model)
                ]
                task_examples = []
                for id_task, task in enumerate(tasks):
                    df_sub_task = df_sub[df_sub["task"] == task]
                    task_examples = df_sub_task["correctness"]
                    # TODO: generalize the mean to other aggregation methods / or to 4d somehow
                    data[id_format, id_model, id_task] = np.mean(task_examples)
        return data

    def _load_data(self) -> pd.DataFrame:
        final_list = []
        for config in self.task_list:
            df = self._load_config_data(config)
            if df is not None:
                final_list.append(df)
                # TODO: Caleb switch to logging
                # print(f"Loaded data for config: {config}")
            # else:
            # TODO: Caleb switch to logging
            # print(f"Failed to load data for config: {config}")

        df_final = pd.concat(final_list, ignore_index=True)
        return df_final

    def _load_config_data(self, config_name: str) -> pd.DataFrame | None:
        """Loads the dataset for a specific task configuration and returns it as a DataFrame.

        Args:
            config_name (str): The name of the task configuration to load.

        """
        try:
            # Load the dataset with the specific config
            dataset = load_dataset(
                "PromptEval/PromptEval_MMLU_correctness", config_name
            )
            list_dfs = []
            for key in dataset:
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
                        #print(f"Data for key '{key}' is not a DataFrame")
                        raise ValueError(
                            f"Data for key '{key}' is not a DataFrame"
                        )
                else:
                    raise ValueError("Loaded dataset is not a DatasetDict")
            df_final = pd.concat(list_dfs, ignore_index=True)
            return df_final
        except Exception as e:
            #print(f"Error loading config '{config_name}': {e}")
            #TODO: Caleb logging
            return None

    def _generate_mask_mcar(self, data: np.ndarray, p: float) -> np.ndarray:
        """Generates a mask for the data using MCAR (Missing Completely At Random) strategy."""
        mask = np.ones([data.shape[0], data.shape[1]], dtype=int)
        # Randomly set some values to 0 (indicating missing)
        rng = np.random.default_rng(seed=self.seed)
        mask[rng.random(mask.shape) < p] = 0
        return mask
