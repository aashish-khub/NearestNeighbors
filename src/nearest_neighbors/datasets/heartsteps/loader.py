from examples.base import NNDataLoader
from examples.factory import register_dataset

@register_dataset("heartsteps")
class HeartStepsDataLoader(NNDataLoader):
    urls = {
        "jbsteps.csv": "https://raw.githubusercontent.com/klasnja/HeartStepsV1/refs/heads/main/data_files/jbsteps.csv",
        "suggestions.csv": "https://raw.githubusercontent.com/klasnja/HeartStepsV1/refs/heads/main/data_files/suggestions.csv",
    }
    
    def download_data(self):
        # download the data from urls
        pass
    
    def process_data_scalar(self, cached:bool=False) -> tuple[np.ndarray, np.ndarray]:
        # process the data into scalar matrix setting
        pass
    
    def process_data_distribution(self, cached:bool=False) -> tuple[np.ndarray, np.ndarray]:
        # process the data into distributional setting
        pass
