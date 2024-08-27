# A datasetloader to load the wake vision dataset

# Standard Library Imports
from typing import Optional, Any

# Third Party Imports
import datasets
from numpy import ndarray

# Local Imports
import datasetloader
import data


class WakeVisionDatasetLoader(datasetloader.DatasetLoader):
    def __init__(
        self,
    ) -> None:
        self.dataset = datasets.load_dataset(
            path="Harvard-Edge/Wake-Vision",
            cache_dir="/dtu-compute/emjn/huggingface/datasets",
        )

    def configure_dataset(self, **kwargs: Any) -> Any:
        # TODO: For now we have no configuration of the wake vision dataset
        return self.dataset

    def supervised_dataset(self, *input_data: Any, test_size: float) -> data.Data:
        return data.Data(
            X_train=self.dataset["train_quality"]["image"],
            X_val=self.dataset["validation"]["image"],
            X_test=self.dataset["test"]["image"],
            y_train=self.dataset["train_quality"]["person"],
            y_val=self.dataset["validation"]["person"],
            y_test=self.dataset["test"]["person"],
        )


if __name__ == "__main__":
    dataset_loader = WakeVisionDatasetLoader()
