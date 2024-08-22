# A datasetloader to load the wake vision dataset

# Should integrate the datasetloader of this project with either the tfds or huggingface version of wake vision

# Standard Library Imports
from typing import Optional, Any

# Third Party Imports
import datasets

# Local Imports
import datasetloader


class WakeVisionDatasetLoader(datasetloader.DatasetLoader):
    def __init__(
        self,
    ) -> None:
        self.dataset = datasets.load_dataset(
            path="Harvard-Edge/Wake-Vision",
            cache_dir="/dtu-compute/emjn/huggingface/datasets",
        )


if __name__ == "__main__":
    dataset_loader = WakeVisionDatasetLoader()
