# A datasetloader to load the wake vision dataset

# Should integrate the datasetloader of this project with either the tfds or huggingface version of wake vision

# Standard Library Imports
from typing import Optional, Any

# Local Imports
import datasetloader


class WakeVisionDatasetLoader(datasetloader.DatasetLoader):
    def __init__(
        self,
        file_path: str,
        num_files: list[int],
        dataset_options: dict[str, Any],
        num_cores_to_use: int,
    ) -> None:
        pass
