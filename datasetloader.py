# A base class for dataset loaders for the Data Aware Neural Architecture Search. Used as a template for creating other dataset loaders.

# Standard Library Imports
from abc import ABC, abstractmethod
from typing import Any, Optional

# Third Party Imports
import numpy as np


class DatasetLoader(ABC):
    # We define no default behavior for the constructor. This also means that there is no need to call super().__init__().
    def __init__(self) -> None:
        pass

    def supervised_dataset(
        self, *data: Any, test_size: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore This is an abstract base class that is not supposed to provide the required functionality.
        pass

    def load_dataset(self, **kwargs: Any) -> tuple[list, ...]:  # type: ignore This is an abstract base class that is not supposed to provide the required functionality.
        pass

    def num_samples_per_class(self) -> dict[int, int]:  # type: ignore This is an abstract base class that is not supposed to provide the required functionality.
        pass
