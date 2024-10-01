# A base class for dataset loaders for the Data Aware Neural Architecture Search. Used as a template for creating other dataset loaders.

# Standard Library Imports
from abc import ABC
from typing import Any

# Local Imports
from data import Data


class DatasetLoader(ABC):
    # We define no default behavior for the constructor. This also means that there is no need to call super().__init__().
    def __init__(self) -> None:
        pass

    # This method is returns a configured dataset according to a specific data configuration.
    def configure_dataset(self, **kwargs: Any) -> Any:  # type: ignore This is an abstract base class that is not supposed to provide the required functionality.
        pass

    def supervised_dataset(
        self, *data: Any, test_size: float
    ) -> Data:  # type: ignore This is an abstract base class that is not supposed to provide the required functionality.
        pass
