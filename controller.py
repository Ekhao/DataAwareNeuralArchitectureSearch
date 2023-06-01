# A base class for a controller - also known as a search strategy for the Data Aware Neural Architecture Search. Used as a template for creating other controllers.

# Standard Library Imports
from abc import ABC, abstractmethod
from typing import Any

# Local Imports
from searchspace import SearchSpace
from datamodel import DataModel

Configuration = tuple[tuple[Any, ...], list[tuple[Any, ...]]]
Individual = tuple[DataModel, float]


class Controller(ABC):
    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space

    @abstractmethod
    def generate_configuration(self) -> Configuration:
        pass

    @abstractmethod
    def update_parameters(self, data_model: DataModel) -> None:
        pass
