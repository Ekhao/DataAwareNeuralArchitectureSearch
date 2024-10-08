# A base class for a search strategies for the Data Aware Neural Architecture Search. Used as a template for creating other search strategies.

# Standard Library Imports
from abc import ABC, abstractmethod
from typing import Any, Optional

# Local Imports
from searchspace import SearchSpace
from datamodel import DataModel
from configuration import Configuration


class SearchStrategy(ABC):
    def __init__(self, search_space: SearchSpace, seed: Optional[int] = None) -> None:
        self.search_space = search_space
        self.seed = seed

    @abstractmethod
    def generate_configuration(self) -> Configuration:
        pass

    @abstractmethod
    def update_parameters(self, data_model: DataModel) -> None:
        pass
