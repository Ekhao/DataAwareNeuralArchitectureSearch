# A base class for a controller - also known as a search strategy for the Data Aware Neural Architecture Search. Used as a template for creating other controllers.

# Standard Library Imports
from abc import ABC, abstractmethod

# Local Imports
from searchspace import SearchSpace


class Controller(ABC):

    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space

    @abstractmethod
    def generate_configuration(self):
        pass

    @abstractmethod
    def update_parameters(self, data_model):
        pass
