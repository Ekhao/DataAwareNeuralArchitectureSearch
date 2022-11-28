from abc import ABC, abstractmethod


class Controller(ABC):

    def __init__(self, search_space) -> None:
        self.search_space = search_space

    @abstractmethod
    def generate_configuration(self):
        pass

    @abstractmethod
    def update_parameters(self, input_model):
        pass
