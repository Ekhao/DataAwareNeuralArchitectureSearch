from abc import ABC, abstractmethod


class Controller(ABC):

    @abstractmethod
    def generate_configuration(self):
        pass

    @abstractmethod
    def update_parameters(self, loss):
        pass

    def get_number_of_search_space_combinations(self, search_space):
        x = 0
        for element in search_space:
            if x == 0:
                x = len(element)
            else:
                x *= len(element)
        return x
