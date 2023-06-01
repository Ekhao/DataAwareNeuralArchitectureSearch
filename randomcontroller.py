# A class that implements a completely random controller (search strategy).

# Standard Library Imports
import random
from typing import Optional, Any

# Local Imports
import controller
from searchspace import SearchSpace
from datamodel import DataModel

Configuration = tuple[tuple[Any, ...], list[tuple[Any, ...]]]


class RandomController(controller.Controller):
    def __init__(
        self, search_space: SearchSpace, max_num_layers: int, seed: Optional[int] = None
    ) -> None:
        super().__init__(search_space, seed)
        random.seed(seed)
        self.max_num_layers = max_num_layers

    def generate_configuration(self) -> Configuration:
        # Generate a data configuration by randomly selecting a choice for each data granularity type.
        data_configuration = tuple(
            random.choice(x) for x in self.search_space.data_granularity_search_space
        )

        # Generate a random number of layers.
        number_of_layers = random.randint(1, self.max_num_layers)

        # Generate a model layer configuration by randomly selecting a choice for each layer.
        model_layer_configuration = [
            tuple(random.choice(x) for x in self.search_space.model_layer_search_space)
            for layer in range(number_of_layers)
        ]

        return (data_configuration, model_layer_configuration)

    def update_parameters(self, data_model: DataModel):
        # The random controller does not have any parameters that should be updated
        pass
