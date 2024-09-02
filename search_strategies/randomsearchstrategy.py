# A class that implements a completely random search strategy.

# Standard Library Imports
import random
from typing import Optional, Any

# Local Imports
import searchstrategy
from configuration import Configuration
from searchspace import SearchSpace
from datamodel import DataModel


class RandomSearchStrategy(searchstrategy.SearchStrategy):
    def __init__(
        self, search_space: SearchSpace, max_num_layers: int, seed: Optional[int] = None
    ) -> None:
        super().__init__(search_space, seed)
        random.seed(seed)
        self.max_num_layers = max_num_layers

    def generate_configuration(self) -> Configuration:
        # Generate a data configuration by randomly selecting a choice for each data granularity type.
        data_configuration = {
            key: random.choice(value)
            for key, value in self.search_space.data_search_space.items()
        }

        # Generate a random number of layers.
        number_of_layers = random.randint(1, self.max_num_layers)

        # Generate a model layer configuration by randomly selecting a choice for each layer.
        model_configuration = []
        for layer in range(number_of_layers):
            # Pick a random type of layer:
            layer_type = random.choice(
                list(self.search_space.model_search_space.keys())
            )

            # Create a dictionary to be added to the model configuration list
            layer = {}

            # Add the type of the layer to this dictionary
            layer["type"] = layer_type

            # Populate the rest of the layer dictionary with random choices from the possible choices for this layer
            for key, value in self.search_space.model_search_space[layer_type].items():
                layer[key] = random.choice(value)

            model_configuration.append(layer)

        return Configuration(
            data_configuration=data_configuration,
            model_configuration=model_configuration,
        )

    def update_parameters(self, data_model: DataModel):
        # The random search strategy does not have any parameters that should be updated
        pass
