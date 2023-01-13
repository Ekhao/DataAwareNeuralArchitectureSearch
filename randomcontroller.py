# A class that implements a completely random controller (search strategy).

# Standard Library Imports
import random

# Local Imports
import controller
import constants


class RandomController (controller.Controller):
    def __init__(self, search_space, seed=None, max_num_layers=constants.MAX_NUM_LAYERS):
        super().__init__(search_space)
        random.seed(seed)
        self.seed = seed
        self.max_num_layers = max_num_layers

    def generate_configuration(self):
        data_configuation = random.randrange(
            0, len(self.search_space.data_search_space_enumerated))

        # Generate a number of layers. Maybe have this be something other than uniformly distributed at some point.
        number_of_layers = random.randint(1, self.max_num_layers)
        model_layer_configuration = []
        for layer in range(number_of_layers):
            model_layer_configuration.append(random.randrange(
                0, len(self.search_space.model_layer_search_space_enumerated)))

        return (data_configuation, model_layer_configuration)

    def update_parameters(self, data_model):
        # The random controller does not have any parameters that should be updated
        pass
