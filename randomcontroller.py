import controller
from constants import *
import random


class RandomController (controller.Controller):
    def __init__(self, search_space, seed=None):
        super().__init__(search_space)
        random.seed(seed)

    def generate_configuration(self):
        input_configuation = random.randrange(
            0, super().get_number_of_search_space_combinations(INPUT_SEARCH_SPACE))

        # Generate a number of layers. Maybe have this be something other than uniformly distributed at some point.
        number_of_layers = random.randint(1, MAX_NUM_LAYERS)
        model_layer_configuration = []
        for layer in range(number_of_layers):
            model_layer_configuration.append(random.randrange(
                0, super().get_number_of_search_space_combinations(MODEL_LAYER_SEARCH_SPACE)))

        return (input_configuation, model_layer_configuration)

    def update_parameters(self, input_model):
        # The random controller does not have any parameters that should be updated
        pass
