# A class that implements a controller based on evolutionary algorithms.
import controller
from constants import *
import random


class EvolutionaryController(controller.Controller):
    # Generates an initial population. The "trivial" parameter is a boolean that decides whether the initial population is generated out of random one layer models (True) or general random models (False)
    def __init__(self, seed=None, trivial=True, population_size=POPULATION_SIZE) -> None:

        # A paper I read claims that it is good to start from an initial trivial solution. Therefore the initial population created here only contains models with only one layer.
        # Due to the general way that the search space is defined I do not believe that it is possible to generate trivial inputs or individual layers without other assumptions.
        random.seed(seed)
        population = []
        if trivial:
            for i in range(POPULATION_SIZE):
                population.append((random.randint(
                    0, super().get_number_of_search_space_combinations(INPUT_SEARCH_SPACE) - 1), [random.randint(0, super().get_number_of_search_space_combinations(MODEL_LAYER_SEARCH_SPACE) - 1)]))
        # Another common way to generate an intial configuration for evolutionary algorithms is to generate random models from the search space.
        else:
            for i in range(POPULATION_SIZE):
                number_of_layers = random.randint(1, MAX_NUM_LAYERS)
                model_layer_configuration = []
                for layer in range(number_of_layers):
                    model_layer_configuration.append(random.randint(
                        0, super().get_number_of_search_space_combinations(MODEL_LAYER_SEARCH_SPACE) - 1))

        # We create a tuple of each entry in the population and the False boolean. This is used in the generate_configuration method to signal that this entry has not been evauluated yet.
        population = zip(population, [False] * len(population))

    # Fetches an element that has not yet been evaluated from the population
    def generate_configuration(self):
        # Generate a candidate off the new population (either mutation og crossing of parents in the old population)

        # Serve an unevaluated candidate from the current population
        raise NotImplementedError()

    def update_parameters(self, loss):
        # Add the candidate to the population based on its performance
        # Maybe use tournament selection to decide which population to breed

        # When an entire population has been evaluated we generate a new population
        raise NotImplementedError()


# It seems that ENAS often generates an initial group of solutions - evaluate all of them.
# Then generate a new group of solutions and evaluate all of them.
# This is a little different than the framework set up here.

# Maybe split evaluation functions into cheap and expensive to calculate functions and evaluate the cheap functions more often. Like in LEMONADE.

# Methods to improve evaluation speed: Have offspring inherit the weights of their parents, Early stopping of training e.g. after a few epochs, reduce the size of the training set, reducing the size of the ENAS population (Not sure that this is such a good idea), not reevaluating the same individuals several times.
