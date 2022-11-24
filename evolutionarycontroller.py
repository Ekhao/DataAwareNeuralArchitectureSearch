# A class that implements a controller based on evolutionary algorithms.
import controller
from constants import *
import random
import queue
import numpy as np


class EvolutionaryController(controller.Controller):
    # Generates an initial population. The "trivial" parameter is a boolean that decides whether the initial population is generated out of random one layer models (True) or general random models (False)
    def __init__(self, search_space, seed=None, trivial_initialization=True, population_size=POPULATION_SIZE, input_search_space=INPUT_SEARCH_SPACE, model_layer_search_space=MODEL_LAYER_SEARCH_SPACE, max_num_layers=MAX_NUM_LAYERS, crossover_ratio=CROSSOVER_RATIO, tournament_size=TOURNAMENT_SIZE) -> None:
        super().__init__(search_space)
        random.seed(seed)
        self.currently_evaluating = None
        self.unevaluated_population = queue.SimpleQueue()
        self.population = []
        self.population_size = population_size
        self.input_search_space = input_search_space
        self.model_layer_search_space = model_layer_search_space
        self.max_num_layers = max_num_layers
        self.crossover_ratio = crossover_ratio
        self.tournament_size = tournament_size

        # A paper I read claims that it is good to start from an initial trivial solution. Therefore the initial population created here only contains models with only one layer.
        # Due to the general way that the search space is defined I do not believe that it is possible to generate trivial inputs or individual layers without other assumptions.
        if trivial_initialization:
            for i in range(population_size):
                self.unevaluated_population.put((random.randrange(
                    0, super().get_number_of_search_space_combinations(input_search_space)), [random.randrange(0, super().get_number_of_search_space_combinations(model_layer_search_space))]))
        # Another common way to generate an intial configuration for evolutionary algorithms is to generate random models from the search space.
        else:
            for i in range(population_size):
                number_of_layers = random.randint(1, max_num_layers)
                model_layer_configuration = []
                for layer in range(number_of_layers):
                    model_layer_configuration.append(random.randrange(
                        0, super().get_number_of_search_space_combinations(model_layer_search_space)))
                self.unevaluated_population.put((random.randrange(
                    0, super().get_number_of_search_space_combinations(input_search_space)), model_layer_configuration))

    # Fetches an element that has not yet been evaluated from the population
    def generate_configuration(self):
        return self.unevaluated_population.get(block=False)

    # Updates the input_model with its measured performance.
    # Generates a new population if all of the current population has been evaluated.
    def update_parameters(self, input_model):
        # Add performance of the currently evaluating input model to the population
        fitness = self.__evaluate_fitness(input_model)
        self.population.append((input_model, fitness))

        # When an entire population has been evaluated we generate a new population
        if self.unevaluated_population.empty():
            self.__generate_new_population()

    def __evaluate_fitness(self, input_model):
        return input_model.accuracy + input_model.precision + input_model.recall

    def __generate_new_population(self):
        # Use tournament selection to decide which population to breed
        breeders = self.__tournament_selection()

        amount_of_new_individuals = self.population_size - len(breeders)
        amount_of_mutations = amount_of_new_individuals / \
            (1 - self.crossover_ratio)
        amount_of_crossovers = amount_of_new_individuals / self.crossover_ratio

        new_mutations = self.__create_mutations(
            individuals=breeders, amount=amount_of_mutations)
        new_crossovers = self.__create_crossovers(
            breeders=breeders, amount=amount_of_crossovers)

        # self.population.clear()
        self.population = breeders + new_mutations + new_crossovers

    def __tournament_selection(self):
        tournaments = np.array_split(self.population, self.tournament_size)

        winners = map(lambda x: x[1].max(), tournaments)

        return winners

    def __create_mutations(self, individuals, amount):
        # Generate a random number to choose which mutation to use:
        mutations = []
        for i in range(amount):
            random_mutation_number = random.random()
            random_individual = individuals[random.randrange(len(individuals))]

            match random_mutation_number:
                case x if 0 <= x < 0.1:
                    mutation = self.__new_convolutional_layer_mutation(
                        random_individual)
                case x if 0.1 <= x < 0.2:
                    mutation = self.__remove_convolutional_layer_mutation(
                        random_individual)
                case x if 0.2 <= x < 0.3:
                    mutation = self.__increase_filter_size_mutation(
                        random_individual)
                case x if 0.3 <= x < 0.4:
                    mutation = self.__decrease_filter_size_mutation(
                        random_individual)
                case x if 0.4 <= x < 0.5:
                    mutation = self.__increase_number_of_filters_mutation(
                        random_individual)
                case x if 0.5 <= x < 0.6:
                    mutation = self.__decrease_number_of_filters_mutation(
                        random_individual)
                case x if 0.6 <= x < 0.7:
                    mutation = self.__change_activation_function_mutation(
                        random_individual)
                case x if 0.7 <= x < 0.8:
                    mutation = self.__increase_sample_rate_mutation(
                        random_individual)
                case x if 0.8 <= x < 0.9:
                    mutation = self.__decrease_sample_rate_mutation(
                        random_individual)
                case x if 0.9 <= x < 1:
                    mutation = self.__change_preprocessing_mutation()
            mutations.append(mutation)

        return mutations

    def __create_crossovers(self, breeders, amount):
        raise NotImplementedError()

    # Generate a random new convolutional layer and add it to the end of the convolutional part of the model.
    def __new_convolutional_layer_mutation(self, configuration):
        new_conv_layer = random.randrange(
            0, super().get_number_of_search_space_combinations(self.model_layer_search_space))
        assert type(configuration[1]) == list
        return configuration[1].append(new_conv_layer)

    # Remove the last convolutional layer of the model
    def __remove_convolutional_layer_mutation(self, configuration):
        assert type(configuration[1]) == list
        configuration[1].pop()
        return configuration[1]

    # Increase the filter size of a random convolutional layer
    def __increase_filter_size_mutation(self, configuration):
        assert type(configuration[1] == list)
        random_conv_layer_number = self.__random_conv_layer_number(
            configuration)
        layer_to_modify = configuration[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change filter size. Filter size in the current search space is in the second position of the search space tuple.
        current_filter_size = decoded_layer[1]
        new_filter_size = next(
            (x for x in self.search_space.model_layer_search_space[1] if x > current_filter_size), max(self.search_space.model_layer_search_space[1]))

        # Encode layer again
        decoded_layer[1] = new_filter_size
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        configuration[1][random_conv_layer_number] = new_layer

        return configuration

    def __decrease_filter_size_mutation(self, configuration):
        assert type(configuration[1] == list)
        random_conv_layer_number = self.__random_conv_layer_number(
            configuration)
        layer_to_modify = configuration[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change filter size. Filter size in the current search space is in the second position of the search space tuple.
        current_filter_size = decoded_layer[1]
        new_filter_size = next(
            (x for x in self.search_space.model_layer_search_space[1].reverse() if x < current_filter_size), min(self.search_space.model_layer_search_space[1]))

        # Encode layer again
        decoded_layer[1] = new_filter_size
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        configuration[1][random_conv_layer_number] = new_layer

        return configuration

    def __increase_number_of_filters_mutation(self, configuration):
        assert type(configuration[1] == list)
        random_conv_layer_number = self.__random_conv_layer_number(
            configuration)
        layer_to_modify = configuration[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change amount of filters. Amount of filters in the current search space is in the first position of the search space tuple.
        current_filter_amount = decoded_layer[0]
        new_filter_size = next(
            (x for x in self.search_space.model_layer_search_space[0] if x > current_filter_amount), max(self.search_space.model_layer_search_space[0]))

        # Encode layer again
        decoded_layer[0] = new_filter_size
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        configuration[1][random_conv_layer_number] = new_layer

        return configuration

    def __decrease_number_of_filters_mutation(self, configuration):
        assert type(configuration[1] == list)
        random_conv_layer_number = self.__random_conv_layer_number(
            configuration)
        layer_to_modify = configuration[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change filter size. Amount of filters in the current search space is in the first position of the search space tuple.
        current_filter_amount = decoded_layer[0]
        new_filter_size = next(
            (x for x in self.search_space.model_layer_search_space[0].reverse() if x < current_filter_amount), min(self.search_space.model_layer_search_space[0]))

        # Encode layer again
        decoded_layer[0] = new_filter_size
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        configuration[1][random_conv_layer_number] = new_layer

        return configuration

    def __change_activation_function_mutation(self, configuration):
        assert type(configuration[1] == list)
        random_conv_layer_number = self.__random_conv_layer_number(
            configuration)
        layer_to_modify = configuration[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change activation function. Activation function in the current search space is in the third position of the search space tuple.
        current_activation_function = decoded_layer[2]
        new_activation_function = current_activation_function
        while current_activation_function == new_activation_function:
            new_activation_function = random.choice(
                self.search_space.model_layer_search_space[2])

        # Encode layer again
        decoded_layer[2] = new_activation_function
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        configuration[1][random_conv_layer_number] = new_layer

        return configuration

    def __increase_sample_rate_mutation(self, configuration):
        # Decode input configuration
        decoded_input = self.search_space.input_decode(configuration[0])

        # Change sample rate. Sample rate in the current search space is in the first position of the search space tuple.
        current_sample_rate = decoded_input[0]
        new_sample_rate = next(
            (x for x in self.search_space.input_search_space[0] if x > current_sample_rate), max(self.search_space.input_search_space[0]))

        # Encode layer again
        decoded_input[0] = new_sample_rate
        new_input = self.search_space.input_encode(decoded_input)

        # Add the layer to the configuration again
        configuration[0] = new_input

        return configuration

    def __decrease_sample_rate_mutation(self, configuration):
        # Decode input configuration
        decoded_input = self.search_space.input_decode(configuration[0])

        # Change sample rate. Sample rate in the current search space is in the first position of the search space tuple.
        current_sample_rate = decoded_input[0]
        new_sample_rate = next(
            (x for x in self.search_space.input_search_space[0].reverse() if x < current_sample_rate), min(self.search_space.input_search_space[0]))

        # Encode layer again
        decoded_input[0] = new_sample_rate
        new_input = self.search_space.input_encode(decoded_input)

        # Add the layer to the configuration again
        configuration[0] = new_input

        return configuration

    def __change_preprocessing_mutation(self, configuration):
        # Decode input configuration
        decoded_input = self.search_space.input_decode(configuration[0])

        # Change preprocessing. Preprocessing in the current search space is in the second position of the search space tuple.
        current_preprocessing = decoded_input[1]
        new_preprocessing = current_preprocessing
        while new_preprocessing == current_preprocessing:
            new_preprocessing = random.choice(
                self.search_space.input_search_space[1])

        # Encode layer again
        decoded_input[1] = new_preprocessing
        new_input = self.search_space.input_encode(decoded_input)

        # Add the layer to the configuration again
        configuration[0] = new_input

        return configuration

    def __random_conv_layer_number(self, configuration):
        return random.randrange(0, len(configuration[1]))

    # Maybe split evaluation functions into cheap and expensive to calculate functions and evaluate the cheap functions more often. Like in LEMONADE.

    # Methods to improve evaluation speed: Have offspring inherit the weights of their parents, Early stopping of training e.g. after a few epochs, reduce the size of the training set, reducing the size of the ENAS population (Not sure that this is such a good idea), not reevaluating the same individuals several times.
