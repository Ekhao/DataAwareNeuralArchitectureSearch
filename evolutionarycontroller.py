# A class that implements a controller (search strategy) based on evolutionary algorithms.

# Standard Library Imports
import random
import copy
import math

# Third Party Imports
import numpy as np

# Local Imports
import controller
import constants


class EvolutionaryController(controller.Controller):
    # Generates an initial population. The "trivial" parameter is a boolean that decides whether the initial population is generated out of random one layer models (True) or general random models (False)
    def __init__(self, search_space, seed=None, population_size=constants.POPULATION_SIZE, max_num_layers=constants.MAX_NUM_LAYERS, crossover_ratio=constants.CROSSOVER_RATIO, tournament_amount=constants.POPULATION_UPDATE_RATIO) -> None:
        super().__init__(search_space)
        random.seed(seed)
        self.seed = seed
        self.currently_evaluating = None
        self.unevaluated_configurations = []
        self.population = []
        self.population_size = population_size
        self.max_num_layers = max_num_layers
        self.crossover_ratio = crossover_ratio
        self.tournament_amount = max(
            1, round(population_size * tournament_amount))

    def initialize_controller(self, trivial_initialization=True):
        # A paper I read claims that it is good to start from an initial trivial solution. Therefore the initial population created here only contains models with only one layer.
        self.trivial_initialization = trivial_initialization
        if trivial_initialization:
            for i in range(self.population_size):
                self.unevaluated_configurations.append((random.randrange(
                    0, len(self.search_space.data_search_space_enumerated)), [random.randrange(0, len(self.search_space.model_layer_search_space_enumerated))]))
        # Another common way to generate an intial configuration for evolutionary algorithms is to generate random models from the search space.
        else:
            for i in range(self.population_size):
                number_of_layers = random.randint(1, self.max_num_layers)
                model_layer_configuration = []
                for layer in range(number_of_layers):
                    model_layer_configuration.append(random.randrange(
                        0, len(self.search_space.model_layer_search_space_enumerated)))
                self.unevaluated_configurations.append((random.randrange(
                    0, len(self.search_space.data_search_space_enumerated)), model_layer_configuration))

    # Fetches an element that has not yet been evaluated from the population
    def generate_configuration(self):
        # When an entire population has been evaluated we generate a new population
        if not self.unevaluated_configurations:
            self._generate_new_unevaluated_configurations()

        return self.unevaluated_configurations.pop(0)

    # Updates the data_model with its measured performance.
    # Generates a new population if all of the current population has been evaluated.

    def update_parameters(self, data_model):
        # Add performance of the currently evaluating data model to the population
        fitness = self._evaluate_fitness(data_model)
        self.population.append((data_model, fitness))

    @staticmethod
    def _evaluate_fitness(data_model):
        model_size_score = math.exp(-data_model.model_size /
                                    constants.MODEL_SIZE_APPROXIMATE_RANGE)
        return data_model.accuracy + data_model.precision + data_model.recall + model_size_score

    def _generate_new_unevaluated_configurations(self):
        # If there is no current population to generate new unevaluated configurations from we need to generate a new initial unevaluated configuration
        if not self.population:
            self.initialize_controller(self.trivial_initialization)
            return
        # Use tournament selection to decide which population to breed
        breeders = self._tournament_selection()

        # After this we would like breeders to be configurations instead of a tuple of a DataModel and a fitness
        breeder_configurations = self._get_breeder_configurations(breeders)

        amount_of_new_individuals = self.population_size - \
            len(breeder_configurations)
        amount_of_mutations = round(amount_of_new_individuals *
                                    (1 - self.crossover_ratio))
        amount_of_crossovers = round(
            amount_of_new_individuals * self.crossover_ratio)

        new_mutations = self._create_mutations(
            configurations_to_mutate=breeder_configurations, amount=amount_of_mutations)
        new_crossovers = self._create_crossovers(
            configurations_to_crossover=breeder_configurations, amount=amount_of_crossovers)

        self.population.clear()
        self.population.extend(breeders)
        self.unevaluated_configurations = new_mutations + new_crossovers

    def _tournament_selection(self):
        tournaments = np.array_split(self.population, self.tournament_amount)

        winners = []
        for tournament in tournaments:
            best_tournament_fitness = 0
            best_contestant = None
            for contestant in tournament:
                if contestant[1] > best_tournament_fitness:
                    best_tournament_fitness = contestant[1]
                    best_contestant = contestant
            if not best_contestant is None:
                winners.append(list(best_contestant))

        return winners

    # This function takes a list of tuples of DataModels and their fitness.
    # It should return the configurations that generated those data models to create mutations and crossovers of them
    def _get_breeder_configurations(self, breeders):
        return [[breeder[0].data_configuration,
                 breeder[0].model_configuration] for breeder in breeders]

    def _create_mutations(self, configurations_to_mutate, amount):
        # Generate a random number to choose which mutation to use:
        mutations = []
        for i in range(amount):
            configuration_to_mutate = configurations_to_mutate[i % amount]
            mutation = configuration_to_mutate

            # Sometimes a mutation does nothing, so continue until a change is made.
            # A mutation may be doing nothing if it is randomly chosen to decrease a value that is already at its minimum
            while mutation == configuration_to_mutate:
                random_mutation_number = random.random()
                match random_mutation_number:
                    case x if 0 <= x < 0.1:
                        mutation = self._new_convolutional_layer_mutation(
                            configuration_to_mutate)
                    case x if 0.1 <= x < 0.2:
                        mutation = self._remove_convolutional_layer_mutation(
                            configuration_to_mutate)
                    case x if 0.2 <= x < 0.3:
                        mutation = self._increase_filter_size_mutation(
                            configuration_to_mutate)
                    case x if 0.3 <= x < 0.4:
                        mutation = self._decrease_filter_size_mutation(
                            configuration_to_mutate)
                    case x if 0.4 <= x < 0.5:
                        mutation = self._increase_number_of_filters_mutation(
                            configuration_to_mutate)
                    case x if 0.5 <= x < 0.6:
                        mutation = self._decrease_number_of_filters_mutation(
                            configuration_to_mutate)
                    case x if 0.6 <= x < 0.7:
                        mutation = self._change_activation_function_mutation(
                            configuration_to_mutate)
                    case x if 0.7 <= x < 0.8:
                        mutation = self._increase_sample_rate_mutation(
                            configuration_to_mutate)
                    case x if 0.8 <= x < 0.9:
                        mutation = self._decrease_sample_rate_mutation(
                            configuration_to_mutate)
                    case x if 0.9 <= x < 1:
                        mutation = self._change_preprocessing_mutation(
                            configuration_to_mutate)
            mutations.append(mutation)

        return mutations
    # Generate a random new convolutional layer and add it to the end of the convolutional part of the model.

    def _new_convolutional_layer_mutation(self, configuration):
        mutation = copy.deepcopy(configuration)
        new_conv_layer = random.randrange(
            0, len(self.search_space.model_layer_search_space_enumerated))
        assert type(mutation[1]) == list
        if len(mutation[1]) < self.max_num_layers:
            mutation[1].append(new_conv_layer)
        return mutation

    # Remove the last convolutional layer of the model
    def _remove_convolutional_layer_mutation(self, configuration):
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1]) == list
        if len(mutation[1]) > 1:
            mutation[1].pop()
        return mutation

    # Increase the filter size of a random convolutional layer
    def _increase_filter_size_mutation(self, configuration):
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(
            mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change filter size. Filter size in the current search space is in the second position of the search space tuple.
        current_filter_size = decoded_layer[1]
        new_filter_size = None
        for seach_space_filter_size in self.search_space.model_layer_search_space_options[1]:
            if seach_space_filter_size > current_filter_size:
                new_filter_size = seach_space_filter_size
                break
        if new_filter_size == None:
            new_filter_size = max(
                self.search_space.model_layer_search_space_options[1])

        # Encode layer again
        decoded_layer = (
            decoded_layer[0], new_filter_size, decoded_layer[2])
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = new_layer

        return mutation

    def _decrease_filter_size_mutation(self, configuration):
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(
            mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change filter size. Filter size in the current search space is in the second position of the search space tuple.
        current_filter_size = decoded_layer[1]
        new_filter_size = None
        for seach_space_filter_size in reversed(self.search_space.model_layer_search_space_options[1]):
            if seach_space_filter_size < current_filter_size:
                new_filter_size = seach_space_filter_size
                break
        if new_filter_size == None:
            new_filter_size = min(
                self.search_space.model_layer_search_space_options[1])

        # Encode layer again
        decoded_layer = (decoded_layer[0], new_filter_size, decoded_layer[2])
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = new_layer

        return mutation

    def _increase_number_of_filters_mutation(self, configuration):
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(
            mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change filter amount. Amount of filters in the current search space is in the first position of the search space tuple.
        current_filter_amount = decoded_layer[0]
        new_filter_amount = None
        for seach_space_filter_amount in self.search_space.model_layer_search_space_options[0]:
            if seach_space_filter_amount > current_filter_amount:
                new_filter_amount = seach_space_filter_amount
                break
        if new_filter_amount == None:
            new_filter_amount = max(
                self.search_space.model_layer_search_space_options[0])

        # Encode layer again
        decoded_layer = (new_filter_amount, decoded_layer[1], decoded_layer[2])
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = new_layer

        return mutation

    def _decrease_number_of_filters_mutation(self, configuration):
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(
            mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change filter amount. Amount of filters in the current search space is in the first position of the search space tuple.
        current_filter_amount = decoded_layer[0]
        new_filter_amount = None
        for seach_space_filter_amount in reversed(self.search_space.model_layer_search_space_options[0]):
            if seach_space_filter_amount < current_filter_amount:
                new_filter_amount = seach_space_filter_amount
                break
        if new_filter_amount == None:
            new_filter_amount = min(
                self.search_space.model_layer_search_space_options[0])

        # Encode layer again
        decoded_layer = (
            new_filter_amount, decoded_layer[1], decoded_layer[2])
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = new_layer

        return mutation

    def _change_activation_function_mutation(self, configuration):
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(
            mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Decode layer
        decoded_layer = self.search_space.model_layer_decode(layer_to_modify)

        # Change activation function. Activation function in the current search space is in the third position of the search space tuple.
        current_activation_function = decoded_layer[2]
        new_activation_function = current_activation_function

        while current_activation_function == new_activation_function:
            new_activation_function = random.choice(
                self.search_space.model_layer_search_space_options[2])

        # Encode layer again
        decoded_layer = (
            decoded_layer[0], decoded_layer[1], new_activation_function)
        new_layer = self.search_space.model_layer_encode(decoded_layer)

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = new_layer

        return mutation

    def _increase_sample_rate_mutation(self, configuration):
        # Decode data configuration
        decoded_data = self.search_space.data_decode(configuration[0])

        # Change sample rate. Sample rate in the current search space is in the first position of the search space tuple.
        current_sample_rate = decoded_data[0]
        new_sample_rate = None
        for seach_space_sample_rate in reversed(self.search_space.data_search_space_options[0]):
            if seach_space_sample_rate > current_sample_rate:
                new_sample_rate = seach_space_sample_rate
                break
        if new_sample_rate == None:
            new_sample_rate = max(
                self.search_space.data_search_space_options[0])

        # Encode layer again
        decoded_data = (new_sample_rate, decoded_data[1])
        new_data = self.search_space.data_encode(decoded_data)

        # Add the layer to the configuration again
        configuration = (new_data, configuration[1])

        return configuration

    def _decrease_sample_rate_mutation(self, configuration):
        # Decode data configuration
        decoded_data = self.search_space.data_decode(configuration[0])

        # Change sample rate. Sample rate in the current search space is in the first position of the search space tuple.
        current_sample_rate = decoded_data[0]
        new_sample_rate = None
        for seach_space_sample_rate in self.search_space.data_search_space_options[0]:
            if seach_space_sample_rate < current_sample_rate:
                new_sample_rate = seach_space_sample_rate
                break
        if new_sample_rate == None:
            new_sample_rate = min(
                self.search_space.data_search_space_options[0])

        # Encode layer again
        decoded_data = (new_sample_rate, decoded_data[1])
        new_data = self.search_space.data_encode(decoded_data)

        # Add the layer to the configuration again
        configuration = (new_data, configuration[1])

        return configuration

    def _change_preprocessing_mutation(self, configuration):
        # Decode data configuration
        decoded_data = self.search_space.data_decode(configuration[0])

        # Change preprocessing. Preprocessing in the current search space is in the second position of the search space tuple.
        current_preprocessing = decoded_data[1]
        new_preprocessing = current_preprocessing
        if len(self.search_space.data_search_space_options[1]) == 1:
            return configuration

        while new_preprocessing == current_preprocessing:
            new_preprocessing = random.choice(
                self.search_space.data_search_space_options[1])

        # Encode layer again
        decoded_data = (decoded_data[0], new_preprocessing)
        new_data = self.search_space.data_encode(decoded_data)

        # Add the layer to the configuration again
        configuration = (new_data, configuration[1])

        return configuration

    def _random_conv_layer_number(self, configuration):
        return random.randrange(0, len(configuration[1]))

    def _create_crossovers(self, configurations_to_crossover, amount):
        crossovers = []
        for i in range(amount):
            random_parents = random.choices(configurations_to_crossover, k=2)
            crossovers.append(self._crossover(*random_parents))

        return crossovers

    def _crossover(self, configuration1, configuration2):
        decoded_data1 = self.search_space.data_decode(configuration1[0])
        decoded_data2 = self.search_space.data_decode(configuration2[0])

        decoded_model1 = self.search_space.model_decode(configuration1[1])
        decoded_model2 = self.search_space.model_decode(configuration2[1])

        new_data = (random.choice((decoded_data1[0], decoded_data2[0])), random.choice((
            decoded_data1[1], decoded_data2[1])))

        num_layers_model1 = len(decoded_model1)
        num_layers_model2 = len(decoded_model2)

        if num_layers_model1 == num_layers_model2:
            num_layers_new_model = num_layers_model1
            min_layers = num_layers_model1
        else:
            min_layers = min(num_layers_model1, num_layers_model2)
            max_layers = max(num_layers_model1, num_layers_model2)

            num_layers_new_model = random.randint(min_layers, max_layers)

        new_model = []
        for i in range(min_layers):
            layer = (random.choice((decoded_model1[i][0], decoded_model2[i][0])), random.choice(
                (decoded_model1[i][1], decoded_model2[i][1])), random.choice((decoded_model1[i][2], decoded_model2[i][2])))
            new_model.append(layer)

        if num_layers_model1 > num_layers_model2:
            for i in range(min_layers, num_layers_new_model):
                new_model.append(copy.deepcopy(decoded_model1[i]))
        elif num_layers_model2 > num_layers_model1:
            for i in range(min_layers, num_layers_new_model):
                new_model.append(copy.deepcopy(decoded_model2[i]))

        encoded_data = self.search_space.data_encode(new_data)
        encoded_model = self.search_space.model_encode(new_model)
        return (encoded_data, encoded_model)
