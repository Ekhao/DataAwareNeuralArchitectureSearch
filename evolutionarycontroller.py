# A class that implements a controller (search strategy) based on evolutionary algorithms.

# Standard Library Imports
import random
import copy
import math
from typing import Optional, Any

# Third Party Imports
import numpy as np

# Local Imports
import controller
from searchspace import SearchSpace
from datamodel import DataModel

# Type Aliases
Configuration = tuple[tuple[Any, ...], list[tuple[Any, ...]]]
Individual = tuple[DataModel, float]


class EvolutionaryController(controller.Controller):
    # Generates an initial population. The "trivial" parameter is a boolean that decides whether the initial population is generated out of random one layer models (True) or general random models (False)
    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int,
        max_num_layers: int,
        population_update_ratio: float,
        crossover_ratio: float,
        approximate_model_size: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(search_space, seed)
        random.seed(seed)
        self.currently_evaluating = None
        self.unevaluated_configurations = []
        self.population = []
        self.population_size = population_size
        self.max_num_layers = max_num_layers
        self.crossover_ratio = crossover_ratio
        self.approximate_model_size = approximate_model_size
        self.tournament_amount = max(
            1, round(population_size * population_update_ratio)
        )

    def initialize_controller(self, trivial_initialization: bool = True) -> None:
        # A paper (Kenneth O Stanley, Jeff Clune, Joel Lehman, and Risto Miikkulainen. 2019. Designing neural networks through neuroevolution. Nature Machine Intelligence 1, 1 (2019), 24–35.) claims that it is good to start from an initial trivial solution. Therefore the initial population created here only contains models with only one layer.
        self.trivial_initialization = trivial_initialization
        if trivial_initialization:
            self.unevaluated_configurations = [
                (
                    tuple(
                        random.choice(x)
                        for x in self.search_space.data_granularity_search_space
                    ),
                    [
                        tuple(
                            random.choice(x)
                            for x in self.search_space.model_layer_search_space
                        )
                        for layer in range(1)
                    ],
                )
                for i in range(self.population_size)
            ]
        # Another common way to generate an intial configuration for evolutionary algorithms is to generate random models from the search space.
        else:
            self.unevaluated_configurations = [
                (
                    tuple(
                        random.choice(x)
                        for x in self.search_space.data_granularity_search_space
                    ),
                    [
                        tuple(
                            random.choice(x)
                            for x in self.search_space.model_layer_search_space
                        )
                        for layer in range(random.randint(1, self.max_num_layers))
                    ],
                )
                for i in range(self.population_size)
            ]

    # Fetches an element that has not yet been evaluated from the population
    def generate_configuration(self) -> Configuration:
        # When an entire population has been evaluated we generate a new population
        if not self.unevaluated_configurations:
            self._generate_new_unevaluated_configurations()

        return self.unevaluated_configurations.pop(0)

    # Updates the data_model with its measured performance.
    # Generates a new population if all of the current population has been evaluated.

    def update_parameters(self, data_model: DataModel) -> None:
        # Add performance of the currently evaluating data model to the population
        fitness = self._evaluate_fitness(data_model, self.approximate_model_size)
        self.population.append((data_model, fitness))

    @staticmethod
    # TODO: Maybe move this into the DataModel class
    def _evaluate_fitness(
        data_model: DataModel, model_size_approximate_range: int
    ) -> float:
        model_size_score = math.exp(
            -data_model.model_size / model_size_approximate_range
        )
        return (
            data_model.accuracy
            + data_model.precision
            + data_model.recall
            + model_size_score
        )

    def _generate_new_unevaluated_configurations(self) -> None:
        # If there is no current population to generate new unevaluated configurations from we need to generate a new initial unevaluated configuration
        if not self.population:
            self.initialize_controller(self.trivial_initialization)
            return
        # Use tournament selection to decide which population to breed
        breeders = self._tournament_selection()

        # After this we would like breeders to be configurations instead of a tuple of a DataModel and a fitness
        breeder_configurations = self._get_breeder_configurations(breeders)

        amount_of_new_individuals = self.population_size - len(breeder_configurations)
        amount_of_mutations = round(
            amount_of_new_individuals * (1 - self.crossover_ratio)
        )
        amount_of_crossovers = round(amount_of_new_individuals * self.crossover_ratio)

        new_mutations = self._create_mutations(
            configurations_to_mutate=breeder_configurations, amount=amount_of_mutations
        )
        new_crossovers = self._create_crossovers(
            configurations_to_crossover=breeder_configurations,
            amount=amount_of_crossovers,
        )

        self.population.clear()
        self.population.extend(breeders)
        self.unevaluated_configurations = new_mutations + new_crossovers

    def _tournament_selection(self) -> list[Individual]:
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
    def _get_breeder_configurations(
        self, breeders: list[Individual]
    ) -> list[Configuration]:
        return [
            (breeder[0].data_configuration, breeder[0].model_configuration)
            for breeder in breeders
        ]

    def _create_mutations(
        self, configurations_to_mutate: list[Configuration], amount: int
    ) -> list[Configuration]:
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
                            configuration_to_mutate
                        )
                    case x if 0.1 <= x < 0.2:
                        mutation = self._remove_convolutional_layer_mutation(
                            configuration_to_mutate
                        )
                    case x if 0.2 <= x < 0.3:
                        mutation = self._increase_filter_size_mutation(
                            configuration_to_mutate
                        )
                    case x if 0.3 <= x < 0.4:
                        mutation = self._decrease_filter_size_mutation(
                            configuration_to_mutate
                        )
                    case x if 0.4 <= x < 0.5:
                        mutation = self._increase_number_of_filters_mutation(
                            configuration_to_mutate
                        )
                    case x if 0.5 <= x < 0.6:
                        mutation = self._decrease_number_of_filters_mutation(
                            configuration_to_mutate
                        )
                    case x if 0.6 <= x < 0.7:
                        mutation = self._change_activation_function_mutation(
                            configuration_to_mutate
                        )
                    case x if 0.7 <= x < 0.8:
                        mutation = self._increase_sample_rate_mutation(
                            configuration_to_mutate
                        )
                    case x if 0.8 <= x < 0.9:
                        mutation = self._decrease_sample_rate_mutation(
                            configuration_to_mutate
                        )
                    case x if 0.9 <= x < 1:
                        mutation = self._change_preprocessing_mutation(
                            configuration_to_mutate
                        )
            mutations.append(mutation)

        return mutations

    # Generate a random new convolutional layer and add it to the end of the convolutional part of the model.

    def _new_convolutional_layer_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        mutation = copy.deepcopy(configuration)
        new_conv_layer = tuple(
            random.choice(x) for x in self.search_space.model_layer_search_space
        )
        assert type(mutation[1]) == list
        if len(mutation[1]) < self.max_num_layers:
            mutation[1].append(new_conv_layer)
        return mutation

    # Remove the last convolutional layer of the model
    def _remove_convolutional_layer_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1]) == list
        if len(mutation[1]) > 1:
            mutation[1].pop()
        return mutation

    # Increase the filter size of a random convolutional layer
    def _increase_filter_size_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Change filter size. Filter size in the current search space is in the second position of the search space tuple.
        current_filter_size = layer_to_modify[1]
        new_filter_size = None
        for seach_space_filter_size in self.search_space.model_layer_search_space[1]:
            if seach_space_filter_size > current_filter_size:
                new_filter_size = seach_space_filter_size
                break
        if new_filter_size == None:
            new_filter_size = max(self.search_space.model_layer_search_space[1])

        # Encode layer again
        layer_to_modify = (layer_to_modify[0], new_filter_size, layer_to_modify[2])

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = layer_to_modify

        return mutation

    def _decrease_filter_size_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Change filter size. Filter size in the current search space is in the second position of the search space tuple.
        current_filter_size = layer_to_modify[1]
        new_filter_size = None
        for seach_space_filter_size in reversed(
            self.search_space.model_layer_search_space[1]
        ):
            if seach_space_filter_size < current_filter_size:
                new_filter_size = seach_space_filter_size
                break
        if new_filter_size == None:
            new_filter_size = min(self.search_space.model_layer_search_space[1])

        # Encode layer again
        layer_to_modify = (layer_to_modify[0], new_filter_size, layer_to_modify[2])

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = layer_to_modify

        return mutation

    def _increase_number_of_filters_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Change filter amount. Amount of filters in the current search space is in the first position of the search space tuple.
        current_filter_amount = layer_to_modify[0]
        new_filter_amount = None
        for seach_space_filter_amount in self.search_space.model_layer_search_space[0]:
            if seach_space_filter_amount > current_filter_amount:
                new_filter_amount = seach_space_filter_amount
                break
        if new_filter_amount == None:
            new_filter_amount = max(self.search_space.model_layer_search_space[0])

        # Encode layer again
        layer_to_modify = (new_filter_amount, layer_to_modify[1], layer_to_modify[2])

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = layer_to_modify

        return mutation

    def _decrease_number_of_filters_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Change filter amount. Amount of filters in the current search space is in the first position of the search space tuple.
        current_filter_amount = layer_to_modify[0]
        new_filter_amount = None
        for seach_space_filter_amount in reversed(
            self.search_space.model_layer_search_space[0]
        ):
            if seach_space_filter_amount < current_filter_amount:
                new_filter_amount = seach_space_filter_amount
                break
        if new_filter_amount == None:
            new_filter_amount = min(self.search_space.model_layer_search_space[0])

        # Encode layer again
        layer_to_modify = (new_filter_amount, layer_to_modify[1], layer_to_modify[2])

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = layer_to_modify

        return mutation

    def _change_activation_function_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1] == list)
        random_conv_layer_number = self._random_conv_layer_number(mutation)
        layer_to_modify = mutation[1][random_conv_layer_number]

        # Change activation function. Activation function in the current search space is in the third position of the search space tuple.
        current_activation_function = layer_to_modify[2]
        new_activation_function = current_activation_function

        while current_activation_function == new_activation_function:
            new_activation_function = random.choice(
                self.search_space.model_layer_search_space[2]
            )

        # Encode layer again
        layer_to_modify = (
            layer_to_modify[0],
            layer_to_modify[1],
            new_activation_function,
        )

        # Add the layer to the configuration again
        mutation[1][random_conv_layer_number] = layer_to_modify

        return mutation

    def _increase_sample_rate_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        # Change sample rate. Sample rate in the current search space is in the first position of the search space tuple.
        current_sample_rate = configuration[0][0]
        new_sample_rate = None
        for seach_space_sample_rate in reversed(
            self.search_space.data_granularity_search_space[0]
        ):
            if seach_space_sample_rate > current_sample_rate:
                new_sample_rate = seach_space_sample_rate
                break
        if new_sample_rate == None:
            new_sample_rate = max(self.search_space.data_granularity_search_space[0])

        # Encode layer again
        new_data_granularity = (new_sample_rate, configuration[0][1])

        # Add the layer to the configuration again
        configuration = (new_data_granularity, configuration[1])

        return configuration

    def _decrease_sample_rate_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        # Change sample rate. Sample rate in the current search space is in the first position of the search space tuple.
        current_sample_rate = configuration[0][0]
        new_sample_rate = None
        for seach_space_sample_rate in self.search_space.data_granularity_search_space[
            0
        ]:
            if seach_space_sample_rate < current_sample_rate:
                new_sample_rate = seach_space_sample_rate
                break
        if new_sample_rate == None:
            new_sample_rate = min(self.search_space.data_granularity_search_space[0])

        # Encode layer again
        new_data_granularity = (new_sample_rate, configuration[0][1])

        # Add the layer to the configuration again
        configuration = (new_data_granularity, configuration[1])

        return configuration

    def _change_preprocessing_mutation(
        self, configuration: Configuration
    ) -> Configuration:
        # Change preprocessing. Preprocessing in the current search space is in the second position of the search space tuple.
        current_preprocessing = configuration[0][1]
        new_preprocessing = current_preprocessing
        if len(self.search_space.data_granularity_search_space[1]) == 1:
            return configuration

        while new_preprocessing == current_preprocessing:
            new_preprocessing = random.choice(
                self.search_space.data_granularity_search_space[1]
            )

        # Encode layer again
        new_data_granularity = (configuration[0][0], new_preprocessing)

        # Add the layer to the configuration again
        configuration = (new_data_granularity, configuration[1])

        return configuration

    def _random_conv_layer_number(self, configuration: Configuration) -> int:
        return random.randrange(0, len(configuration[1]))

    def _create_crossovers(
        self, configurations_to_crossover: list[Configuration], amount: int
    ) -> list[Configuration]:
        crossovers = []
        for i in range(amount):
            random_parents = random.choices(configurations_to_crossover, k=2)
            crossovers.append(self._crossover(*random_parents))

        return crossovers

    def _crossover(
        self, configuration1: Configuration, configuration2: Configuration
    ) -> Configuration:
        new_data = (
            random.choice((configuration1[0][0], configuration2[0][0])),
            random.choice((configuration1[0][1], configuration2[0][1])),
        )

        num_layers_model1 = len(configuration1[1])
        num_layers_model2 = len(configuration2[1])

        if num_layers_model1 == num_layers_model2:
            num_layers_new_model = num_layers_model1
            min_layers = num_layers_model1
        else:
            min_layers = min(num_layers_model1, num_layers_model2)
            max_layers = max(num_layers_model1, num_layers_model2)

            num_layers_new_model = random.randint(min_layers, max_layers)

        new_model = []
        for i in range(min_layers):
            layer = (
                random.choice((configuration1[1][i][0], configuration2[1][i][0])),
                random.choice((configuration1[1][i][1], configuration2[1][i][1])),
                random.choice((configuration1[1][i][2], configuration2[1][i][2])),
            )
            new_model.append(layer)

        if num_layers_model1 > num_layers_model2:
            for i in range(min_layers, num_layers_new_model):
                new_model.append(copy.deepcopy(configuration1[1][i]))
        elif num_layers_model2 > num_layers_model1:
            for i in range(min_layers, num_layers_new_model):
                new_model.append(copy.deepcopy(configuration2[1][i]))

        return (new_data, new_model)
