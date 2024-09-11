# A class that implements a search strategy based on evolutionary algorithms.

# Standard Library Imports
import random
import copy
import math
import numbers
from typing import Optional, Any

# Third Party Imports
import numpy as np

# Local Imports
import searchstrategy
from searchspace import SearchSpace
from datamodel import DataModel
from configuration import Configuration

# Type Aliases
Individual = tuple[DataModel, float]


class EvolutionarySearchStrategy(searchstrategy.SearchStrategy):
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

    def initialize_search_strategy(self, trivial_initialization: bool = True) -> None:
        # A paper (Kenneth O Stanley, Jeff Clune, Joel Lehman, and Risto Miikkulainen. 2019. Designing neural networks through neuroevolution. Nature Machine Intelligence 1, 1 (2019), 24–35.) claims that it is good to start from an initial trivial solution. Therefore the initial population created here only contains models with only one layer.
        self.trivial_initialization = trivial_initialization
        if trivial_initialization:
            self.unevaluated_configurations = [
                Configuration(
                    data_configuration={
                        key: random.choice(value)
                        for key, value in self.search_space.data_search_space.items()
                    },
                    # Generate a model configuration by randomly selecting a choice the first (and only) layer.
                    model_configuration=[
                        self._pick_random_model_layer() for _ in range(1)
                    ],
                )
                for i in range(self.population_size)
            ]
        # Another common way to generate an intial configuration for evolutionary algorithms is to generate random models from the search space.
        else:
            self.unevaluated_configurations = [
                Configuration(
                    data_configuration={
                        key: random.choice(value)
                        for key, value in self.search_space.data_search_space.items()
                    },
                    # Generate a model configuration by randomly selecting a choice for all layers
                    model_configuration=[
                        self._pick_random_model_layer()
                        for _ in range(random.randint(1, self.max_num_layers))
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

    def _pick_random_model_layer(self) -> dict:
        # Pick a random type of layer:
        layer_type = random.choice(list(self.search_space.model_search_space.keys()))

        # Create a dictionary to be added to the model configuration list
        layer = {}

        # Add the type of the layer to this dictionary
        layer["type"] = layer_type

        # Populate the rest of the layer dictionary with random choices from the possible choices for this layer
        for key, value in self.search_space.model_search_space[layer_type].items():
            layer[key] = random.choice(value)

        return layer

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
            self.initialize_search_strategy(self.trivial_initialization)
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
    # It should return the configurations that generated those.
    def _get_breeder_configurations(
        self, breeders: list[Individual]
    ) -> list[Configuration]:
        return [breeder[0].configuration for breeder in breeders]

    def _create_mutations(
        self, configurations_to_mutate: list[Configuration], amount: int
    ) -> list[Configuration]:
        # Generate a random number to choose which mutation to use:
        mutations = []
        for i in range(amount):
            configuration_to_mutate = configurations_to_mutate[i % amount]
            mutation = copy.deepcopy(configuration_to_mutate)

            # Sometimes a mutation does nothing, so continue until a change is made.
            # A mutation may be doing nothing if it is randomly chosen to decrease a value that is already at its minimum
            while vars(mutation) == vars(configuration_to_mutate):
                random_mutation_number = random.random()
                match random_mutation_number:
                    # Case for changing the data granularity
                    case x if 0 <= x < 0.4:
                        key_to_mutate = random.choice(
                            list(configuration_to_mutate.data_configuration.keys())
                        )
                        value_to_mutate = mutation.data_configuration[key_to_mutate]
                        mutated_value = self._mutate_value(
                            value_to_mutate,
                            possible_values=self.search_space.data_search_space[
                                key_to_mutate
                            ],
                        )

                        mutation.data_configuration[key_to_mutate] = mutated_value

                    # Case for changing a configuration of a model layer
                    case x if 0.4 <= x < 8:
                        number_of_layer_to_mutate = random.randint(
                            0, len(configuration_to_mutate.model_configuration) - 1
                        )

                        key_to_mutate = random.choice(
                            list(
                                configuration_to_mutate.model_configuration[
                                    number_of_layer_to_mutate
                                ].keys()
                            )
                        )

                        if key_to_mutate == "type":
                            # TODO: We currently do not support mutating the type of a layer to anothe
                            continue
                        else:
                            current_layer_type = (
                                configuration_to_mutate.model_configuration[
                                    number_of_layer_to_mutate
                                ]["type"]
                            )

                            mutated_value = self._mutate_value(
                                configuration_to_mutate.model_configuration[
                                    number_of_layer_to_mutate
                                ][key_to_mutate],
                                possible_values=self.search_space.model_search_space[
                                    current_layer_type
                                ][key_to_mutate],
                            )

                            mutation.model_configuration[number_of_layer_to_mutate][
                                key_to_mutate
                            ] = mutated_value
                    # Case for adding a layer to the model
                    case x if 0.8 <= x < 0.9:
                        mutation = self._new_convolutional_layer_mutation(mutation)  # type: ignore Again the type system gets confused
                    # Case for removing a layer from the model
                    case x if 0.9 <= x < 1:
                        mutation = self._remove_convolutional_layer_mutation(mutation)  # type: ignore Same as above

            mutations.append(mutation)

        return mutations

    # Check if a value is orderable or not. If orderable, apply an increase or decrease mutation. If not orderable, apply a random mutation.
    def _mutate_value(self, value_to_mutate: Any, possible_values: list[Any]) -> Any:
        if (
            getattr(value_to_mutate, "__lt__", None) is not None
            and getattr(value_to_mutate, "__gt__", None) is not None
        ):
            return self._mutate_orderable_value(value_to_mutate, possible_values)
        else:
            return self._mutate_unorderable_value(possible_values)

    # Mutate an orderable value by either increasing or decreasing it
    def _mutate_orderable_value(
        self, value_to_mutate: Any, possible_values: list[Any]
    ) -> Any:
        # Create a copy of possible_values to avoid unwanted side effects from calling this function.
        possible_values_copy = possible_values.copy()
        random_number = random.random()
        match random_number:
            # Increase the value
            case x if 0 <= x < 0.5:
                possible_values_copy.sort()
                for possible_value in possible_values_copy:
                    if possible_value > value_to_mutate:
                        return possible_value
                return value_to_mutate  # If there is no higher value in the list
            # Decrease the value
            case x if 0.5 <= x < 1:
                possible_values_copy.sort(reverse=True)
                for possible_value in possible_values_copy:
                    if possible_value < value_to_mutate:
                        return possible_value
                return value_to_mutate  # If there is no lower value in the list

    # Mutate an unorderable value by choosing a random value from the search space
    def _mutate_unorderable_value(self, possible_values: list[Any]) -> Any:
        return random.choice(possible_values)

    def _new_layer_mutation(self, configuration: Configuration) -> Configuration:
        mutation = copy.deepcopy(configuration)
        new_layer_type = random.choice(
            list(self.search_space.model_search_space.keys())
        )
        new_layer = {
            key: random.choice(value)
            for key, value in self.search_space.model_search_space[new_layer_type]
        }
        assert type(mutation.model_configuration) == list
        if len(mutation[1]) < self.max_num_layers:
            mutation[1].append(new_layer)
        return mutation

    # Remove the last convolutional layer of the model
    def _remove_layer_mutation(self, configuration: Configuration) -> Configuration:
        mutation = copy.deepcopy(configuration)
        assert type(mutation[1]) == list
        if len(mutation[1]) > 1:
            mutation[1].pop()
        return mutation

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
        new_data = {
            key: random.choice(
                [
                    configuration1.data_configuration[key],
                    configuration2.data_configuration[key],
                ]
            )
            for key in list(configuration1.data_configuration.keys())
        }

        num_layers_model1 = len(configuration1.model_configuration)
        num_layers_model2 = len(configuration2.model_configuration)

        if num_layers_model1 == num_layers_model2:
            num_layers_new_model = num_layers_model1
            min_layers = num_layers_model1
        else:
            min_layers = min(num_layers_model1, num_layers_model2)
            max_layers = max(num_layers_model1, num_layers_model2)

            num_layers_new_model = random.randint(min_layers, max_layers)

        new_model = []
        for i in range(min_layers):
            new_layer = random.choice(
                [
                    configuration1.model_configuration[i],
                    configuration2.model_configuration[i],
                ]
            )

            new_model.append(new_layer)

        if num_layers_model1 > num_layers_model2:
            for i in range(min_layers, num_layers_new_model):
                new_model.append(copy.deepcopy(configuration1.model_configuration[i]))
        elif num_layers_model2 > num_layers_model1:
            for i in range(min_layers, num_layers_new_model):
                new_model.append(copy.deepcopy(configuration2.model_configuration[i]))

        return Configuration(data_configuration=new_data, model_configuration=new_model)
