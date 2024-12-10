# A class that implements a search strategy based on evolutionary algorithms.

# Standard Library Imports
import random
import copy
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


class SuperNetEvolutionary(searchstrategy.SearchStrategy):
    # Generates an initial population. The "trivial" parameter is a boolean that decides whether the initial population is generated out of random one layer models (True) or general random models (False)
    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int,
        population_update_ratio: float,
        crossover_ratio: float,
        max_ram_consumption: int,
        max_flash_consumption: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(search_space, seed)
        random.seed(seed)
        self.currently_evaluating = None
        self.unevaluated_configurations = []
        self.population = []
        self.population_size = population_size
        self.crossover_ratio = crossover_ratio
        self.max_ram_consumption = max_ram_consumption
        self.max_flash_consumption = max_flash_consumption
        self.tournament_amount = max(
            1, round(population_size * population_update_ratio)
        )

    def initialize_search_strategy(self) -> None:
        self.unevaluated_configurations = [
            Configuration(
                data_configuration={
                    key: random.choice(value)
                    for key, value in self.search_space.data_search_space.items()
                },
                model_configuration={
                    "stage3depth": random.randint(0, 3),
                    "stage4depth": random.randint(0, 4),
                    "stage5depth": random.randint(0, 3),
                    "stage6depth": random.randint(0, 3),
                    "stage7depth": random.randint(0, 1),
                    "stage1width": random.uniform(0.1, 1),
                    "stage2width": random.uniform(0.1, 1),
                },
            )
            for i in range(self.population_size)
        ]

    # Fetches an element that has not yet been evaluated from the population
    def generate_configuration(self) -> Configuration:
        # When an entire population has been evaluated we generate a new population
        if not self.unevaluated_configurations:
            self._generate_new_unevaluated_configurations()

        return self.unevaluated_configurations.pop()

    # Updates the data_model with its measured performance.
    # Generates a new population if all of the current population has been evaluated.

    def update_parameters(self, data_model: DataModel) -> None:
        # Add performance of the currently evaluating data model to the population
        fitness = self._evaluate_fitness(
            data_model, self.max_ram_consumption, self.max_flash_consumption
        )
        self.population.append((data_model, fitness))

    @staticmethod
    def _evaluate_fitness(
        data_model: DataModel, max_ram_consumption: int, max_flash_consumption: int
    ) -> float:
        ram_violation = max(data_model.ram_consumption - max_ram_consumption, 0)
        flash_violation = max(data_model.flash_consumption - max_flash_consumption, 0)

        ram_score = 1 - (ram_violation / max_ram_consumption)
        flash_score = 1 - (flash_violation / max_flash_consumption)
        return (
            data_model.accuracy
            + data_model.precision
            + data_model.recall
            + ram_score
            + flash_score
        )

    def _generate_new_unevaluated_configurations(self) -> None:
        # If there is no current population to generate new unevaluated configurations from we need to generate a new initial unevaluated configuration
        if not self.population:  # Checks if the list is empty
            self.initialize_search_strategy()
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
        # Shuffle population to generate random tournaments
        random.shuffle(self.population)

        tournaments = np.array_split(self.population, self.tournament_amount)

        winners = []
        for tournament in tournaments:
            best_tournament_fitness = -99
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
            configuration_to_mutate = configurations_to_mutate[
                i % min(len(configurations_to_mutate), amount)
            ]
            mutation = copy.deepcopy(configuration_to_mutate)

            # Sometimes a mutation does nothing, so continue until a change is made.
            # A mutation may be doing nothing if it is randomly chosen to decrease a value that is already at its minimum
            while vars(mutation) == vars(configuration_to_mutate):
                random_mutation_number = random.random()
                match random_mutation_number:
                    # Case for changing the data granularity
                    case x if 0 <= x < 0.5:
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
                    case x if 0.5 <= x < 0.8:
                        number_of_block_to_mutate = random.randint(3, 7)

                        if number_of_block_to_mutate == 4:
                            possible_values = list(range(5))
                        elif number_of_block_to_mutate == 7:
                            possible_values = list(range(2))
                        else:
                            possible_values = list(range(4))

                        mutated_value = self._mutate_value(
                            configuration_to_mutate.model_configuration[
                                f"stage{number_of_block_to_mutate}depth"
                            ],
                            possible_values=possible_values,
                        )

                        mutation.model_configuration[
                            f"stage{number_of_block_to_mutate}depth"
                        ] = mutated_value
                    case x if 0.8 <= x < 1:
                        width_to_mutate = random.randint(1, 2)
                        possible_values = np.linspace(0.1, 1, 10)

                        mutated_value = self._mutate_value(
                            configuration_to_mutate.model_configuration[
                                f"stage{width_to_mutate}width"
                            ],
                            possible_values=possible_values,
                        )

                        mutation.model_configuration[f"stage{width_to_mutate}width"] = (
                            mutated_value
                        )

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

        new_model = {}
        for i in range(3, 8):
            new_model[f"stage{i}depth"] = random.choice(
                [
                    configuration1.model_configuration[f"stage{i}depth"],
                    configuration2.model_configuration[f"stage{i}depth"],
                ]
            )

        return Configuration(data_configuration=new_data, model_configuration=new_model)
