# Standard Library Imports
import unittest
import unittest.mock
import copy
import random

# Local Imports
import search_strategies.evolutionarysearchstrategy as evolutionarysearchstrategy
import datamodel
import searchspace
from configuration import Configuration


class EvolutionarySearchStrategyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.search_space = searchspace.SearchSpace(
            data_search_space={
                "sample_rate": [48000, 24000, 12000, 6000, 3000, 1500, 750, 325],
                "audio_representation": ["spectrogram", "mel-spectrogram", "mfcc"],
            },
            model_search_space={
                "conv_layer": {
                    "filters": [2, 4, 8, 16, 32, 64, 128],
                    "kernel_size": [3, 5],
                    "activation": ["relu", "sigmoid"],
                }
            },
        )
        self.evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 5, 5, 0.5, 0.2, 256000, 32
            )
        )

    def test_init(self):
        self.evolutionary_search_strategy.initialize_search_strategy(
            trivial_initialization=True
        )
        self.assertEqual(
            vars(self.evolutionary_search_strategy.unevaluated_configurations.pop(0)),
            vars(
                Configuration(
                    data_configuration={
                        "sample_rate": 24000,
                        "audio_representation": "spectrogram",
                    },
                    model_configuration=[
                        {
                            "type": "conv_layer",
                            "filters": 8,
                            "kernel_size": 3,
                            "activation": "sigmoid",
                        }
                    ],
                )
            ),
        )
        self.assertEqual(
            vars(self.evolutionary_search_strategy.unevaluated_configurations.pop(0)),
            vars(
                Configuration(
                    data_configuration={
                        "sample_rate": 48000,
                        "audio_representation": "mfcc",
                    },
                    model_configuration=[
                        {
                            "type": "conv_layer",
                            "filters": 2,
                            "kernel_size": 5,
                            "activation": "sigmoid",
                        }
                    ],
                )
            ),
        )

    def test_generate_configuration(self):
        self.evolutionary_search_strategy.initialize_search_strategy(
            trivial_initialization=True
        )
        configuration = self.evolutionary_search_strategy.generate_configuration()
        self.assertEqual(
            configuration.data_configuration,
            {"sample_rate": 24000, "audio_representation": "spectrogram"},
        )
        self.assertEqual(
            configuration.model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                }
            ],
        )

    def test_generate_configuration_no_population(self):
        self.evolutionary_search_strategy.initialize_search_strategy(
            trivial_initialization=True
        )
        self.evolutionary_search_strategy.unevaluated_configurations = []

        configuration = self.evolutionary_search_strategy.generate_configuration()
        self.assertEqual(
            configuration.data_configuration,
            {"sample_rate": 12000, "audio_representation": "mel-spectrogram"},
        )
        self.assertEqual(
            configuration.model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 3,
                    "activation": "relu",
                }
            ],
        )

    def test_update_parameters(self):
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.accuracy = 0.91
        data_model.precision = 0.2
        data_model.recall = 0.5
        data_model.memory_consumption = 64266

        self.evolutionary_search_strategy.update_parameters(data_model)
        self.assertTrue(
            isinstance(
                self.evolutionary_search_strategy.population[0][0], datamodel.DataModel
            )
        )
        self.assertTrue(0 <= self.evolutionary_search_strategy.population[0][1] <= 3)

    def test_evaluate_fitness(self):
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.accuracy = 0.4
        data_model.precision = 0.5
        data_model.recall = 0.6
        data_model.memory_consumption = 156356
        fitness = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy._evaluate_fitness(
                data_model, 170000
            )
        )
        self.assertAlmostEqual(fitness, 2.5)

    def test_generate_new_unevaluated_population(self):
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.configuration = Configuration(
            data_configuration={
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        self.evolutionary_search_strategy.population = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(5)
        ]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(
            self.evolutionary_search_strategy.unevaluated_configurations, []
        )

        self.evolutionary_search_strategy._generate_new_unevaluated_configurations()

        self.assertEqual(
            self.evolutionary_search_strategy.population[0][
                0
            ].configuration.data_configuration,
            {
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
        )
        self.assertEqual(
            self.evolutionary_search_strategy.population[1][
                0
            ].configuration.data_configuration,
            {
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
        )
        self.assertEqual(
            self.evolutionary_search_strategy.population[0][
                0
            ].configuration.model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        self.assertEqual(
            self.evolutionary_search_strategy.population[1][
                0
            ].configuration.model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        self.assertEqual(
            vars(self.evolutionary_search_strategy.unevaluated_configurations[0]),
            vars(
                Configuration(
                    data_configuration={
                        "sample_rate": 24000,
                        "audio_representation": "mfcc",
                    },
                    model_configuration=[
                        {
                            "type": "conv_layer",
                            "filters": 2,
                            "kernel_size": 3,
                            "activation": "relu",
                        },
                        {
                            "type": "conv_layer",
                            "filters": 16,
                            "kernel_size": 5,
                            "activation": "sigmoid",
                        },
                    ],
                )
            ),
        )

    def test_generate_new_unevaluated_population_other_seed2(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 5, 5, 0.5, 0.2, 10000, 40
            )
        )
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.configuration = Configuration(
            data_configuration={
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 4,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        evolutionary_search_strategy.population = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(5)
        ]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(evolutionary_search_strategy.unevaluated_configurations, [])

        evolutionary_search_strategy._generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_search_strategy.population[0][
                0
            ].configuration.data_configuration,
            {
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
        )
        self.assertEqual(
            evolutionary_search_strategy.population[1][
                0
            ].configuration.data_configuration,
            {
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
        )
        self.assertEqual(
            evolutionary_search_strategy.population[0][
                0
            ].configuration.model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 4,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        self.assertEqual(
            evolutionary_search_strategy.population[1][
                0
            ].configuration.model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 4,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        self.assertEqual(
            vars(evolutionary_search_strategy.unevaluated_configurations[0]),
            vars(
                Configuration(
                    data_configuration={
                        "sample_rate": 24000,
                        "audio_representation": "mfcc",
                    },
                    model_configuration=[
                        {
                            "type": "conv_layer",
                            "filters": 8,
                            "kernel_size": 5,
                            "activation": "relu",
                        },
                        {
                            "type": "conv_layer",
                            "filters": 16,
                            "kernel_size": 5,
                            "activation": "sigmoid",
                        },
                    ],
                )
            ),
        )

    def test_generate_new_unevaluated_population_other_seed3(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 5, 5, 0.5, 0.2, 10000, 55
            )
        )
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.configuration = Configuration(
            data_configuration={
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        evolutionary_search_strategy.population = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(5)
        ]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(evolutionary_search_strategy.unevaluated_configurations, [])

        evolutionary_search_strategy._generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_search_strategy.population[0][
                0
            ].configuration.data_configuration,
            {
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
        )
        self.assertEqual(
            evolutionary_search_strategy.population[1][
                0
            ].configuration.data_configuration,
            {
                "sample_rate": 24000,
                "audio_representation": "mfcc",
            },
        )
        self.assertEqual(
            evolutionary_search_strategy.population[0][
                0
            ].configuration.model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        self.assertEqual(
            evolutionary_search_strategy.population[1][
                0
            ].configuration.model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )
        self.assertEqual(
            vars(evolutionary_search_strategy.unevaluated_configurations[0]),
            vars(
                Configuration(
                    data_configuration={
                        "sample_rate": 24000,
                        "audio_representation": "mel-spectrogram",
                    },
                    model_configuration=[
                        {
                            "type": "conv_layer",
                            "filters": 2,
                            "kernel_size": 5,
                            "activation": "relu",
                        },
                        {
                            "type": "conv_layer",
                            "filters": 16,
                            "kernel_size": 5,
                            "activation": "sigmoid",
                        },
                    ],
                )
            ),
        )

    def test_tournament_selection(self):
        random.seed(23)
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 20, 5, 0.5, 0.2, 10000, 32
            )
        )
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        evolutionary_search_strategy.population = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(20)
        ]
        winners = evolutionary_search_strategy._tournament_selection()
        self.assertEqual(len(winners), 10)

    def test_tournament_selection_low_population(self):
        random.seed(23)
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 2, 5, 0.5, 0.2, 10000, 32
            )
        )
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        evolutionary_search_strategy.population = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(1)
        ]
        winners = evolutionary_search_strategy._tournament_selection()
        self.assertEqual(len(winners), 1)

    def test_tournament_selection_no_population(self):
        random.seed(23)
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 2, 5, 0.5, 0.2, 10000, 32
            )
        )

        evolutionary_search_strategy.population = []
        winners = evolutionary_search_strategy._tournament_selection()
        self.assertEqual(len(winners), 0)

    def test_get_breeder_configurations(self):
        random.seed(23)
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 2, 5, 0.5, 0.2, 10000, 32
            )
        )
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.configuration = Configuration(
            data_configuration={
                "sample_rate": 12000,
                "audio_representation": "mel-spectrogram",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 32,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 3,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                },
            ],
        )

        breeders = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(10)
        ]

        breeder_configurations = evolutionary_search_strategy._get_breeder_configurations(
            breeders  # type: ignore : This function would complain about being supplied with a mock object instead of a DataModel object.
        )

        self.assertEqual(
            breeder_configurations.pop().data_configuration,
            {
                "sample_rate": 12000,
                "audio_representation": "mel-spectrogram",
            },
        )
        self.assertEqual(
            breeder_configurations.pop().model_configuration,
            [
                {
                    "type": "conv_layer",
                    "filters": 32,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 3,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "relu",
                },
            ],
        )

    def test_crossover(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 2, 5, 0.5, 0.2, 10000, 70
            )
        )
        configuration_1 = Configuration(
            data_configuration={
                "sample_rate": 6000,
                "audio_representation": "mfcc",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 8,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
            ],
        )

        configuration_2 = Configuration(
            data_configuration={
                "sample_rate": 12000,
                "audio_representation": "mel-spectrogram",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 32,
                    "kernel_size": 3,
                    "activation": "relu",
                },
            ],
        )

        crossover_configuration = evolutionary_search_strategy._crossover(
            configuration1=configuration_1, configuration2=configuration_2
        )

        self.assertEqual(
            {"sample_rate": 6000, "audio_representation": "mel-spectrogram"},
            crossover_configuration.data_configuration,
        )
        self.assertEqual(
            [
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
            ],
            crossover_configuration.model_configuration,
        )

    def test_crossover_different_length(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 2, 5, 0.5, 0.2, 10000, 14
            )
        )

        configuration_1 = Configuration(
            data_configuration={
                "sample_rate": 6000,
                "audio_representation": "mfcc",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 8,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
                {
                    "type": "conv_layer",
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
            ],
        )

        configuration_2 = Configuration(
            data_configuration={
                "sample_rate": 750,
                "audio_representation": "spectrogram",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 32,
                    "kernel_size": 3,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
                {
                    "type": "conv_layer",
                    "filters": 128,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )

        crossover_configuration = evolutionary_search_strategy._crossover(
            configuration1=configuration_1, configuration2=configuration_2
        )

        self.assertEqual(
            {
                "sample_rate": 6000,
                "audio_representation": "mfcc",
            },
            crossover_configuration.data_configuration,
        )
        self.assertEqual(
            [
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
                {
                    "type": "conv_layer",
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
            ],
            crossover_configuration.model_configuration,
        )

    def test_create_crossovers(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 10, 5, 0.5, 0.2, 10000, 2
            )
        )
        evolutionary_search_strategy.initialize_search_strategy(
            trivial_initialization=False
        )

        deep_copy_unevaluated_configurations = copy.deepcopy(
            evolutionary_search_strategy.unevaluated_configurations
        )

        crossovers = evolutionary_search_strategy._create_crossovers(
            evolutionary_search_strategy.unevaluated_configurations, 5
        )

        self.assertEqual(
            vars(deep_copy_unevaluated_configurations[0]),
            vars(evolutionary_search_strategy.unevaluated_configurations[0]),
        )

        self.assertEqual(
            vars(deep_copy_unevaluated_configurations[3]),
            vars(evolutionary_search_strategy.unevaluated_configurations[3]),
        )

        self.assertEqual(
            vars(crossovers[1]),
            vars(
                Configuration(
                    data_configuration={
                        "sample_rate": 48000,
                        "audio_representation": "mfcc",
                    },
                    model_configuration=[
                        {
                            "activation": "sigmoid",
                            "filters": 128,
                            "kernel_size": 3,
                            "type": "conv_layer",
                        },
                        {
                            "activation": "relu",
                            "filters": 8,
                            "kernel_size": 5,
                            "type": "conv_layer",
                        },
                        {
                            "activation": "relu",
                            "filters": 64,
                            "kernel_size": 5,
                            "type": "conv_layer",
                        },
                        {
                            "activation": "sigmoid",
                            "filters": 8,
                            "kernel_size": 5,
                            "type": "conv_layer",
                        },
                    ],
                )
            ),
        )
