# Standard Library Imports
import unittest
import unittest.mock
import copy
import random

# Local Imports
import search_strategies.evolutionarysearchstrategy as evolutionarysearchstrategy
import datamodel
import searchspace


class EvolutionarySearchStrategyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = searchspace.SearchSpace(
            [
                [48000, 24000, 12000, 6000, 3000, 1500, 750, 325],
                ["spectrogram", "mel-spectrogram", "mfcc"],
            ],
            [[2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]],
        )
        self.evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 5, 5, 0.5, 0.2, 10000, 32
            )
        )

    def test_init(self):
        self.evolutionary_search_strategy.initialize_search_strategy(
            trivial_initialization=True
        )
        self.assertEqual(
            self.evolutionary_search_strategy.unevaluated_configurations.pop(0),
            ((24000, "spectrogram"), [(4, 5, "relu")]),
        )
        self.assertEqual(
            self.evolutionary_search_strategy.unevaluated_configurations.pop(0),
            ((325, "spectrogram"), [(64, 3, "relu")]),
        )

    def test_generate_configuration(self):
        self.evolutionary_search_strategy.initialize_search_strategy(
            trivial_initialization=True
        )
        (
            data_configuration,
            model_configuration,
        ) = self.evolutionary_search_strategy.generate_configuration()
        self.assertEqual(data_configuration, (24000, "spectrogram"))
        self.assertEqual(model_configuration, [(4, 5, "relu")])

    def test_generate_configuration_no_population(self):
        self.evolutionary_search_strategy.initialize_search_strategy(
            trivial_initialization=True
        )
        self.evolutionary_search_strategy.unevaluated_configurations = []

        (
            data_configuration,
            model_configuration,
        ) = self.evolutionary_search_strategy.generate_configuration()
        self.assertEqual(data_configuration, (6000, "mel-spectrogram"))
        self.assertEqual(model_configuration, [(2, 5, "relu")])

    def test_update_parameters(self):
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.accuracy = 0.91
        data_model.precision = 0.2
        data_model.recall = 0.5
        data_model.model_size = 64266

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
        data_model.model_size = 56356
        fitness = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy._evaluate_fitness(
                data_model, 100000
            )
        )
        self.assertAlmostEqual(fitness, 2.06917917)

    def test_generate_new_unevaluated_population(self):
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.data_configuration = (24000, "mfcc")
        data_model.model_configuration = [(2, 5, "relu"), (16, 5, "sigmoid")]
        self.evolutionary_search_strategy.population = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(5)
        ]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(
            self.evolutionary_search_strategy.unevaluated_configurations, []
        )

        self.evolutionary_search_strategy._generate_new_unevaluated_configurations()

        self.assertEqual(
            self.evolutionary_search_strategy.population[0][0].data_configuration,
            (24000, "mfcc"),
        )
        self.assertEqual(
            self.evolutionary_search_strategy.population[1][0].data_configuration,
            (24000, "mfcc"),
        )
        self.assertEqual(
            self.evolutionary_search_strategy.population[0][0].model_configuration,
            [(2, 5, "relu"), (16, 5, "sigmoid")],
        )
        self.assertEqual(
            self.evolutionary_search_strategy.population[1][0].model_configuration,
            [(2, 5, "relu"), (16, 5, "sigmoid")],
        )
        self.assertEqual(
            self.evolutionary_search_strategy.unevaluated_configurations,
            [
                ((24000, "mfcc"), [(2, 3, "relu"), (16, 5, "sigmoid")]),
                ((24000, "mfcc"), [(2, 5, "relu"), (16, 3, "sigmoid")]),
                ((24000, "mfcc"), [(2, 5, "relu"), (16, 5, "sigmoid")]),
            ],
        )

    def test_generate_new_unevaluated_population_other_seed2(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 5, 5, 0.5, 0.2, 10000, 40
            )
        )
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.data_configuration = (24000, "mfcc")
        data_model.model_configuration = [(2, 5, "relu"), (16, 5, "sigmoid")]
        evolutionary_search_strategy.population = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(5)
        ]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(evolutionary_search_strategy.unevaluated_configurations, [])

        evolutionary_search_strategy._generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_search_strategy.population[0][0].data_configuration,
            (24000, "mfcc"),
        )
        self.assertEqual(
            evolutionary_search_strategy.population[1][0].data_configuration,
            (24000, "mfcc"),
        )
        self.assertEqual(
            evolutionary_search_strategy.population[0][0].model_configuration,
            [(2, 5, "relu"), (16, 5, "sigmoid")],
        )
        self.assertEqual(
            evolutionary_search_strategy.population[1][0].model_configuration,
            [(2, 5, "relu"), (16, 5, "sigmoid")],
        )
        self.assertEqual(
            evolutionary_search_strategy.unevaluated_configurations,
            [
                ((12000, "mfcc"), [(2, 5, "relu"), (16, 5, "sigmoid")]),
                ((24000, "mel-spectrogram"), [(2, 5, "relu"), (16, 5, "sigmoid")]),
                ((24000, "mfcc"), [(2, 5, "relu"), (16, 5, "sigmoid")]),
            ],
        )

    def test_generate_new_unevaluated_population_other_seed3(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 5, 5, 0.5, 0.2, 10000, 55
            )
        )
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.data_configuration = (24000, "mfcc")
        data_model.model_configuration = [(2, 5, "relu"), (16, 5, "sigmoid")]
        evolutionary_search_strategy.population = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(5)
        ]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(evolutionary_search_strategy.unevaluated_configurations, [])

        evolutionary_search_strategy._generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_search_strategy.population[0][0].data_configuration,
            (24000, "mfcc"),
        )
        self.assertEqual(
            evolutionary_search_strategy.population[1][0].data_configuration,
            (24000, "mfcc"),
        )
        self.assertEqual(
            evolutionary_search_strategy.population[0][0].model_configuration,
            [(2, 5, "relu"), (16, 5, "sigmoid")],
        )
        self.assertEqual(
            evolutionary_search_strategy.population[1][0].model_configuration,
            [(2, 5, "relu"), (16, 5, "sigmoid")],
        )
        self.assertEqual(
            evolutionary_search_strategy.unevaluated_configurations,
            [
                ((24000, "mfcc"), [(2, 5, "relu"), (32, 5, "sigmoid")]),
                ((24000, "mfcc"), [(2, 5, "relu"), (16, 5, "relu")]),
                ((24000, "mfcc"), [(2, 5, "relu"), (16, 5, "sigmoid")]),
            ],
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
        data_model.data_configuration = (12000, "mel-spectrogram")
        data_model.model_configuration = [
            (32, 5, "sigmoid"),
            (16, 3, "relu"),
            (8, 3, "relu"),
        ]
        breeders = [
            (copy.deepcopy(data_model), random.uniform(0, 3)) for i in range(10)
        ]

        breeder_configurations = evolutionary_search_strategy._get_breeder_configurations(
            breeders  # type: ignore : This function would complain about being supplied with a mock object instead of a DataModel object.
        )

        self.assertEqual(breeder_configurations.pop()[0], (12000, "mel-spectrogram"))
        self.assertEqual(
            breeder_configurations.pop()[1],
            [(32, 5, "sigmoid"), (16, 3, "relu"), (8, 3, "relu")],
        )

    def test_crossover(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 2, 5, 0.5, 0.2, 10000, 70
            )
        )

        data_configuration1 = (6000, "mfcc")
        model_configuration1 = [(8, 5, "relu"), (4, 3, "relu"), (64, 3, "sigmoid")]

        data_configuration2 = (12000, "mel-spectrogram")
        model_configuration2 = [(64, 5, "relu"), (64, 5, "relu"), (32, 3, "relu")]

        (
            crossovered_data_configuration,
            crossovered_model_configuration,
        ) = evolutionary_search_strategy._crossover(
            (data_configuration1, model_configuration1),
            (data_configuration2, model_configuration2),
        )

        self.assertEqual((6000, "mel-spectrogram"), crossovered_data_configuration)
        self.assertEqual(
            [(64, 5, "relu"), (64, 3, "relu"), (64, 3, "relu")],
            crossovered_model_configuration,
        )

    def test_crossover_different_length(self):
        evolutionary_search_strategy = (
            evolutionarysearchstrategy.EvolutionarySearchStrategy(
                self.search_space, 2, 5, 0.5, 0.2, 10000, 14
            )
        )

        data_configuration1 = (6000, "mfcc")
        model_configuration1 = [
            (8, 3, "sigmoid"),
            (4, 3, "sigmoid"),
            (64, 3, "sigmoid"),
        ]

        data_configuration2 = (750, "spectrogram")
        model_configuration2 = [
            (64, 5, "relu"),
            (64, 5, "relu"),
            (32, 3, "relu"),
            (64, 3, "sigmoid"),
            (128, 5, "sigmoid"),
        ]

        (
            crossovered_data_configuration,
            crossovered_model_configuration,
        ) = evolutionary_search_strategy._crossover(
            (data_configuration1, model_configuration1),
            (data_configuration2, model_configuration2),
        )

        self.assertEqual((6000, "mfcc"), crossovered_data_configuration)
        self.assertEqual(
            [
                (64, 5, "sigmoid"),
                (64, 5, "relu"),
                (32, 3, "sigmoid"),
                (64, 3, "sigmoid"),
            ],
            crossovered_model_configuration,
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
            deep_copy_unevaluated_configurations,
            evolutionary_search_strategy.unevaluated_configurations,
        )
        self.assertEqual(
            crossovers,
            [
                ((1500, "mel-spectrogram"), [(8, 3, "sigmoid")]),
                ((6000, "spectrogram"), [(4, 5, "sigmoid")]),
                (
                    (1500, "mfcc"),
                    [(128, 3, "sigmoid"), (8, 3, "sigmoid"), (64, 3, "sigmoid")],
                ),
                ((1500, "spectrogram"), [(2, 3, "relu"), (128, 3, "relu")]),
                (
                    (48000, "mfcc"),
                    [
                        (2, 3, "relu"),
                        (128, 3, "relu"),
                        (2, 3, "sigmoid"),
                        (32, 3, "relu"),
                    ],
                ),
            ],
        )
