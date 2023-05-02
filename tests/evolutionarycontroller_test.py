# Standard Library Imports
import unittest
import unittest.mock
import copy
import random

# Local Imports
import evolutionarycontroller
import datamodel
import searchspace


class EvolutionaryontrollerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.search_space = searchspace.SearchSpace([[48000, 24000, 12000, 6000, 3000, 1500, 750, 325], [
            "spectrogram", "mel-spectrogram", "mfcc"]], [[2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]])
        self.evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 5, 5, 0.5, 0.2, 10000, 32)

    def test_init(self):
        self.evolutionary_controller.initialize_controller(
            trivial_initialization=True)
        self.assertEqual(self.evolutionary_controller.unevaluated_configurations.pop(
            0), ((24000, 'spectrogram'), [(4, 5, 'relu')]))
        self.assertEqual(
            self.evolutionary_controller.unevaluated_configurations.pop(0), ((325, 'spectrogram'), [(64, 3, 'relu')]))

    def test_generate_configuration(self):
        self.evolutionary_controller.initialize_controller(
            trivial_initialization=True)
        data_configuration, model_configuration = self.evolutionary_controller.generate_configuration()
        self.assertEqual(data_configuration,
                         (24000, 'spectrogram'))
        self.assertEqual(model_configuration, [(4, 5, 'relu')])

    def test_generate_configuration_no_population(self):
        self.evolutionary_controller.initialize_controller(
            trivial_initialization=True)
        self.evolutionary_controller.unevaluated_configurations = []

        data_configuration, model_configuration = self.evolutionary_controller.generate_configuration()
        self.assertEqual(data_configuration,
                         (6000, 'mel-spectrogram'))
        self.assertEqual(model_configuration, [(2, 5, 'relu')])

    def test_update_parameters(self):
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.accuracy = 0.91
        data_model.precision = 0.2
        data_model.recall = 0.5
        data_model.model_size = 64266

        self.evolutionary_controller.update_parameters(data_model)
        self.assertTrue(
            isinstance(
                self.evolutionary_controller.population[0][0], datamodel.DataModel))
        self.assertTrue(
            0 <= self.evolutionary_controller.population[0][1] <= 3)

    def test_evaluate_fitness(self):
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.accuracy = 0.4
        data_model.precision = 0.5
        data_model.recall = 0.6
        data_model.model_size = 56356
        fitness = evolutionarycontroller.EvolutionaryController._evaluate_fitness(
            data_model, 100000)
        self.assertAlmostEqual(fitness, 2.06917917)

    def test_generate_new_unevaluated_population(self):
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.data_configuration = (24000, "mfcc")
        data_model.model_configuration = [(2, 5, "relu"), (16, 5, "sigmoid")]
        self.evolutionary_controller.population = [(copy.deepcopy(data_model), random.uniform(0, 3))
                                                   for i in range(5)]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(
            self.evolutionary_controller.unevaluated_configurations, [])

        self.evolutionary_controller._generate_new_unevaluated_configurations()

        self.assertEqual(
            self.evolutionary_controller.population[0][0].data_configuration, (24000, 'mfcc'))
        self.assertEqual(
            self.evolutionary_controller.population[1][0].data_configuration, (24000, 'mfcc'))
        self.assertEqual(
            self.evolutionary_controller.population[0][0].model_configuration, [(2, 5, 'relu'), (16, 5, 'sigmoid')])
        self.assertEqual(
            self.evolutionary_controller.population[1][0].model_configuration, [(2, 5, 'relu'), (16, 5, 'sigmoid')])
        self.assertEqual(
            self.evolutionary_controller.unevaluated_configurations, [((48000, 'mfcc'), [(2, 5, 'relu'), (16, 5, 'sigmoid')]), ((24000, 'mfcc'), [(2, 5, 'relu')]), ((24000, 'mfcc'), [(2, 5, 'relu'), (16, 5, 'sigmoid')])])

    def test_generate_new_unevaluated_population_other_seed2(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 5, 5, 0.5, 0.2, 10000, 40)
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.data_configuration = (24000, "mfcc")
        data_model.model_configuration = [(2, 5, "relu"), (16, 5, "sigmoid")]
        evolutionary_controller.population = [(copy.deepcopy(data_model), random.uniform(0, 3))
                                              for i in range(5)]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [])

        evolutionary_controller._generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_controller.population[0][0].data_configuration, (24000, "mfcc"))
        self.assertEqual(
            evolutionary_controller.population[1][0].data_configuration, (24000, "mfcc"))
        self.assertEqual(
            evolutionary_controller.population[0][0].model_configuration, [(2, 5, "relu"), (16, 5, "sigmoid")])
        self.assertEqual(
            evolutionary_controller.population[1][0].model_configuration, [(2, 5, "relu"), (16, 5, "sigmoid")])
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [((24000, 'mfcc'), [(2, 5, 'sigmoid'), (16, 5, 'sigmoid')]), ((
                12000, 'mfcc'), [(2, 5, 'relu'), (16, 5, 'sigmoid')]), ((24000, 'mfcc'), [(2, 5, 'relu'), (16, 5, 'sigmoid')])])

    def test_generate_new_unevaluated_population_other_seed3(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 5, 5, 0.5, 0.2, 10000, 55)
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.data_configuration = (24000, "mfcc")
        data_model.model_configuration = [(2, 5, "relu"), (16, 5, "sigmoid")]
        evolutionary_controller.population = [(copy.deepcopy(data_model), random.uniform(0, 3))
                                              for i in range(5)]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [])

        evolutionary_controller._generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_controller.population[0][0].data_configuration, (24000, "mfcc"))
        self.assertEqual(
            evolutionary_controller.population[1][0].data_configuration, (24000, "mfcc"))
        self.assertEqual(
            evolutionary_controller.population[0][0].model_configuration, [(2, 5, "relu"), (16, 5, "sigmoid")])
        self.assertEqual(
            evolutionary_controller.population[1][0].model_configuration, [(2, 5, "relu"), (16, 5, "sigmoid")])
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [((48000, 'mfcc'), [(2, 5, 'relu'), (16, 5, 'sigmoid')]), ((24000, 'mfcc'), [(2, 5, 'relu'), (16, 3, 'sigmoid')]), ((24000, 'mfcc'), [(2, 5, 'relu'), (16, 5, 'sigmoid')])])

    def test_tournament_selection(self):
        random.seed(23)
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 20, 5, 0.5, 0.2, 10000, 32)
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        evolutionary_controller.population = [(copy.deepcopy(data_model), random.uniform(0, 3))
                                              for i in range(20)]
        winners = evolutionary_controller._tournament_selection()
        self.assertEqual(len(winners), 10)

    def test_tournament_selection_low_population(self):
        random.seed(23)
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        evolutionary_controller.population = [(copy.deepcopy(data_model), random.uniform(0, 3))
                                              for i in range(1)]
        winners = evolutionary_controller._tournament_selection()
        self.assertEqual(len(winners), 1)

    def test_tournament_selection_no_population(self):
        random.seed(23)
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)

        evolutionary_controller.population = []
        winners = evolutionary_controller._tournament_selection()
        self.assertEqual(len(winners), 0)

    def test_get_breeder_configurations(self):
        random.seed(23)
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_model = unittest.mock.MagicMock(spec=datamodel.DataModel)
        data_model.data_configuration = (12000, "mel-spectrogram")
        data_model.model_configuration = [
            (32, 5, "sigmoid"), (16, 3, "relu"), (8, 3, "relu")]
        breeders = [(copy.deepcopy(data_model), random.uniform(0, 3))
                    for i in range(10)]

        breeder_configurations = evolutionary_controller._get_breeder_configurations(
            breeders)  # type: ignore : This function would complain about being supplied with a mock object instead of a DataModel object.

        self.assertEqual(breeder_configurations.pop()[
                         0], (12000, 'mel-spectrogram'))
        self.assertEqual(breeder_configurations.pop()[1], [
                         (32, 5, 'sigmoid'), (16, 3, 'relu'), (8, 3, 'relu')])

    def test_new_convolutional_layer_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._new_convolutional_layer_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual(len(model_configuration) + 1,
                         len(mutated_model_configuration))

    def test_new_convolutional_layer_mutation_layer_already_max(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 3, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._new_convolutional_layer_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual(len(model_configuration),
                         len(mutated_model_configuration))

    def test_remove_convolutional_layer_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._remove_convolutional_layer_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual([(8, 3, "relu"), (128, 3, "relu")],
                         mutated_model_configuration)

    def test_remove_convolutional_layer_mutation_already_min_layers(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._remove_convolutional_layer_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual([(8, 3, "relu")], mutated_model_configuration)

    def test_increase_number_of_filters_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._increase_number_of_filters_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual([(16, 3, "relu"), (128, 3, "relu"),
                         (64, 5, "sigmoid")], mutated_model_configuration)

    def test_decrease_number_of_filters_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._decrease_number_of_filters_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual([(4, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")],
                         mutated_model_configuration)

    def test_increase_filter_size_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._increase_filter_size_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual([(8, 5, "relu"), (128, 3, "relu"),
                         (64, 3, "sigmoid")], mutated_model_configuration)

    # This is not chaning anything as the decreased filter size is already at the minimum. Maybe this should be an error.
    def test_decrease_filter_size_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._decrease_filter_size_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual([(8, 3, "relu"), (128, 3, "relu"),
                         (64, 5, "sigmoid")], mutated_model_configuration)

    def test_change_activation_function_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._change_activation_function_mutation(
            (data_configuration, model_configuration))

        self.assertEqual(data_configuration, mutated_data_configuration)
        self.assertEqual([(8, 3, "sigmoid"), (128, 3, "relu"),
                         (64, 3, "sigmoid")], mutated_model_configuration)

    def test_increase_sample_rate_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._increase_sample_rate_mutation(
            (data_configuration, model_configuration))

        self.assertEqual((48000, "spectrogram"), mutated_data_configuration)
        self.assertEqual(model_configuration, mutated_model_configuration)

    def test_decrease_sample_rate_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._decrease_sample_rate_mutation(
            (data_configuration, model_configuration))

        self.assertEqual((12000, "spectrogram"), mutated_data_configuration)
        self.assertEqual(model_configuration, mutated_model_configuration)

    def test_decrease_sample_rate_mutation_already_minimum(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (325, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._decrease_sample_rate_mutation(
            (data_configuration, model_configuration))

        self.assertEqual((325, "spectrogram"), mutated_data_configuration)
        self.assertEqual(model_configuration, mutated_model_configuration)

    def test_change_preprocessing_type_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 32)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._change_preprocessing_mutation(
            (data_configuration, model_configuration))

        self.assertEqual((24000, "mel-spectrogram"),
                         mutated_data_configuration)
        self.assertEqual(model_configuration, mutated_model_configuration)

    def test_change_preprocessing_type_mutation_other_seed(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 51)
        data_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]

        mutated_data_configuration, mutated_model_configuration = evolutionary_controller._change_preprocessing_mutation(
            (data_configuration, model_configuration))

        self.assertEqual((24000, "mfcc"), mutated_data_configuration)
        self.assertEqual(model_configuration, mutated_model_configuration)

    def test_crossover(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 70)

        data_configuration1 = (6000, "mfcc")
        model_configuration1 = [
            (8, 5, "relu"), (4, 3, "relu"), (64, 3, "sigmoid")]

        data_configuration2 = (12000, "mel-spectrogram")
        model_configuration2 = [
            (64, 5, "relu"), (64, 5, "relu"), (32, 3, "relu")]

        crossovered_data_configuration, crossovered_model_configuration = evolutionary_controller._crossover(
            (data_configuration1, model_configuration1), (data_configuration2, model_configuration2))

        self.assertEqual((6000, "mel-spectrogram"),
                         crossovered_data_configuration)
        self.assertEqual([(64, 5, "relu"), (64, 3, "relu"),
                         (64, 3, "relu")], crossovered_model_configuration)

    def test_crossover_different_length(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 2, 5, 0.5, 0.2, 10000, 14)

        data_configuration1 = (6000, "mfcc")
        model_configuration1 = [
            (8, 3, "sigmoid"), (4, 3, "sigmoid"), (64, 3, "sigmoid")]

        data_configuration2 = (750, "spectrogram")
        model_configuration2 = [
            (64, 5, "relu"), (64, 5, "relu"), (32, 3, "relu"), (64, 3, "sigmoid"), (128, 5, "sigmoid")]

        crossovered_data_configuration, crossovered_model_configuration = evolutionary_controller._crossover(
            (data_configuration1, model_configuration1), (data_configuration2, model_configuration2))

        self.assertEqual((6000, "mfcc"), crossovered_data_configuration)
        self.assertEqual([(64, 5, 'sigmoid'), (64, 5, 'relu'), (32, 3,
                         'sigmoid'), (64, 3, 'sigmoid')], crossovered_model_configuration)

    def test_create_crossovers(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            self.search_space, 10, 5, 0.5, 0.2, 10000, 2)
        evolutionary_controller.initialize_controller(
            trivial_initialization=False)

        deep_copy_unevaluated_configurations = copy.deepcopy(
            evolutionary_controller.unevaluated_configurations)

        crossovers = evolutionary_controller._create_crossovers(
            evolutionary_controller.unevaluated_configurations, 5)

        self.assertEqual(deep_copy_unevaluated_configurations,
                         evolutionary_controller.unevaluated_configurations)
        self.assertEqual(
            crossovers, [((1500, 'mel-spectrogram'), [(8, 3, 'sigmoid')]), ((6000, 'spectrogram'),
                         [(4, 5, 'sigmoid')]), ((1500, 'mfcc'), [(128, 3, 'sigmoid'), (8, 3, 'sigmoid'),
                                                                 (64, 3, 'sigmoid')]), ((1500, 'spectrogram'), [(2, 3, 'relu'), (128, 3, 'relu')]),
                         ((48000, 'mfcc'), [(2, 3, 'relu'), (128, 3, 'relu'), (2, 3, 'sigmoid'), (32, 3, 'relu')])])
