import unittest
import unittest.mock

import evolutionarycontroller

import inputmodel
import searchspace

import copy
import random


class EvolutionaryontrollerTestCase(unittest.TestCase):
    def test_init(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        evolutionary_controller.initialize_controller(
            trivial_initialization=True)
        self.assertEqual(evolutionary_controller.unevaluated_configurations.pop(0), (
            2, [6]))
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations.pop(0), (4, [9]))

    def test_generate_configuration(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        evolutionary_controller.initialize_controller(
            trivial_initialization=True)
        input_configuration, model_configuration = evolutionary_controller.generate_configuration()
        self.assertEqual(input_configuration,
                         2)
        self.assertEqual(model_configuration, [6])

    def test_generate_configuration_no_population(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        evolutionary_controller.initialize_controller(
            trivial_initialization=True)
        evolutionary_controller.unevaluated_configurations = []

        input_configuration, model_configuration = evolutionary_controller.generate_configuration()
        self.assertEqual(input_configuration,
                         22)
        self.assertEqual(model_configuration, [7])

    def test_update_parameters(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_model = inputmodel.InputModel()
        input_model.accuracy = 0.91
        input_model.precision = 0.2
        input_model.recall = 0.5

        evolutionary_controller.update_parameters(input_model)
        self.assertTrue(
            type(evolutionary_controller.population[0][0]) is inputmodel.InputModel)
        self.assertTrue(0 <= evolutionary_controller.population[0][1] <= 3)

    def test_evaluate_fitness(self):
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        input_model.accuracy = 0.4
        input_model.precision = 0.5
        input_model.recall = 0.6
        fitness = evolutionarycontroller.EvolutionaryController._EvolutionaryController__evaluate_fitness(
            input_model)
        self.assertEqual(fitness, 1.5)

    def test_generate_new_unevaluated_population(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=5)
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        input_model.input_configuration = 8
        input_model.model_configuration = [20, 5, 13]
        evolutionary_controller.population = [(copy.deepcopy(input_model), random.uniform(0, 3))
                                              for i in range(5)]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [])

        evolutionary_controller._EvolutionaryController__generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_controller.population[0][0].input_configuration, 8)
        self.assertEqual(
            evolutionary_controller.population[1][0].input_configuration, 8)
        self.assertEqual(
            evolutionary_controller.population[0][0].model_configuration, [20, 5, 13])
        self.assertEqual(
            evolutionary_controller.population[1][0].model_configuration, [20, 5, 13])
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [(5, [20, 5, 13]), [8, [20, 5]], (8, [20, 5, 13])])

    def test_generate_new_unevaluated_population_other_seed2(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=40, population_size=5)
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        input_model.input_configuration = 8
        input_model.model_configuration = [20, 5, 13]
        evolutionary_controller.population = [(copy.deepcopy(input_model), random.uniform(0, 3))
                                              for i in range(5)]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [])

        evolutionary_controller._EvolutionaryController__generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_controller.population[0][0].input_configuration, 8)
        self.assertEqual(
            evolutionary_controller.population[1][0].input_configuration, 8)
        self.assertEqual(
            evolutionary_controller.population[0][0].model_configuration, [20, 5, 13])
        self.assertEqual(
            evolutionary_controller.population[1][0].model_configuration, [20, 5, 13])
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [[8, [21, 5, 13]], [8, [20, 5, 15]], (8, [20, 5, 13])])

    def test_generate_new_unevaluated_population_other_seed3(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=55, population_size=5)
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        input_model.input_configuration = 8
        input_model.model_configuration = [20, 5, 13]
        evolutionary_controller.population = [(copy.deepcopy(input_model), random.uniform(0, 3))
                                              for i in range(5)]

        # Before generating a new list of unevaluated configurations the unevaluated configurations is empty
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [])

        evolutionary_controller._EvolutionaryController__generate_new_unevaluated_configurations()

        self.assertEqual(
            evolutionary_controller.population[0][0].input_configuration, 8)
        self.assertEqual(
            evolutionary_controller.population[1][0].input_configuration, 8)
        self.assertEqual(
            evolutionary_controller.population[0][0].model_configuration, [20, 5, 13])
        self.assertEqual(
            evolutionary_controller.population[1][0].model_configuration, [20, 5, 13])
        self.assertEqual(
            evolutionary_controller.unevaluated_configurations, [(5, [20, 5, 13]), (7, [20, 5, 13]), (8, [20, 5, 13])])

    def test_tournament_selection(self):
        random.seed(23)
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=20)
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        evolutionary_controller.population = [(copy.deepcopy(input_model), random.uniform(0, 3))
                                              for i in range(20)]
        winners = evolutionary_controller._EvolutionaryController__tournament_selection()
        self.assertEqual(len(winners), 10)

    def test_tournament_selection_low_population(self):
        random.seed(23)
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        evolutionary_controller.population = [(copy.deepcopy(input_model), random.uniform(0, 3))
                                              for i in range(1)]
        winners = evolutionary_controller._EvolutionaryController__tournament_selection()
        self.assertEqual(len(winners), 1)

    def test_tournament_selection_no_population(self):
        random.seed(23)
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        evolutionary_controller.population = []
        winners = evolutionary_controller._EvolutionaryController__tournament_selection()
        self.assertEqual(len(winners), 0)

    def test_get_breeder_configurations(self):
        random.seed(23)
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        input_model.input_configuration = 6
        input_model.model_configuration = [10, 7, 14]
        breeders = [(copy.deepcopy(input_model), random.uniform(0, 3))
                    for i in range(10)]

        breeder_configurations = evolutionary_controller._EvolutionaryController__get_breeder_configurations(
            breeders)

        self.assertEqual(breeder_configurations.pop()[0], 6)
        self.assertEqual(breeder_configurations.pop()[1], [10, 7, 14])

    def test_new_convolutional_layer_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__new_convolutional_layer_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual(input_configuration, evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual(len(model_configuration) + 1, len(
            evolutionary_controller.search_space.model_decode(mutated_model_configuration)))

    def test_remove_convolutional_layer_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__remove_convolutional_layer_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual(input_configuration, evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual([(8, 3, "relu"), (128, 3, "relu")],
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_increase_number_of_filters_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__increase_number_of_filters_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual(input_configuration, evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual([(16, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")],
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_decrease_number_of_filters_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__decrease_number_of_filters_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual(input_configuration, evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual([(4, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")],
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_increase_filter_size_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__increase_filter_size_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual(input_configuration, evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual([(8, 5, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")],
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    # This is not chaning anything as the decreased filter size is already at the minimum. Maybe this should be an error.
    def test_decrease_filter_size_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__decrease_filter_size_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual(input_configuration, evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual([(8, 3, "relu"), (128, 3, "relu"), (64, 5, "sigmoid")],
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_change_activation_function_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32,  population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__change_activation_function_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual(input_configuration, evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual([(8, 3, "sigmoid"), (128, 3, "relu"), (64, 3, "sigmoid")],
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_increase_sample_rate_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__increase_sample_rate_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual((48000, "spectrogram"), evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual(model_configuration,
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_decrease_sample_rate_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__decrease_sample_rate_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual((12000, "spectrogram"), evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual(model_configuration,
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_decrease_sample_rate_mutation_already_minimum(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (325, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__decrease_sample_rate_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual((325, "spectrogram"), evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual(model_configuration,
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_change_preprocessing_type_mutation(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=32, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__change_preprocessing_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual((24000, "mel-spectrogram"), evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual(model_configuration,
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_change_preprocessing_type_mutation_other_seed(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=51, population_size=2)
        input_configuration = (24000, "spectrogram")
        model_configuration = [
            (8, 3, "relu"), (128, 3, "relu"), (64, 3, "sigmoid")]
        encoded_input_configuration = evolutionary_controller.search_space.input_encode(
            input_configuration)
        encoded_model_configuration = evolutionary_controller.search_space.model_encode(
            model_configuration)

        mutated_input_configuration, mutated_model_configuration = evolutionary_controller._EvolutionaryController__change_preprocessing_mutation(
            (encoded_input_configuration, encoded_model_configuration))

        self.assertEqual((24000, "mfcc"), evolutionary_controller.search_space.input_decode(
            mutated_input_configuration))
        self.assertEqual(model_configuration,
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_crossover(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=70, population_size=2)

        input_configuration1 = (6000, "mfcc")
        model_configuration1 = [
            (8, 5, "relu"), (4, 3, "relu"), (64, 3, "sigmoid")]
        encoded_input_configuration1 = evolutionary_controller.search_space.input_encode(
            input_configuration1)
        encoded_model_configuration1 = evolutionary_controller.search_space.model_encode(
            model_configuration1)

        input_configuration2 = (12000, "mel-spectrogram")
        model_configuration2 = [
            (64, 5, "relu"), (64, 5, "relu"), (32, 3, "relu")]
        encoded_input_configuration2 = evolutionary_controller.search_space.input_encode(
            input_configuration2)
        encoded_model_configuration2 = evolutionary_controller.search_space.model_encode(
            model_configuration2)

        crossovered_input_configuration, crossovered_model_configuration = evolutionary_controller._EvolutionaryController__crossover(
            (encoded_input_configuration1, encoded_model_configuration1), (encoded_input_configuration2, encoded_model_configuration2))

        self.assertEqual((6000, "mel-spectrogram"), evolutionary_controller.search_space.input_decode(
            crossovered_input_configuration))
        self.assertEqual([(64, 5, "relu"), (64, 3, "relu"), (64, 3, "relu")],
                         evolutionary_controller.search_space.model_decode(crossovered_model_configuration))

    def test_crossover_different_length(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=14, population_size=2)

        input_configuration1 = (6000, "mfcc")
        model_configuration1 = [
            (8, 3, "sigmoid"), (4, 3, "sigmoid"), (64, 3, "sigmoid")]
        encoded_input_configuration1 = evolutionary_controller.search_space.input_encode(
            input_configuration1)
        encoded_model_configuration1 = evolutionary_controller.search_space.model_encode(
            model_configuration1)

        input_configuration2 = (750, "spectrogram")
        model_configuration2 = [
            (64, 5, "relu"), (64, 5, "relu"), (32, 3, "relu"), (64, 3, "sigmoid"), (128, 5, "sigmoid")]
        encoded_input_configuration2 = evolutionary_controller.search_space.input_encode(
            input_configuration2)
        encoded_model_configuration2 = evolutionary_controller.search_space.model_encode(
            model_configuration2)

        crossovered_input_configuration, crossovered_model_configuration = evolutionary_controller._EvolutionaryController__crossover(
            (encoded_input_configuration1, encoded_model_configuration1), (encoded_input_configuration2, encoded_model_configuration2))

        self.assertEqual((6000, "mfcc"), evolutionary_controller.search_space.input_decode(
            crossovered_input_configuration))
        self.assertEqual([(64, 5, 'sigmoid'), (64, 5, 'relu'), (32, 3, 'sigmoid'), (64, 3, 'sigmoid')],
                         evolutionary_controller.search_space.model_decode(crossovered_model_configuration))

    def test_create_crossovers(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=search_space, seed=2, population_size=10)
        evolutionary_controller.initialize_controller(
            trivial_initialization=False)

        deep_copy_unevaluated_configurations = copy.deepcopy(
            evolutionary_controller.unevaluated_configurations)

        crossovers = evolutionary_controller._EvolutionaryController__create_crossovers(
            evolutionary_controller.unevaluated_configurations, 5)

        self.assertEqual(deep_copy_unevaluated_configurations,
                         evolutionary_controller.unevaluated_configurations)
        self.assertEqual(
            crossovers, [(22, [11, 17, 14, 16]), (16, [4, 16]), (12, [15, 20]), (10, [5, 7, 7, 0, 5]), (1, [11, 17, 14])])
