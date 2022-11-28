import unittest
import unittest.mock

import evolutionarycontroller

import inputmodel
import searchspace
import datasetloader
# Only for the path to test files. The rest of the constants should not be used in the test cases to not get failed test cases when changing the configuration.
import constants

import tensorflow as tf
import copy
import random


class EvolutionaryontrollerTestCase(unittest.TestCase):
    def test_init(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
        self.assertEqual(evolutionary_controller.unevaluated_input_model.get(), (
            2, [6]))
        self.assertEqual(
            evolutionary_controller.unevaluated_input_model.get(), (4, [9]))

    def test_generate_configuration(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram",   "mel-spectrogram",        "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
        input_configuration, model_configuration = evolutionary_controller.generate_configuration()
        self.assertEqual(input_configuration,
                         2)
        self.assertEqual(model_configuration, [6])

    def test_update_parameters(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
        input_configuration, model_configuration = evolutionary_controller.generate_configuration()
        input_model = inputmodel.InputModel(input_configuration=input_configuration, model_configuration=model_configuration, search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
                                            3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), dataset_loader=datasetloader.    DatasetLoader(constants.PATH_TO_NORMAL_FILES, constants.PATH_TO_ANOMALOUS_FILES, 90, 20, 1), frame_size=2048, hop_length=512, num_mel_banks=80,     num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),   model_metrics=["accuracy"],  model_width_dense_layer=10)
        input_model.evaluate_input_model(num_epochs=5, batch_size=32)
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

    def test_tournament_selection(self):
        random.seed(23)
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
        input_model = unittest.mock.MagicMock(spec="inputmodel.InputModel")
        evolutionary_controller.population = [(copy.deepcopy(input_model), random.randint(0, 3))
                                              for i in range(10)]
        winners = evolutionary_controller._EvolutionaryController__tournament_selection()
        self.assertEqual(len(winners), 5)

    def test_new_convolutional_layer_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
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
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
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
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
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
        self.assertEqual([(8, 3, "relu"), (128, 3, "relu"), (128, 5, "sigmoid")],
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))

    def test_decrease_number_of_filters_mutation(self):
        evolutionary_controller = evolutionarycontroller.EvolutionaryController(
            search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
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
        self.assertEqual([(8, 3, "relu"), (128, 3, "relu"), (32, 5, "sigmoid")],
                         evolutionary_controller.search_space.model_decode(mutated_model_configuration))
