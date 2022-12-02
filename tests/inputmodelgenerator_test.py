import unittest

import inputmodelgenerator

import datasetloader
import randomcontroller
import searchspace
import constants
import inputmodel

import tensorflow as tf
import copy


class InputModelGeneratorTestCase(unittest.TestCase):

    def test_adding_pareto_optimal_model(self):
        input_model1 = inputmodel.InputModel()
        input_model2 = inputmodel.InputModel()
        input_model3 = inputmodel.InputModel()
        input_model1.accuracy = 0.60
        input_model1.precision = 0.56
        input_model1.recall = 0.82
        input_model2.accuracy = 0.61
        input_model2.precision = 0.4
        input_model2.recall = 0.99
        input_model3.accuracy = 0.59
        input_model3.precision = 0.57
        input_model3.recall = 0.81

        input_model_generator = inputmodelgenerator.InputModelGenerator(
            2, tf.keras.losses.SparseCategoricalCrossentropy(), controller=randomcontroller.RandomController(constants.SEARCH_SPACE, seed=20), dataset_loader=None)
        pareto_optimal_models = [input_model1, input_model2, input_model3]

        input_model4 = inputmodel.InputModel()
        input_model4.accuracy = 0.02
        input_model4.precision = 0.58
        input_model4.recall = 0.82
        input_model_generator.save_pareto_optimal_models(
            input_model4, pareto_optimal_models)
        self.assertEqual(pareto_optimal_models, [
            input_model1, input_model2, input_model3])

    def test_adding_non_pareto_optimal_model(self):
        input_model1 = inputmodel.InputModel()
        input_model2 = inputmodel.InputModel()
        input_model3 = inputmodel.InputModel()

        input_model1.accuracy = 0.60
        input_model1.precision = 0.56
        input_model1.recall = 0.82
        input_model2.accuracy = 0.61
        input_model2.precision = 0.4
        input_model2.recall = 0.99
        input_model3.accuracy = 0.59
        input_model3.precision = 0.57
        input_model3.recall = 0.81

        input_model_generator = inputmodelgenerator.InputModelGenerator(
            2, tf.keras.losses.SparseCategoricalCrossentropy(), controller=randomcontroller.RandomController(constants.SEARCH_SPACE, seed=20), dataset_loader=None)
        pareto_optimal_models = [input_model1, input_model2, input_model3]

        input_model4 = inputmodel.InputModel()
        input_model4.accuracy = 0.02
        input_model4.precision = 0.55
        input_model4.recall = 0.82
        input_model_generator.save_pareto_optimal_models(
            input_model4, pareto_optimal_models)
        self.assertEqual(pareto_optimal_models, [
            input_model1, input_model2, input_model3])

    # An integration test such as the one below takes a very long time. How should this be included?
    # def test_integration_random_controller(self):
    #     search_space = searchspace.SearchSpace(([16, 32, 64, 128], [
    #         5, 8], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000], ["spectrogram", "mel-spectrogram", "mfcc"]))
    #     random_controller = randomcontroller.RandomController(
    #         search_space, seed=20)
    #     input_model_generator = inputmodelgenerator.InputModelGenerator(
    #         2, tf.keras.losses.SparseCategoricalCrossentropy(), controller=random_controller)
    #     pareto_optimal_models = input_model_generator.run_input_nas(10)
    #     self.assertEqual(pareto_optimal_models, None)
