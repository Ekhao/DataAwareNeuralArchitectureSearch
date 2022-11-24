import evolutionarycontroller

import inputmodel
import searchspace
import datasetloader
# Only for the path to test files. The rest of the constants should not be used in the test cases to not get failed test cases when changing the configuration.
import constants

import tensorflow as tf


def test_init():
    evolutionary_controller = evolutionarycontroller.EvolutionaryController(
        search_space=searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
    assert evolutionary_controller.unevaluated_population.get() == (
        2, [6]) and evolutionary_controller.unevaluated_population.get() == (4, [9])


def test_generate_configuration():
    evolutionary_controller = evolutionarycontroller.EvolutionaryController(
        search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
    input_configuration, model_configuration = evolutionary_controller.generate_configuration()
    assert input_configuration == 2 and model_configuration == [6]


def test_update_parameters():
    evolutionary_controller = evolutionarycontroller.EvolutionaryController(
        search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
    input_configuration, model_configuration = evolutionary_controller.generate_configuration()
    input_model = inputmodel.InputModel(input_configuration=input_configuration, model_configuration=model_configuration, search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
                                        3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), dataset_loader=datasetloader.DatasetLoader(constants.PATH_TO_NORMAL_FILES, constants.PATH_TO_ANOMALOUS_FILES, 90, 20, 1), frame_size=2048, hop_length=512, num_mel_banks=80, num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), model_metrics=["accuracy"],  model_width_dense_layer=10)
    input_model.evaluate_input_model(num_epochs=5, batch_size=32)
    evolutionary_controller.update_parameters(input_model)
    assert type(
        evolutionary_controller.population[0][0]) is inputmodel.InputModel and 0 <= evolutionary_controller.population[0][1] <= 3


def test_evaluate_fitness():
    evolutionary_controller = evolutionarycontroller.EvolutionaryController(
        search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
    input_model = inputmodel.InputModel(input_configuration=5, model_configuration=[9, 3, 5], search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
                                        3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), dataset_loader=datasetloader.DatasetLoader(constants.PATH_TO_NORMAL_FILES, constants.PATH_TO_ANOMALOUS_FILES, 90, 20, 1), frame_size=2048, hop_length=512, num_mel_banks=80, num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), model_metrics=["accuracy"],  model_width_dense_layer=10)
    input_model.accuracy = 0.4
    input_model.precision = 0.5
    input_model.recall = 0.6
    fitness = evolutionary_controller._EvolutionaryController__evaluate_fitness(
        input_model)
    assert fitness == 1.5
