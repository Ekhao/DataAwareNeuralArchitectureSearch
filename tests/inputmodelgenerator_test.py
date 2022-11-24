# Modules to test
import inputmodelgenerator

# Internal modules required for testing
import datasetloader
import searchspace
import constants
import inputmodel

# External modules required for testing
import tensorflow as tf
import copy


def test_adding_pareto_optimal_model():
    dataset_loader = datasetloader.DatasetLoader(
        constants.PATH_TO_NORMAL_FILES, constants.PATH_TO_ANOMALOUS_FILES, 90, 20, 1)
    search_space = searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
        3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
    input_model1 = inputmodel.InputModel(input_configuration=5, model_configuration=[16, 5, 8], search_space=search_space, dataset_loader=dataset_loader, frame_size=2048, hop_length=512, num_mel_banks=80,
                                         num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), model_metrics=["accuracy"],  model_width_dense_layer=10)
    input_model2 = copy.deepcopy(input_model1)
    input_model3 = copy.deepcopy(input_model1)
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
        2, tf.keras.losses.SparseCategoricalCrossentropy(), dataset_loader=dataset_loader)
    input_model_generator.pareto_optimal_models.extend(
        [input_model1, input_model2, input_model3])

    input_model4 = copy.deepcopy(input_model1)
    input_model4.accuracy = 0.02
    input_model4.precision = 0.58
    input_model4.recall = 0.82
    input_model_generator.save_pareto_optimal_models(input_model4)
    assert input_model_generator.pareto_optimal_models == [
        input_model1, input_model2, input_model3]


def test_adding_non_pareto_optimal_model():
    dataset_loader = datasetloader.DatasetLoader(
        constants.PATH_TO_NORMAL_FILES, constants.PATH_TO_ANOMALOUS_FILES, 90, 20, 1)
    search_space = searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
        3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
    input_model1 = inputmodel.InputModel(input_configuration=5, model_configuration=[16, 5, 8], search_space=search_space, dataset_loader=dataset_loader, frame_size=2048, hop_length=512, num_mel_banks=80,
                                         num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), model_metrics=["accuracy"],  model_width_dense_layer=10)
    input_model2 = copy.deepcopy(input_model1)
    input_model3 = copy.deepcopy(input_model1)

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
        2, tf.keras.losses.SparseCategoricalCrossentropy(), dataset_loader=dataset_loader)
    input_model_generator.pareto_optimal_models.extend(
        [input_model1, input_model2, input_model3])

    input_model4 = copy.deepcopy(input_model1)
    input_model4.accuracy = 0.02
    input_model4.precision = 0.55
    input_model4.recall = 0.82
    input_model_generator.save_pareto_optimal_models(input_model4)
    assert input_model_generator.pareto_optimal_models == [
        input_model1, input_model2, input_model3]
