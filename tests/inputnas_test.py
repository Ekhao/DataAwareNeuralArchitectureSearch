# Modules to test
import inputnas


# Modules required to do testing
import tensorflow as tf
from constants import *


def test_search_space_gen():
    assert 1+1 == 2
    search_space = inputnas.SearchSpace(
        ([3, 5, 7], ["relu"]), (["Spectrogram", "MFCC"], [48000, 24000]))
    assert search_space.input_search_space == {0: ("Spectrogram", 48000), 1: (
        "Spectrogram", 24000), 2: ("MFCC", 48000), 3: ("MFCC", 24000)} and search_space.model_layer_search_space == {
        0: (3, "relu"), 1: (5, "relu"), 2: (7, "relu")}


def test_input_model_gen():
    input_model_generator = inputnas.InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION, number_of_normal_files=180, number_of_anomalous_files=40)
    input_model = input_model_generator.create_input_model(3, [10, 5, 3])
    assert isinstance(input_model.model, tf.keras.Model)


def test_better_configuration():
    input_model1 = inputnas.InputModel(None, None)
    input_model1.accuracy = 0.98
    input_model1.precision = 0.7
    input_model1.recall = 0.99
    input_model2 = inputnas.InputModel(None, None)
    input_model2.accuracy = 0.9
    input_model2.precision = 0.69
    input_model2.recall = 0.89
    assert input_model1.better_input_model(input_model2)


def test_not_better_configuration():
    input_model1 = inputnas.InputModel(None, None)
    input_model1.accuracy = 0.60
    input_model1.precision = 0.56
    input_model1.recall = 0.82
    input_model2 = inputnas.InputModel(None, None)
    input_model2.accuracy = 0.02
    input_model2.precision = 0.55
    input_model2.recall = 0.82
    assert not input_model2.better_input_model(input_model1)


def test_adding_pareto_optimal_model():
    input_model_generator = inputnas.InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION)
    input_model1 = inputnas.InputModel(None, None)
    input_model1.accuracy = 0.60
    input_model1.precision = 0.56
    input_model1.recall = 0.82
    input_model2 = inputnas.InputModel(None, None)
    input_model2.accuracy = 0.61
    input_model2.precision = 0.4
    input_model2.recall = 0.99
    input_model3 = inputnas.InputModel(None, None)
    input_model3.accuracy = 0.59
    input_model3.precision = 0.57
    input_model3.recall = 0.81
    input_model_generator.pareto_optimal_models.extend(
        [input_model1, input_model2, input_model3])
    input_model4 = inputnas.InputModel(None, None)
    input_model4.accuracy = 0.02
    input_model4.precision = 0.58
    input_model4.recall = 0.82
    input_model_generator.save_pareto_optimal_models(input_model4)
    assert input_model_generator.pareto_optimal_models == [
        input_model1, input_model2, input_model3]


def test_adding_non_pareto_optimal_model():
    input_model_generator = inputnas.InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION)
    input_model1 = inputnas.InputModel(None, None)
    input_model1.accuracy = 0.60
    input_model1.precision = 0.56
    input_model1.recall = 0.82
    input_model2 = inputnas.InputModel(None, None)
    input_model2.accuracy = 0.61
    input_model2.precision = 0.4
    input_model2.recall = 0.99
    input_model3 = inputnas.InputModel(None, None)
    input_model3.accuracy = 0.59
    input_model3.precision = 0.57
    input_model3.recall = 0.81
    input_model_generator.pareto_optimal_models.extend(
        [input_model1, input_model2, input_model3])
    input_model4 = inputnas.InputModel(None, None)
    input_model4.accuracy = 0.02
    input_model4.precision = 0.55
    input_model4.recall = 0.82
    input_model_generator.save_pareto_optimal_models(input_model4)
    assert input_model_generator.pareto_optimal_models == [
        input_model1, input_model2, input_model3]
