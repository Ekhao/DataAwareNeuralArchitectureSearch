# Modules to test
import inputmodelgenerator


# Modules required to do testing
import tensorflow as tf
import inputmodel
from constants import *


def test_input_model_gen():
    input_model_generator = inputmodelgenerator.InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION, number_of_normal_files=180, number_of_anomalous_files=40)
    input_model = input_model_generator.create_input_model(3, [10, 5, 3])
    assert isinstance(input_model.model, tf.keras.Model)


def test_adding_pareto_optimal_model():
    input_model_generator = inputmodelgenerator.InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION)
    input_model1 = inputmodel.InputModel(None, None)
    input_model1.accuracy = 0.60
    input_model1.precision = 0.56
    input_model1.recall = 0.82
    input_model2 = inputmodel.InputModel(None, None)
    input_model2.accuracy = 0.61
    input_model2.precision = 0.4
    input_model2.recall = 0.99
    input_model3 = inputmodel.InputModel(None, None)
    input_model3.accuracy = 0.59
    input_model3.precision = 0.57
    input_model3.recall = 0.81
    input_model_generator.pareto_optimal_models.extend(
        [input_model1, input_model2, input_model3])
    input_model4 = inputmodel.InputModel(None, None)
    input_model4.accuracy = 0.02
    input_model4.precision = 0.58
    input_model4.recall = 0.82
    input_model_generator.save_pareto_optimal_models(input_model4)
    assert input_model_generator.pareto_optimal_models == [
        input_model1, input_model2, input_model3]


def test_adding_non_pareto_optimal_model():
    input_model_generator = inputmodelgenerator.InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION)
    input_model1 = inputmodel.InputModel(None, None)
    input_model1.accuracy = 0.60
    input_model1.precision = 0.56
    input_model1.recall = 0.82
    input_model2 = inputmodel.InputModel(None, None)
    input_model2.accuracy = 0.61
    input_model2.precision = 0.4
    input_model2.recall = 0.99
    input_model3 = inputmodel.InputModel(None, None)
    input_model3.accuracy = 0.59
    input_model3.precision = 0.57
    input_model3.recall = 0.81
    input_model_generator.pareto_optimal_models.extend(
        [input_model1, input_model2, input_model3])
    input_model4 = inputmodel.InputModel(None, None)
    input_model4.accuracy = 0.02
    input_model4.precision = 0.55
    input_model4.recall = 0.82
    input_model_generator.save_pareto_optimal_models(input_model4)
    assert input_model_generator.pareto_optimal_models == [
        input_model1, input_model2, input_model3]
