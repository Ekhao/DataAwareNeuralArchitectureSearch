# Modules to test
import inputnas
import datasetloading


# Modules required to do testing
import tensorflow as tf
from constants import *
import math


def test_search_space_gen():
    search_space = inputnas.SearchSpace(
        ([3, 5, 7], ["relu"]), (["Spectrogram", "MFCC"], [48000, 24000]))
    assert search_space.input_search_space == {0: ("Spectrogram", 48000), 1: (
        "Spectrogram", 24000), 2: ("MFCC", 48000), 3: ("MFCC", 24000)} and search_space.model_layer_search_space == {
        0: (3, "relu"), 1: (5, "relu"), 2: (7, "relu")}


def test_input_model_gen():
    model_generator = inputnas.InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION)
    model = model_generator.create_input_model([6, 8, 2], (32, 32, 3))
    assert isinstance(model, tf.keras.Model)


def test_waveform_dataset_loading():
    dataset_loader = datasetloading.DatasetLoader()
    dataset = dataset_loader.load_dataset(PATH_TO_NORMAL_FILES,
                                          PATH_TO_ANOMALOUS_FILES, 48000, "waveform", 10, 2)
    assert math.isclose(dataset[0][0][0][0], -6.1035156e-05, rel_tol=1e-7) and math.isclose(
        dataset[0][0][0][1], -9.1552734e-05, rel_tol=1e-7) and math.isclose(dataset[0][0][0][2], -1.2207031e-04, rel_tol=1e-7)
