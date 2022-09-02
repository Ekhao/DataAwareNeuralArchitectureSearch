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
    model_generator = inputnas.InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION)
    _, model = model_generator.create_input_model(3, [10, 5, 3])
    assert isinstance(model, tf.keras.Model)
