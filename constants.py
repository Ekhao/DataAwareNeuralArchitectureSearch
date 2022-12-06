import tensorflow as tf
import searchspace

# Search space parameters:
# Model Layer: Amount of filters, filter size, and activation function
# Input: Sample rate and preprocessing type
SEARCH_SPACE = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [
    3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 375], [
        "spectrogram", "mel-spectrogram", "mfcc"]))

# Model Parameters
NUM_OUTPUT_CLASSES = 2
LOSS_FUNCTION = tf.keras.losses.SparseCategoricalCrossentropy()
WIDTH_OF_DENSE_LAYER = 10

# Dataset parameters
PATH_TO_NORMAL_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/"
PATH_TO_ANOMALOUS_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/"
PATH_TO_NOISE_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/EnvironmentalNoise_CNT/"
CASE_NOISE_FILES = "case1"
NUMBER_OF_NORMAL_FILES_TO_USE = 900
NUMBER_OF_ANOMALOUS_FILES_TO_USE = 200
DATASET_CHANNEL_TO_USE = 1
SOUND_GAIN = 10**(0/20)
NOISE_GAIN = 10**(0/20)

# Audio preprocessing parameters
FRAME_SIZE = 2048  # Traditional values
HOP_LENGTH = 512  # Traditional values
NUMBER_OF_MEL_FILTER_BANKS = 80  # Typically between 40 and 128
NUMBER_OF_MFCCS = 13  # Traditional values

# Controller parameters
MAX_NUM_LAYERS = 5

# Evaluation parameters
OPTIMIZER = tf.keras.optimizers.Adam()
NUM_EPOCHS = 30
BATCH_SIZE = 32

# Evolutionary parameters
POPULATION_SIZE = 10
# The ratio of the population to be discarded and regnerated when the population is updated. Not used for Tournament Selection. See tournament size instead.
# POPULATION_UPDATE_RATIO = 0.5
# The ratio of the updated population to be generated by crossover. The rest of the updated population is generated by mutations
CROSSOVER_RATIO = 0.2
TOURNAMENT_AMOUNT = 5

# Do not modify these if you do not know what you are doing.
# The normal and anomalous files are both 10 seconds. Noise files are however longer and need to be cut to 10 seconds.
AUDIO_SECONDS_TO_LOAD = 10
