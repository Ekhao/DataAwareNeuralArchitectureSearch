import tensorflow as tf

# Search space parameters
# Number of filters, filter size and activation function
MODEL_LAYER_SEARCH_SPACE = ([8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"])
INPUT_SEARCH_SPACE = ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], [
                      "spectrogram", "mfe", "mfcc", "waveform"])  # Sample rate and preprocessing type - maybe also add bit width

# Model Parameters
NUM_OUTPUT_CLASSES = 2
LOSS_FUNCTION = tf.keras.losses.CategoricalCrossentropy

# Dataset parameters
PATH_TO_NORMAL_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/"
PATH_TO_ANOMALOUS_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/"
NUMBER_OF_NORMAL_FILES_TO_USE = 900
NUMBER_OF_ABNORMAL_FILES_TO_USE = 200

# STFT parameters
FRAME_SIZE = 2048
HOP_SIZE = 512

# Spectrogram parameters
NUMBER_OF_MEL_FILTER_BANKS = 80
