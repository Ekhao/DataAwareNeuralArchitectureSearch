import tensorflow as tf

# Search space parameters
# Number of filters, filter size and activation function
MODEL_LAYER_SEARCH_SPACE = ([2, 4, 8, 16, 32, 64, 128], [
                            3, 5], ["relu", "sigmoid"])
INPUT_SEARCH_SPACE = ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], [
                      "spectrogram", "mel-spectrogram", "mfcc"])  # Sample rate and preprocessing type

# Model Parameters
NUM_OUTPUT_CLASSES = 2
LOSS_FUNCTION = tf.keras.losses.SparseCategoricalCrossentropy()

# Dataset parameters
PATH_TO_NORMAL_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/"
PATH_TO_ANOMALOUS_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/"
NUMBER_OF_NORMAL_FILES_TO_USE = 900
NUMBER_OF_ABNORMAL_FILES_TO_USE = 200

# Audio preprocessing parameters
FRAME_SIZE = 2048  # Traditional values
HOP_SIZE = 512  # Traditional values
NUMBER_OF_MEL_FILTER_BANKS = 80  # Typically between 40 and 128
NUMBER_OF_MFCCS = 13  # Traditional values

# Controller parameters
MAX_NUM_LAYERS = 5

# Evaluation parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
