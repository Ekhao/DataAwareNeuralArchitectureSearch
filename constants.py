import tensorflow as tf

NUM_OUTPUT_CLASSES = 2
LOSS_FUNCTION = tf.keras.losses.CategoricalCrossentropy
# Number of filters, filter size and activation function
MODEL_LAYER_SEARCH_SPACE = ([8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"])
INPUT_SEARCH_SPACE = ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], [
                      "spectrogram", "mfe", "mfcc", "waveform"])  # Sample rate and preprocessing type - maybe also add bit width
PATH_TO_NORMAL_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/"
PATH_TO_ANOMALOUS_FILES = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/"
