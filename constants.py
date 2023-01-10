# Joblib Parameters
NUM_CORES_TO_USE = 1


# Search space parameters
# Data: Sample rate and preprocessing type
DATA_SEARCH_SPACE = [[2, 4, 8, 16, 32, 64, 128], [
    3, 5], ["relu", "sigmoid"]]
# Model Layer: Amount of filters, filter size, and activation function
MODEL_LAYER_SEARCH_SPACE = [[48000, 24000, 12000, 6000, 3000, 1500, 750, 375], [
    "spectrogram", "mel-spectrogram", "mfcc"]]

# Model Parameters
NUM_OUTPUT_CLASSES = 2
LOSS_FUNCTION = "sparse_categorical_crossentropy"
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
OPTIMIZER = "adam"
NUM_EPOCHS = 20
BATCH_SIZE = 32
MODEL_SIZE_APPROXIMATE_RANGE = 100000

# Evolutionary parameters
POPULATION_SIZE = 10
# The ratio of the population to be discarded and regnerated when the population is updated.
POPULATION_UPDATE_RATIO = 0.5
# The ratio of the updated population to be generated by crossover. The rest of the updated population is generated by mutations
CROSSOVER_RATIO = 0.2

# Do not modify these if you do not know what you are doing.
# The normal and anomalous files are both 10 seconds. Noise files are however longer and need to be cut to 10 seconds.
AUDIO_SECONDS_TO_LOAD = 10
