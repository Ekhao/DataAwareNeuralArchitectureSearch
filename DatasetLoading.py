from joblib import Parallel, delayed
import time
import librosa
import tensorflow as tf

# There are 1800 normal files and 400 anomalous files (in channel 1)
NUMBER_OF_NORMAL_FILES = 1800
NUMBER_OF_ANOMALOUS_FILES = 100

# Input the path to the normal and anomalous data
normal_path = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/"
anomalous_path = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/"

# Find every channel one file in the listed directories
normal_files = tf.io.gfile.glob(normal_path  + "*ch1*.wav")
anomalous_files = tf.io.gfile.glob(anomalous_path + "*ch1*.wav")

# Cut the amount of processed files according to the program parameters
normal_files = normal_files[:NUMBER_OF_NORMAL_FILES]
anomalous_files = anomalous_files[:NUMBER_OF_ANOMALOUS_FILES]

# Get the sample rate of the audio files. The sample rate is assumed to be the same for every file.
sample_rate = librosa.get_samplerate(normal_files[0])

# Load the audio files using librosa. Use multiple python processes in order to speed up loading.
normal_audio = Parallel(n_jobs=-1)(delayed(librosa.load)(file) for file in normal_files)
anomalous_audio = Parallel(n_jobs=-1)(delayed(librosa.load)(file) for file in anomalous_files)

# Things to vary: sample rate, mfcc-spectrogram-melspectrogram-rawwaveform, bit-width-representation. Neural network architecture