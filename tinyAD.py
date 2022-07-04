from joblib import Parallel, delayed
import time
import librosa
import tensorflow as tf

#There are 1800 normal files and 400 anomalous files (in channel 1)
NUMBER_OF_NORMAL_FILES = 1800
NUMBER_OF_ANOMALOUS_FILES = 100

normal_path = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/"
anomalous_path = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/"

normal_files = tf.io.gfile.glob(normal_path  + "*ch1*.wav")
anomalous_files = tf.io.gfile.glob(anomalous_path + "*ch1*.wav")

normal_files = normal_files[:NUMBER_OF_NORMAL_FILES]
anomalous_files = anomalous_files[:NUMBER_OF_ANOMALOUS_FILES]

start = time.perf_counter()
normal_audio = Parallel(n_jobs=-1)(delayed(librosa.load)(file) for file in normal_files)
stop = time.perf_counter()
print(stop-start)