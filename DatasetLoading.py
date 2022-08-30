from joblib import Parallel, delayed
import time
import librosa
import tensorflow as tf


class DatasetLoader:
    def __init__(self) -> None:
        pass
    # There are 1800 normal files and 400 anomalous files (in channel 1)

    def load_dataset(self, path_normal_files, path_anomalous_files, sample_rate, preprocessing_type,
                     num_normal_files=1800, num_anomalous_files=400, channel=1) -> tuple:
        # Find every channel one file in the listed directories
        normal_files = tf.io.gfile.glob(
            f"{path_normal_files}*ch{channel}*.wav")
        anomalous_files = tf.io.gfile.glob(
            f"{path_anomalous_files}*ch{channel}*.wav")

        # Cut the amount of processed files according to the program parameters
        normal_files = normal_files[:num_normal_files]
        anomalous_files = anomalous_files[:num_anomalous_files]

        # Get the sample rate of the audio files. The sample rate is assumed to be the same for every file.
        sample_rate = librosa.get_samplerate(normal_files[0])

        # Load the audio files using librosa. Use multiple python processes in order to speed up loading.
        normal_audio = Parallel(
            n_jobs=-1)(delayed(librosa.load)(file, sr=sample_rate) for file in normal_files)
        anomalous_audio = Parallel(
            n_jobs=-1)(delayed(librosa.load)(file, sr=sample_rate) for file in anomalous_files)

        match preprocessing_type:
            case "waveform":
                return (normal_audio, anomalous_audio)
            case "spectrogram":
                raise NotImplementedError(
                    "Loading data as a spectrogram has not been implemented yet")
            case "mel-spectrogram":
                raise NotImplementedError(
                    "Loading data as a mel-spectrogram has not been implemented yet")
            case "mfcc":
                raise NotImplementedError(
                    "Loading data as a MFCC has not been implemented yet")

        # Things to vary: sample rate, mfcc-spectrogram-melspectrogram-rawwaveform, bit-width-representation. Neural network architecture
        # I think that it makes sense to vary this as a part of the NAS file when the architecture is constructed based on the genotype.


# Stupid_testing
dataset_loader = DatasetLoader()
loaded_dataset = dataset_loader.load_dataset(path_normal_files="/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/",
                                             path_anomalous_files="/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/",
                                             sample_rate=48000, preprocessing_type="waveform", num_normal_files=100, num_anomalous_files=20)
print(loaded_dataset)
