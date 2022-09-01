from joblib import Parallel, delayed
import librosa
import tensorflow as tf
import numpy as np

from constants import *


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
            n_jobs=-1)(delayed(self.librosa_load_without_sample_rate)(file, sr=sample_rate) for file in normal_files)
        anomalous_audio = Parallel(
            n_jobs=-1)(delayed(self.librosa_load_without_sample_rate)(file, sr=sample_rate) for file in anomalous_files)

        match preprocessing_type:
            case "waveform":
                # While we can easily load data as a waveform it can not be processed by a standard convolutional block that expects image like dimensions.
                raise NotImplementedError(
                    "Loading data as a waveform has not been implemented yet")
                # return (normal_audio, anomalous_audio)
            case "spectrogram":
                normal_spectrograms = tf.map_fn(
                    self.create_spectrogram, normal_audio)
                anomalous_spectrograms = tf.map_fn(
                    self.create_spectrogram, anomalous_audio)
                return (normal_spectrograms, anomalous_spectrograms)
            case "mel-spectrogram":
                normal_mel_spectrograms = tf.map_fn(
                    lambda x: self.create_mel_spectrogram(x, sample_rate), normal_audio)
                anomalous_mel_spectrograms = tf.map_fn(
                    lambda x: self.create_mel_spectrogram(x, sample_rate), anomalous_audio)
                return (normal_mel_spectrograms, anomalous_mel_spectrograms)
            case "mfcc":
                raise NotImplementedError(
                    "Loading data as a MFCC has not been implemented yet")

        # Things to vary: sample rate, mfcc-spectrogram-melspectrogram-rawwaveform, bit-width-representation. Neural network architecture
        # I think that it makes sense to vary this as a part of the NAS file when the architecture is constructed based on the genotype.

    def librosa_load_without_sample_rate(self, file, sr):
        audio, _ = librosa.load(file, sr=sr)
        return audio

    def create_spectrogram(self, audio_sample):
        # |stft|^2 is the power spectrogram, |stft| is the amplitude spectrogram
        # We can convert both into the log-amplitude spectrogram using librosa.power_to_db or librosa.amplitude_to_db respectivly
        # TODO: Should we do log representation of both amplitude and frequence?
        return librosa.power_to_db(np.abs(librosa.stft(audio_sample, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)) ** 2)

    def create_mel_spectrogram(self, audio_sample, sample_rate):
        # Unlike for the spectrogram, the mel spectrogram has a directly accessible function in librosa.
        return librosa.power_to_db(librosa.feature.melspectrogram(audio_sample, sample_rate, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=NUMBER_OF_MEL_FILTER_BANKS))


# Stupid_testing
# dataset_loader = DatasetLoader()
# loaded_dataset = dataset_loader.load_dataset(PATH_TO_NORMAL_FILES, PATH_TO_ANOMALOUS_FILES,
#                                             sample_rate = 48000, preprocessing_type = "waveform", num_normal_files = 100, num_anomalous_files = 20)
