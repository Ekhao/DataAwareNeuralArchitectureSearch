from joblib import Parallel, delayed
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import copy


class DatasetLoader:

    # We do the initial part of the dataset loading in the constructor to avoid loading from files continiously
    # There are 1800 normal files and 400 anomalous files (in channel 1)
    def __init__(self, path_normal_files, path_anomalous_files, num_normal_files, num_anomalous_files, channel) -> None:
        # Find all files in the requested channel directory
        normal_files = tf.io.gfile.glob(
            f"{path_normal_files}*ch{channel}*.wav")
        anomalous_files = tf.io.gfile.glob(
            f"{path_anomalous_files}*ch{channel}*.wav")

        # Cut the amount of processed files according to the program parameters
        normal_files = normal_files[:num_normal_files]
        anomalous_files = anomalous_files[:num_anomalous_files]

        # Get the sample rate of the audio files. The sample rate is assumed to be the same for every file.
        self.base_sr = librosa.get_samplerate(normal_files[0])

        # Load the audio files using librosa. Use multiple python processes in order to speed up loading.
        self.base_normal_audio = Parallel(
            n_jobs=-1)(delayed(self.librosa_load_without_sample_rate)(file, sr=self.base_sr) for file in normal_files)
        self.base_anomalous_audio = Parallel(
            n_jobs=-1)(delayed(self.librosa_load_without_sample_rate)(file, sr=self.base_sr) for file in anomalous_files)

    def supervised_dataset(self, normal_preprocessed, anomalous_preprocessed, test_size=0.2):
        normal_y = tf.zeros(len(normal_preprocessed))
        abnormal_y = tf.ones(len(anomalous_preprocessed))

        X = tf.concat([normal_preprocessed, anomalous_preprocessed], 0)
        y = tf.concat([normal_y, abnormal_y], 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=test_size, stratify=y)

        return X_train, X_test, y_train, y_test

    # This method is called to load the dataset according to a specific input configuration.
    # The base dataset (full granularity) should already have been loaded in the constructor of this class.
    def load_dataset(self, target_sr, preprocessing_type, frame_size, hop_length, num_mel_banks=None, num_mfccs=None) -> tuple:
        normal_audio = copy.deepcopy(self.base_normal_audio)
        anomalous_audio = copy.deepcopy(self.base_anomalous_audio)

        normal_audio = Parallel(
            n_jobs=-1)(delayed(librosa.resample)(audio, self.base_sr, target_sr) for audio in normal_audio)
        anomalous_audio = Parallel(n_jobs=-1)(delayed(librosa.resample)
                                              (audio, self.base_sr, target_sr) for audio in anomalous_audio)

        match preprocessing_type:
            case "waveform":
                # While we can easily load data as a waveform it can not be processed by a standard convolutional block that expects image like dimensions.
                raise NotImplementedError(
                    "Loading data as a waveform has not been implemented yet")
                return (normal_audio, anomalous_audio)
            case "spectrogram":
                normal_spectogram = Parallel(
                    n_jobs=-1)(delayed(self.create_spectrogram)(audio, frame_size, hop_length) for audio in normal_audio)
                anomalous_spectrograms = Parallel(
                    n_jobs=-1)(delayed(self.create_spectrogram)(audio, frame_size, hop_length) for audio in anomalous_audio)
                return (normal_spectogram, anomalous_spectrograms)
            case "mel-spectrogram":
                normal_mel_spectogram = Parallel(
                    n_jobs=-1)(delayed(self.create_mel_spectrogram)(audio, target_sr, frame_size, hop_length, num_mel_banks) for audio in normal_audio)
                anomalous_mel_spectrograms = Parallel(
                    n_jobs=-1)(delayed(self.create_mel_spectrogram)(audio, target_sr, frame_size, hop_length, num_mel_banks) for audio in anomalous_audio)
                return (normal_mel_spectogram, anomalous_mel_spectrograms)
            case "mfcc":
                normal_mfccs = Parallel(
                    n_jobs=-1)(delayed(self.create_mfcc)(audio, target_sr, frame_size, hop_length, num_mel_banks, num_mfccs) for audio in normal_audio)
                anomalous_mfccs = Parallel(
                    n_jobs=-1)(delayed(self.create_mfcc)(audio, target_sr, frame_size, hop_length, num_mel_banks, num_mfccs) for audio in anomalous_audio)
                return (normal_mfccs, anomalous_mfccs)
            case _:
                raise NotImplementedError(
                    "This dataloader only supports loading audio as spectrograms, mel-spectrograms and mfccs.")

    def librosa_load_without_sample_rate(self, file, sr):
        audio, _ = librosa.load(file, sr=sr)
        return audio

    def create_spectrogram(self, audio_sample, frame_size, hop_length):
        # |stft|^2 is the power spectrogram, |stft| is the amplitude spectrogram
        # We can convert both into the log-amplitude spectrogram using librosa.power_to_db or librosa.amplitude_to_db respectivly
        # It is unclear whether we should do a log representation of both amplitude and frequence in a standard spectrogram.
        # In this function we only do a log representation of amplitude.
        # We always return the spectrograms with an additional axis as this is expected by tensorflows convolutional layers (since images have a channel dimension)
        return librosa.power_to_db(np.abs(librosa.stft(audio_sample, n_fft=frame_size, hop_length=hop_length)) ** 2)[..., tf.newaxis]

    def create_mel_spectrogram(self, audio_sample, sample_rate, frame_size, hop_length, num_mel_banks):
        # Unlike for the spectrogram, the mel spectrogram has a directly accessible function in librosa.
        return librosa.power_to_db(librosa.feature.melspectrogram(y=audio_sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_length, n_mels=num_mel_banks))[..., tf.newaxis]

    def create_mfcc(self, audio_sample, sample_rate, frame_size, hop_length, num_mel_banks, num_mfccs):
        # Maybe also do first and second derivatives - is supposedly improving accuracy
        # Is often not used that much in deep learning, and is made to understand speech and music - not machine sounds.
        mfcc = librosa.feature.mfcc(
            y=audio_sample, sr=sample_rate, n_fft=frame_size, hop_length=hop_length, n_mels=num_mel_banks, n_mfcc=num_mfccs)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        return np.concatenate((mfcc, delta_mfcc, delta2_mfcc))[..., tf.newaxis]
