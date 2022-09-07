# Module to test
import datasetloader

# Modules needed for testing
from constants import *


def test_spectrogram_loading():
    dataset_loader = datasetloader.DatasetLoader()
    spectrograms = dataset_loader.load_dataset(path_normal_files=PATH_TO_NORMAL_FILES, path_anomalous_files=PATH_TO_ANOMALOUS_FILES,
                                               sample_rate=48000, preprocessing_type="spectrogram", num_normal_files=1, num_anomalous_files=0)
    assert spectrograms[0][0].shape == (1025, 938, 1)


def test_mel_spectrogram_loading():
    dataset_loader = datasetloader.DatasetLoader()
    spectrograms = dataset_loader.load_dataset(path_normal_files=PATH_TO_NORMAL_FILES, path_anomalous_files=PATH_TO_ANOMALOUS_FILES,
                                               sample_rate=48000, preprocessing_type="mel-spectrogram", num_normal_files=1, num_anomalous_files=0)
    assert spectrograms[0][0].shape == (80, 938, 1)


def test_mfcc_loading():
    dataset_loader = datasetloader.DatasetLoader()
    spectrograms = dataset_loader.load_dataset(path_normal_files=PATH_TO_NORMAL_FILES, path_anomalous_files=PATH_TO_ANOMALOUS_FILES,
                                               sample_rate=48000, preprocessing_type="mfcc", num_normal_files=1, num_anomalous_files=0)
    assert spectrograms[0][0].shape == (39, 938, 1)


def test_supervised_dataset_generator():
    dataset_loader = datasetloader.DatasetLoader()
    normal_preprocessed, anomalous_preprocessed = dataset_loader.load_dataset(
        path_normal_files=PATH_TO_NORMAL_FILES, path_anomalous_files=PATH_TO_ANOMALOUS_FILES, sample_rate=48000, preprocessing_type="mel-spectrogram", num_normal_files=2, num_anomalous_files=2)
    X_train, X_test, y_train, y_test = dataset_loader.supervised_dataset(
        normal_preprocessed, anomalous_preprocessed, test_size=0.5)
    assert X_train.shape == (2, 80, 938, 1) and X_test.shape == (
        2, 80, 938, 1) and y_train.shape == (2,) and y_test.shape == (2,)
