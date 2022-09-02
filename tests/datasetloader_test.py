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
