# Module to test
import datasetloader

# Internal modules needed for testing
# The "constants" module is only used for the path to test files. The rest of the constants should not be used in the test cases to not get failed test cases when changing the configuration.
import constants


def test_spectrogram_loading():
    dataset_loader = datasetloader.DatasetLoader(path_normal_files=constants.PATH_TO_NORMAL_FILES,
                                                 path_anomalous_files=constants.PATH_TO_ANOMALOUS_FILES, num_normal_files=1, num_anomalous_files=0, channel=1)
    spectrograms = dataset_loader.load_dataset(
        target_sr=48000, preprocessing_type="spectrogram")
    assert spectrograms[0][0].shape == (1025, 938, 1)


def test_mel_spectrogram_loading():
    dataset_loader = datasetloader.DatasetLoader(path_normal_files=constants.PATH_TO_NORMAL_FILES,
                                                 path_anomalous_files=constants.PATH_TO_ANOMALOUS_FILES, num_normal_files=1, num_anomalous_files=0, channel=1)
    spectrograms = dataset_loader.load_dataset(
        target_sr=48000, preprocessing_type="mel-spectrogram")
    assert spectrograms[0][0].shape == (80, 938, 1)


def test_mfcc_loading():
    dataset_loader = datasetloader.DatasetLoader(path_normal_files=constants.PATH_TO_NORMAL_FILES,
                                                 path_anomalous_files=constants.PATH_TO_ANOMALOUS_FILES, num_normal_files=1, num_anomalous_files=0, channel=1)
    spectrograms = dataset_loader.load_dataset(
        target_sr=48000, preprocessing_type="mfcc")
    assert spectrograms[0][0].shape == (39, 938, 1)


def test_supervised_dataset_generator():
    dataset_loader = datasetloader.DatasetLoader(path_normal_files=constants.PATH_TO_NORMAL_FILES,
                                                 path_anomalous_files=constants.PATH_TO_ANOMALOUS_FILES, num_normal_files=2, num_anomalous_files=2, channel=1)
    normal_preprocessed, anomalous_preprocessed = dataset_loader.load_dataset(
        target_sr=48000, preprocessing_type="mel-spectrogram")

    X_train, X_test, y_train, y_test = dataset_loader.supervised_dataset(
        normal_preprocessed, anomalous_preprocessed, test_size=0.5)
    assert X_train.shape == (2, 80, 938, 1) and X_test.shape == (
        2, 80, 938, 1) and y_train.shape == (2,) and y_test.shape == (2,)
