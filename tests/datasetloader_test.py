import unittest

import datasetloader

# The "constants" module is only used for the path to test files. The rest of the constants should not be used in the test cases to not get failed test cases when changing the configuration.
import constants


class DatasetLoaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_loader = datasetloader.DatasetLoader(constants.PATH_NORMAL_FILES, constants.PATH_ANOMALOUS_FILES,
                                                          constants.PATH_NOISE_FILES, "case1", 2, 2, 1, 1, 1, 1, 10)

    def test_spectrogram_loading(self):
        spectrograms = self.dataset_loader.load_dataset(
            target_sr=48000, preprocessing_type="spectrogram", frame_size=2048, hop_length=512)
        self.assertEqual(spectrograms[0][0].shape, (1025, 938, 1))

    def test_mel_spectrogram_loading(self):
        spectrograms = self.dataset_loader.load_dataset(
            target_sr=48000, preprocessing_type="mel-spectrogram", frame_size=2048, hop_length=512, num_mel_banks=80)
        self.assertEqual(spectrograms[0][0].shape, (80, 938, 1))

    def test_mfcc_loading(self):
        spectrograms = self.dataset_loader.load_dataset(
            target_sr=48000, preprocessing_type="mfcc", frame_size=2048, hop_length=512, num_mel_banks=80, num_mfccs=13)
        self.assertEqual(spectrograms[0][0].shape, (39, 938, 1))

    def test_supervised_dataset_generator(self):
        normal_preprocessed, anomalous_preprocessed = self.dataset_loader.load_dataset(
            target_sr=48000, preprocessing_type="mel-spectrogram", frame_size=2048, hop_length=512, num_mel_banks=80)

        X_train, X_test, y_train, y_test = self.dataset_loader.supervised_dataset(
            normal_preprocessed, anomalous_preprocessed, test_size=0.5)
        self.assertEqual(X_train.shape, (2, 80, 938, 1))
        self.assertEqual(X_test.shape, (
            2, 80, 938, 1))
        self.assertEqual(y_train.shape, (2,))
        self.assertEqual(y_test.shape, (2,))
