# Standard Library Imports
import unittest
import json  # Loaded to get the dataset path from configuration file.

# Local Imports
import datasetloader


class DatasetLoaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_file = open("config.json", "r")
        config = json.load(config_file)
        config = config["datanas-config"]
        dataset_config = config["dataset-config"]
        self.dataset_loader = datasetloader.DatasetLoader(
            dataset_config["path-normal-files"], dataset_config["path-anomalous-files"], dataset_config["path-noise-files"], "case1", 2, 2, 1, 1, 1, 1, 10)

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
