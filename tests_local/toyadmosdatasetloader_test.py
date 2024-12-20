# Standard Library Imports
import unittest
import json  # Loaded to get the dataset path from configuration file.

# Local Imports
import dataset_loaders.toyconveyordatasetloader
import data


class ToyConveyorDatasetLoaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        config_file = open("config.json", "r")
        config = json.load(config_file)
        self.dataset_loader = (
            dataset_loaders.toyconveyordatasetloader.ToyConveyorDatasetLoader(
                dataset_options={
                    "channel": 1,
                    "case": "case1",
                    "sound_gain": 1,
                    "noise_gain": 1,
                    "frame_size": 2048,
                    "hop_length": 512,
                    "num_mel_filters": 80,
                    "num_mfccs": 13,
                    "audio_seconds_to_load": 10,
                    "file_path": "/dtu-compute/emjn/ToyConveyor",
                    "num_files": [45, 10],
                },
                num_cores_to_use=-1,
            )
        )

    def test_spectrogram_loading(self):
        spectrograms = self.dataset_loader.configure_dataset(
            sample_rate=48000,
            audio_representation="spectrogram",
            frame_size=2048,
            hop_length=512,
        )
        self.assertEqual(spectrograms[0][0].shape, (1025, 938, 1))

    def test_mel_spectrogram_loading(self):
        spectrograms = self.dataset_loader.configure_dataset(
            sample_rate=48000,
            audio_representation="mel_spectrogram",
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
        )
        self.assertEqual(spectrograms[0][0].shape, (80, 938, 1))

    def test_mfcc_loading(self):
        spectrograms = self.dataset_loader.configure_dataset(
            sample_rate=48000,
            audio_representation="mfcc",
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
            num_mfccs=13,
        )
        self.assertEqual(spectrograms[0][0].shape, (39, 938, 1))

    def test_supervised_dataset_generator(self):
        normal_preprocessed, anomalous_preprocessed = (
            self.dataset_loader.configure_dataset(
                sample_rate=48000,
                audio_representation="mel_spectrogram",
                frame_size=2048,
                hop_length=512,
                num_mel_filters=80,
            )
        )

        data = self.dataset_loader.supervised_dataset(
            (normal_preprocessed, anomalous_preprocessed), test_size=0.5
        )
        self.assertEqual(data.X_train.shape, (27, 80, 938, 1))
        self.assertEqual(data.X_test.shape, (28, 80, 938, 1))
        self.assertEqual(data.y_train.shape, (27,))
        self.assertEqual(data.y_test.shape, (28,))
