import unittest
import unittest.mock

import datamodel

import searchspace
import datasetloader

import tensorflow as tf
import copy


class DataModelTestCase(unittest.TestCase):
    def test_data_model_constructor(self):
        search_space = searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
            3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        # To keep these tests separate from the dataset we use a mock object as the dataset loader.
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(
            return_value=([None, None], [None, None]))
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=[[tf.random.uniform((60, 79, 60, 60))]])
        data_model = datamodel.DataModel.from_data_configuration(data_configuration=3, model_configuration=[10, 5, 3],
                                                                 search_space=search_space, dataset_loader=dataset_loader, frame_size=2048, hop_length=512, num_mel_banks=80, num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), model_metrics=["accuracy"],  model_width_dense_layer=10, seed=52)
        self.assertTrue(isinstance(
            data_model.model, tf.keras.Model))

    def test_better_configuration(self):
        # No need to give any real values to the data model for this test.
        data_model1 = datamodel.DataModel(None, None, None, None, None, None)
        data_model1.accuracy = 0.98
        data_model1.precision = 0.7
        data_model1.recall = 0.99
        data_model1.model_size = 3954353
        data_model2 = copy.deepcopy(data_model1)
        data_model2.accuracy = 0.9
        data_model2.precision = 0.69
        data_model2.recall = 0.89
        data_model2.model_size = 4054353

        self.assertTrue(data_model1.better_data_model(data_model2))

    def test_not_better_configuration(self):
        # No need to give any real values to the data model for this test.
        data_model1 = datamodel.DataModel(None, None, None, None, None, None)
        data_model1.accuracy = 0.60
        data_model1.precision = 0.56
        data_model1.recall = 0.82
        data_model1.model_size = 3954353
        data_model2 = copy.deepcopy(data_model1)
        data_model2.accuracy = 0.02
        data_model2.precision = 0.55
        data_model2.recall = 0.82
        data_model2.model_size = 4054353
        self.assertFalse(data_model2.better_data_model(data_model1))

    def test_better_configuration_model_size(self):
        data_model1 = datamodel.DataModel(None, None, None, None, None, None)
        data_model1.accuracy = 0.9
        data_model1.precision = 0.8
        data_model1.recall = 0.8
        data_model1.model_size = 6954353
        data_model2 = copy.deepcopy(data_model1)
        data_model2.accuracy = 0.5
        data_model2.precision = 0.55
        data_model2.recall = 0.4
        data_model2.model_size = 4054353
        self.assertTrue(data_model2.better_data_model(data_model1))

    def test_evaluate_model_size(self):
        search_space = searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
            3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(
            return_value=([None, None], [None, None]))
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=[[tf.random.uniform((60, 79, 60, 60))]])
        data_model = datamodel.DataModel.from_data_configuration(data_configuration=3, model_configuration=[10, 5, 3],
                                                                 search_space=search_space, dataset_loader=dataset_loader, frame_size=2048, hop_length=512, num_mel_banks=80, num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), model_metrics=["accuracy"],  model_width_dense_layer=10, seed=20)

        model_size_without_training = data_model._evaluate_model_size()

        # The model size is not completely deterministic on seperate devices, so we apply an almost equal test.
        self.assertAlmostEqual(
            model_size_without_training, 16615324, places=-4)

    def test_evaluate_model_size2(self):
        search_space = searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
            3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(
            return_value=([None, None], [None, None]))
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=[[tf.random.uniform((60, 79, 60, 60))]])
        data_model = datamodel.DataModel.from_data_configuration(data_configuration=20, model_configuration=[1], search_space=search_space,
                                                                 dataset_loader=dataset_loader, frame_size=2048, hop_length=512, num_mel_banks=80, num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), model_metrics=["accuracy"],  model_width_dense_layer=10, seed=20)
        model_size_without_training = data_model._evaluate_model_size()

        self.assertAlmostEqual(
            model_size_without_training, 21444220, places=-4)
