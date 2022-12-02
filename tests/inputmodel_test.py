import unittest
import unittest.mock

import inputmodel

import searchspace

import tensorflow as tf
import copy


class InputModelTestCase(unittest.TestCase):
    def test_input_model_constructor(self):
        search_space = searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [
            3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        # To keep these tests separate from the dataset we use a mock object as the dataset loader.
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(
            return_value=(None, None))
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=[[tf.random.uniform((60, 79, 60, 60))]])
        input_model = inputmodel.InputModel()

        input_model.initialize_input_model(input_configuration=3, model_configuration=[10, 5, 3], search_space=search_space, dataset_loader=dataset_loader, frame_size=2048,
                                           hop_length=512, num_mel_banks=80, num_mfccs=13, num_target_classes=2, model_optimizer=tf.keras.optimizers.Adam(),  model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), model_metrics=["accuracy"],  model_width_dense_layer=10)
        self.assertTrue(isinstance(
            input_model.model, tf.keras.Model))

    def test_better_configuration(self):
        input_model1 = inputmodel.InputModel()
        input_model1.accuracy = 0.98
        input_model1.precision = 0.7
        input_model1.recall = 0.99
        input_model2 = copy.deepcopy(input_model1)
        input_model2.accuracy = 0.9
        input_model2.precision = 0.69
        input_model2.recall = 0.89
        self.assertTrue(input_model1.better_input_model(input_model2))

    def test_not_better_configuration(self):
        input_model1 = inputmodel.InputModel()
        input_model1.accuracy = 0.60
        input_model1.precision = 0.56
        input_model1.recall = 0.82
        input_model2 = copy.deepcopy(input_model1)
        input_model2.accuracy = 0.02
        input_model2.precision = 0.55
        input_model2.recall = 0.82
        self.assertFalse(input_model2.better_input_model(input_model1))
