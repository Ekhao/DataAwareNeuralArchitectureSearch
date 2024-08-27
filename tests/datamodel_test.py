# Standard Library Imports
import unittest
import unittest.mock
import copy

# Third Party Imports
import tensorflow as tf

# Local Imports
import data
import datamodel
import searchspace


class DataModelTestCase(unittest.TestCase):
    def test_data_model_constructor(self):
        search_space = searchspace.SearchSpace(
            [
                [48000, 24000, 12000, 6000, 3000, 1500, 750, 325],
                ["spectrogram", "mel-spectrogram", "mfcc"],
            ],
            [[2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]],
        )
        # To keep these tests separate from the dataset we use a mock object as the dataset loader.
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(
            return_value=([None, None], [None, None])
        )
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=data.Data(
                X_train=tf.random.uniform((500, 79, 60, 60)),
                X_test=tf.random.uniform((500, 79, 60, 60)),
                y_train=tf.random.uniform((500,)),
                y_test=tf.random.uniform((500,)),
            )
        )
        data_model = datamodel.DataModel.from_data_configuration(
            data_configuration=(12000, "spectrogram"),
            model_configuration=[(16, 3, "relu"), (8, 5, "sigmoid"), (4, 3, "relu")],
            search_space=search_space,
            dataset_loader=dataset_loader,
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
            num_mfccs=13,
            num_target_classes=2,
            model_optimizer=tf.keras.optimizers.Adam(),
            model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            model_width_dense_layer=10,
            test_size=0.2,
            seed=52,
        )
        self.assertTrue(isinstance(data_model.model, tf.keras.Model))

    def test_better_configuration(self):
        # No need to give any real values to the data model for this test.
        data_model1 = datamodel.DataModel(None, None, None, None, None)  # type: ignore This is a test for other functionality and as such we do not need to provide real values for this constructor
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
        data_model1 = datamodel.DataModel(None, None, None, None, None)  # type: ignore This is a test for other functionality and as such we do not need to provide real values for this constructor
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
        data_model1 = datamodel.DataModel(None, None, None, None, None)  # type: ignore This is a test for other functionality and as such we do not need to provide real values for this constructor
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
        search_space = searchspace.SearchSpace(
            [
                [48000, 24000, 12000, 6000, 3000, 1500, 750, 325],
                ["spectrogram", "mel-spectrogram", "mfcc"],
            ],
            [[2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]],
        )
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(
            return_value=([None, None], [None, None])
        )
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=data.Data(
                X_train=tf.random.uniform((60, 79, 60, 60)),
                X_test=tf.random.uniform((60, 79, 60, 60)),
                y_train=tf.random.uniform((60,)),
                y_test=tf.random.uniform((60,)),
            )
        )
        data_model = datamodel.DataModel.from_data_configuration(
            data_configuration=(24000, "spectrogram"),
            model_configuration=[(8, 5, "relu"), (4, 3, "sigmoid"), (2, 5, "sigmoid")],
            search_space=search_space,
            dataset_loader=dataset_loader,
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
            num_mfccs=13,
            num_target_classes=2,
            model_optimizer=tf.keras.optimizers.Adam(),
            model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            model_width_dense_layer=10,
            test_size=0.2,
            seed=20,
        )

        model_size_without_training = data_model._evaluate_model_size()

        # The model size is not completely deterministic on seperate devices, so we apply an almost equal test.
        self.assertAlmostEqual(model_size_without_training, 329548, places=-2)

    def test_evaluate_model_size2(self):
        search_space = searchspace.SearchSpace(
            [
                [48000, 24000, 12000, 6000, 3000, 1500, 750, 325],
                ["spectrogram", "mel-spectrogram", "mfcc"],
            ],
            [[2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]],
        )
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(return_value=(None))
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=data.Data(
                X_train=tf.random.uniform((60, 79, 60, 60)),
                X_test=tf.random.uniform((60, 79, 60, 60)),
                y_train=tf.random.uniform((60,)),
                y_test=tf.random.uniform((60,)),
            )
        )
        data_model = datamodel.DataModel.from_data_configuration(
            data_configuration=(750, "mfcc"),
            model_configuration=[(2, 3, "sigmoid")],
            search_space=search_space,
            dataset_loader=dataset_loader,
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
            num_mfccs=13,
            num_target_classes=2,
            model_optimizer=tf.keras.optimizers.Adam(),
            model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            model_width_dense_layer=10,
            test_size=0.7,
            seed=20,
        )
        model_size_without_training = data_model._evaluate_model_size()

        self.assertAlmostEqual(model_size_without_training, 364128, places=-2)
