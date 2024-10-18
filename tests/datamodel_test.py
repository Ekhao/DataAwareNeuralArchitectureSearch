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
from configuration import Configuration


class DataModelTestCase(unittest.TestCase):
    def test_data_model_constructor(self):
        # To keep these tests separate from the dataset we use a mock object as the dataset loader.
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(
            return_value=([None, None], [None, None])
        )
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=data.Data(
                X_train=tf.random.uniform((500, 10, 60, 60)).numpy(),
                X_test=tf.random.uniform((500, 10, 60, 60)).numpy(),
                y_train=tf.random.uniform((500,)).numpy(),
                y_test=tf.random.uniform((500,)).numpy(),
            )
        )
        configuration = Configuration(
            data_configuration={
                "sample_rate": 12000,
                "audio_representation": "spectrogram",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 16,
                    "kernel_size": 3,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 8,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
                {
                    "type": "conv_layer",
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "relu",
                },
            ],
        )
        data_instance = datamodel.DataModel.create_data(
            data_configuration=configuration.data_configuration,
            dataset_loader=dataset_loader,
            test_size=0.2,
            max_ram_consumption=256000,
            max_flash_consumption=1000000,
            data_dtype_multiplier=1,
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
            num_mfccs=13,
        )
        model = datamodel.DataModel.create_model(
            model_configuration=configuration.model_configuration,
            data_shape=data_instance.X_train[0].shape,
            num_target_classes=2,
            model_optimizer=tf.keras.optimizers.Adam(),
            model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            model_width_dense_layer=10,
            max_ram_consumption=256000,
            max_flash_consumption=1000000,
            data_dtype_multiplier=1,
            model_dtype_multiplier=1,
        )
        data_model = datamodel.DataModel(
            configuration=configuration,
            data=data_instance,
            model=model,
            data_dtype_multiplier=1,
            model_dtype_multiplier=1,
        )
        self.assertTrue(isinstance(data_model.model, tf.keras.Model))

    def test_better_configuration(self):
        # No need to give any real values to the data model for this test.
        data_model1 = datamodel.DataModel(None, None, None, None, None, None)  # type: ignore This is a test for other functionality and as such we do not need to provide real values for this constructor
        data_model1.accuracy = 0.98
        data_model1.precision = 0.7
        data_model1.recall = 0.99
        data_model1.ram_consumption = 3954353
        data_model1.flash_consumption = 3943354
        data_model2 = copy.deepcopy(data_model1)
        data_model2.accuracy = 0.9
        data_model2.precision = 0.69
        data_model2.recall = 0.89
        data_model2.ram_consumption = 4054353
        data_model2.flash_consumption = 3943354

        self.assertTrue(data_model1.better_data_model(data_model2))

    def test_not_better_configuration(self):
        # No need to give any real values to the data model for this test.
        data_model1 = datamodel.DataModel(None, None, None, None, None, None)  # type: ignore This is a test for other functionality and as such we do not need to provide real values for this constructor
        data_model1.accuracy = 0.60
        data_model1.precision = 0.56
        data_model1.recall = 0.82
        data_model1.ram_consumption = 3954353
        data_model1.flash_consumption = 3943354
        data_model2 = copy.deepcopy(data_model1)
        data_model2.accuracy = 0.02
        data_model2.precision = 0.55
        data_model2.recall = 0.82
        data_model2.ram_consumption = 4054353
        data_model2.flash_consumption = 3943354
        self.assertFalse(data_model2.better_data_model(data_model1))

    def test_better_configuration_model_size(self):
        data_model1 = datamodel.DataModel(None, None, None, None, None, None)  # type: ignore This is a test for other functionality and as such we do not need to provide real values for this constructor
        data_model1.accuracy = 0.9
        data_model1.precision = 0.8
        data_model1.recall = 0.8
        data_model1.ram_consumption = 6954353
        data_model1.flash_consumption = 3943354
        data_model2 = copy.deepcopy(data_model1)
        data_model2.accuracy = 0.5
        data_model2.precision = 0.55
        data_model2.recall = 0.4
        data_model2.ram_consumption = 4054353
        data_model2.flash_consumption = 3943321
        self.assertTrue(data_model2.better_data_model(data_model1))

    def test_evaluate_model_size(self):
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(
            return_value=([None, None], [None, None])
        )
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=data.Data(
                X_train=tf.random.uniform((60, 79, 60, 60)).numpy(),
                X_test=tf.random.uniform((60, 79, 60, 60)).numpy(),
                y_train=tf.random.uniform((60,)).numpy(),
                y_test=tf.random.uniform((60,)).numpy(),
            )
        )
        configuration = Configuration(
            data_configuration={
                "sample_rate": 24000,
                "audio_representation": "spectrogram",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 8,
                    "kernel_size": 5,
                    "activation": "relu",
                },
                {
                    "type": "conv_layer",
                    "filters": 4,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 5,
                    "activation": "sigmoid",
                },
            ],
        )

        data_instance = datamodel.DataModel.create_data(
            data_configuration=configuration.data_configuration,
            dataset_loader=dataset_loader,
            test_size=0.2,
            max_ram_consumption=1500000,
            max_flash_consumption=1000000,
            data_dtype_multiplier=1,
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
            num_mfccs=13,
        )
        model = datamodel.DataModel.create_model(
            model_configuration=configuration.model_configuration,
            data_shape=data_instance.X_train[0].shape,
            num_target_classes=2,
            model_optimizer=tf.keras.optimizers.Adam(),
            model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            model_width_dense_layer=10,
            max_ram_consumption=1500000,
            max_flash_consumption=1000000,
            data_dtype_multiplier=1,
            model_dtype_multiplier=1,
        )
        data_model = datamodel.DataModel(
            configuration=configuration,
            data=data_instance,
            model=model,
            data_dtype_multiplier=1,
            model_dtype_multiplier=1,
        )

        model_size_without_training = data_model._get_model_size(
            data_model.model, model_dtype_multiplier=1
        )

        self.assertEqual(model_size_without_training, 81534)

    def test_evaluate_model_size2(self):
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(return_value=(None))
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=data.Data(
                X_train=tf.random.uniform((60, 79, 60, 60)).numpy(),
                X_test=tf.random.uniform((60, 79, 60, 60)).numpy(),
                y_train=tf.random.uniform((60,)).numpy(),
                y_test=tf.random.uniform((60,)).numpy(),
            )
        )
        configuration = Configuration(
            data_configuration={
                "sample_rate": 750,
                "audio_representation": "mfcc",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
            ],
        )

        data_instance = datamodel.DataModel.create_data(
            data_configuration=configuration.data_configuration,
            dataset_loader=dataset_loader,
            test_size=0.7,
            max_ram_consumption=1500000,
            max_flash_consumption=1000000,
            data_dtype_multiplier=1,
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
            num_mfccs=13,
        )
        model = datamodel.DataModel.create_model(
            model_configuration=configuration.model_configuration,
            data_shape=data_instance.X_train[0].shape,
            num_target_classes=2,
            model_optimizer=tf.keras.optimizers.Adam(),
            model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            model_width_dense_layer=10,
            max_ram_consumption=1500000,
            max_flash_consumption=1000000,
            data_dtype_multiplier=1,
            model_dtype_multiplier=2,
        )
        data_model = datamodel.DataModel(
            configuration=configuration,
            data=data_instance,
            model=model,
            data_dtype_multiplier=1,
            model_dtype_multiplier=1,
        )

        model_size_without_training = data_model._get_model_size(
            data_model.model, model_dtype_multiplier=2
        )

        self.assertEqual(model_size_without_training, 180868)

    def test_get_data_size(self):
        dataset_loader = unittest.mock.MagicMock()
        dataset_loader.load_dataset = unittest.mock.Mock(return_value=(None))
        dataset_loader.supervised_dataset = unittest.mock.Mock(
            return_value=data.Data(
                X_train=tf.random.uniform((60, 79, 60, 60)).numpy(),
                X_test=tf.random.uniform((60, 79, 60, 60)).numpy(),
                y_train=tf.random.uniform((60,)).numpy(),
                y_test=tf.random.uniform((60,)).numpy(),
            )
        )
        configuration = Configuration(
            data_configuration={
                "sample_rate": 750,
                "audio_representation": "mfcc",
            },
            model_configuration=[
                {
                    "type": "conv_layer",
                    "filters": 2,
                    "kernel_size": 3,
                    "activation": "sigmoid",
                },
            ],
        )

        data_instance = datamodel.DataModel.create_data(
            data_configuration=configuration.data_configuration,
            dataset_loader=dataset_loader,
            test_size=0.7,
            max_ram_consumption=1500000,
            max_flash_consumption=1000000,
            data_dtype_multiplier=2,
            frame_size=2048,
            hop_length=512,
            num_mel_filters=80,
            num_mfccs=13,
        )
        model = datamodel.DataModel.create_model(
            model_configuration=configuration.model_configuration,
            data_shape=data_instance.X_train[0].shape,
            num_target_classes=2,
            model_optimizer=tf.keras.optimizers.Adam(),
            model_loss_function=tf.keras.losses.SparseCategoricalCrossentropy(),
            model_width_dense_layer=10,
            max_ram_consumption=1500000,
            max_flash_consumption=1000000,
            data_dtype_multiplier=2,
            model_dtype_multiplier=1,
        )
        data_model = datamodel.DataModel(
            configuration=configuration,
            data=data_instance,
            model=model,
            data_dtype_multiplier=1,
            model_dtype_multiplier=1,
        )

        data_size = data_model._get_data_size(
            data_model.data.X_train, data_dtype_multiplier=2
        )

        self.assertEqual(data_size, 568800)
