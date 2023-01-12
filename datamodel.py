# This python file contains logic for the DataModel class. The conceptual idea of the DataModel class is that an object of this class is an instance of the search space and a potential solution for the Data Aware Neural Architecture Search.

# Standard Library Imports
import pathlib
import struct

# Third Party Imports
import tensorflow as tf
import sklearn.metrics
import numpy as np

# Local Imports
import datasetloader
import searchspace


class DataModel:
    # The primary constructor for the data model class. Assumes that all needed data has already been processed - e.g. data loaded according to data configuration and model created according to model configuration. This constructor likely not used directly.
    def __init__(self, data, data_configuration, model, model_configuration, num_normal_samples, num_anomalous_samples, seed=None) -> None:
        self.data = data
        self.data_configuration = data_configuration
        self.model = model
        self.model_configuration = model_configuration
        self.seed = seed
        self.num_normal_samples = num_normal_samples
        self.num_anomalous_samples = num_anomalous_samples

    # A constructor to use when both data and model need to be created.
    @classmethod
    def from_data_configuration(cls, data_configuration, model_configuration, search_space: searchspace.SearchSpace, dataset_loader: datasetloader.DatasetLoader, frame_size, hop_length, num_mel_banks, num_mfccs, num_target_classes, model_optimizer, model_loss_function, model_metrics, model_width_dense_layer, seed=None) -> None:
        data = cls.create_data(data_configuration, search_space, dataset_loader,
                               frame_size, hop_length, num_mel_banks, num_mfccs)
        # For the data shape we need to subscript the dataset two times.
        # First subscript is to choose the training samples (here we could also chose the test samples - doesnt matter)
        # Second subscript is to choose the first entry (all entries should have the same shape)
        model = cls.create_model(model_configuration, search_space, data[0][0].shape, num_target_classes,
                                 model_optimizer, model_loss_function, model_metrics, model_width_dense_layer)

        num_normal_samples = len(dataset_loader.base_normal_audio)
        num_anomalous_samples = len(dataset_loader.base_anomalous_audio)

        return cls(data, data_configuration, model, model_configuration, num_normal_samples, num_anomalous_samples, seed)

    # An alternative constructor to use when data is already loaded and only model needs to be created.
    @classmethod
    def from_preloaded_data(cls, data, num_normal_samples, num_anomalous_samples, data_configuration, model_configuration, search_space: searchspace.SearchSpace, num_target_classes, model_optimizer, model_loss_function, model_metrics, model_width_dense_layer, seed=None) -> None:

        # For the data shape we need to subscript the dataset two times.
        # First subscript is to choose the normal files (here we could also chose the abnormal files - doesnt matter)
        # Second subscript is to choose the first entry (all entries should have the same shape)
        model = cls.create_model(
            model_configuration, search_space, data[0][0].shape, num_target_classes, model_optimizer, model_loss_function, model_metrics, model_width_dense_layer)

        return cls(data, data_configuration, model, model_configuration, num_normal_samples, num_anomalous_samples, seed)

    @staticmethod
    def create_data(data_configuration: int, search_space: searchspace.SearchSpace, dataset_loader: datasetloader.DatasetLoader, frame_size, hop_length, num_mel_banks, num_mfccs) -> tuple:
        data_config = search_space.data_decode(data_configuration)

        normal_preprocessed, anomalous_preprocessed = dataset_loader.load_dataset(
            data_config[0], data_config[1], frame_size=frame_size, hop_length=hop_length, num_mel_banks=num_mel_banks, num_mfccs=num_mfccs)

        return dataset_loader.supervised_dataset(normal_preprocessed, anomalous_preprocessed)

    @staticmethod
    def create_model(model_configuration: list[int], search_space, data_shape: tuple, num_target_classes, model_optimizer, model_loss_function, model_metrics, model_width_dense_layer) -> tf.keras.Model:
        layer_configs = search_space.model_decode(model_configuration)

        model = tf.keras.Sequential()

        # For the first layer we need to define the data shape
        model.add(tf.keras.layers.Conv2D(
            filters=layer_configs[0][0], kernel_size=layer_configs[0][1], activation=layer_configs[0][2], input_shape=data_shape))

        try:
            for layer_config in layer_configs[1:]:
                model.add(tf.keras.layers.Conv2D(
                    filters=layer_config[0], kernel_size=layer_config[1], activation=layer_config[2]))
        except ValueError:
            return None

        # The standard convolutional model has dense layers at its end for classification - let us make the same assumption
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(
            model_width_dense_layer, activation=tf.keras.activations.relu))

        # Output layer
        model.add(tf.keras.layers.Dense(
            num_target_classes, activation=tf.keras.activations.softmax))

        model.compile(optimizer=model_optimizer,
                      loss=model_loss_function, metrics=model_metrics)

        model.summary()

        return model

    def evaluate_data_model(self, num_epochs, batch_size):
        # Maybe introduce validation data to stop training if the validation error starts increasing.
        X_train, X_test, y_train, y_test = self.data

        total_samples = self.num_normal_samples + self.num_anomalous_samples
        weight_for_0 = (1 / self.num_normal_samples) * (total_samples / 2.0)
        weight_for_1 = (1 / self.num_anomalous_samples) * (total_samples / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        self.model.fit(x=X_train, y=y_train, epochs=num_epochs,
                       batch_size=batch_size, class_weight=class_weight)

        y_hat = self.model.predict(X_test, batch_size=batch_size)

        # Transform the output one hot incoding into class indices
        y_hat = tf.math.top_k(input=y_hat, k=1).indices.numpy()[:, 0]

        # We would like to get accuracy, precision, recall and model size.
        self.accuracy = sklearn.metrics.accuracy_score(
            y_true=y_test, y_pred=y_hat)
        self.precision = sklearn.metrics.precision_score(
            y_true=y_test, y_pred=y_hat)
        self.recall = sklearn.metrics.recall_score(
            y_true=y_test, y_pred=y_hat)
        self.model_size = self._evaluate_model_size()

    def better_accuracy(self, other_configuration):
        return self.accuracy > other_configuration.accuracy

    def better_precision(self, other_configuration):
        return self.precision > other_configuration.precision

    def better_recall(self, other_configuration):
        return self.recall > other_configuration.recall

    def better_model_size(self, other_configuration):
        return self.model_size < other_configuration.model_size

    def better_data_model(self, other_configuration):
        return np.any(np.array([self.better_accuracy(other_configuration), self.better_precision(
            other_configuration), self.better_recall(other_configuration), self.better_model_size(other_configuration)]))

    def free_data_model(self):
        del self.data
        del self.model
        return

    def _evaluate_model_size(self):
        unique_extension = self.seed
        save_directory = pathlib.Path("./tmp/")
        save_directory.mkdir(exist_ok=True)

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(
                self.model)
        except Exception as e:
            print(
                f"{e}\nSaving model size as max size + 1 (2000000001) for this to be identified later.")
            return 2000000001
        try:
            tflite_model = converter.convert()
        except struct.error:
            return 2000000000
        except Exception as e:
            print(
                f"{e}\nSaving model size as max size + 1 (2000000001) for this to be identified later.")
            return 2000000001

        tflite_model_file = save_directory/f"tflite_model-{unique_extension}"

        model_size = tflite_model_file.write_bytes(tflite_model)

        tflite_model_file.unlink()

        return model_size
