import datasetloader
import searchspace

import tensorflow as tf
import sklearn.metrics
import numpy as np
import pathlib


class InputModel:
    def __init__(self) -> None:
        pass

    # A "secondary" constructor to allow the creation of an InputModelClass for access to methods without loading datasets a creating neural network models.
    def initialize_input_model(self, input_configuration, model_configuration, search_space: searchspace.SearchSpace, dataset_loader: datasetloader.DatasetLoader, frame_size, hop_length, num_mel_banks, num_mfccs, num_target_classes, model_optimizer, model_loss_function, model_metrics, model_width_dense_layer) -> None:
        self.input = self.create_input(
            input_configuration=input_configuration, search_space=search_space, dataset_loader=dataset_loader, frame_size=frame_size, hop_length=hop_length, num_mel_banks=num_mel_banks, num_mfccs=num_mfccs)
        # We need to subscript the dataset two times.
        # First subscript is to choose the normal files (here we could also chose the abnormal files - doesnt matter)
        # Second subscript is to choose the first entry (all entries should have the same shape)
        self.model = self.create_model(
            model_configuration=model_configuration, search_space=search_space, input_shape=self.input[0][0].shape, num_target_classes=num_target_classes, model_optimizer=model_optimizer, model_loss_function=model_loss_function, model_metrics=model_metrics, model_width_dense_layer=model_width_dense_layer)

    def create_input(self, input_configuration: int, search_space: searchspace.SearchSpace, dataset_loader: datasetloader.DatasetLoader, frame_size, hop_length, num_mel_banks, num_mfccs) -> tuple:
        input_config = search_space.input_decode(input_configuration)

        normal_preprocessed, abnormal_preprocessed = dataset_loader.load_dataset(
            input_config[0], input_config[1], frame_size=frame_size, hop_length=hop_length, num_mel_banks=num_mel_banks, num_mfccs=num_mfccs)

        return dataset_loader.supervised_dataset(normal_preprocessed, abnormal_preprocessed)

    def create_model(self, model_configuration: list[int], search_space, input_shape: tuple, num_target_classes, model_optimizer, model_loss_function, model_metrics, model_width_dense_layer) -> tf.keras.Model:
        layer_configs = search_space.model_decode(model_configuration)

        model = tf.keras.Sequential()

        # For the first layer we need to define the input shape
        model.add(tf.keras.layers.Conv2D(
            filters=layer_configs[0][0], kernel_size=layer_configs[0][1], activation=layer_configs[0][2], input_shape=input_shape))

        try:
            for layer_config in layer_configs[1:]:
                model.add(tf.keras.layers.Conv2D(
                    filters=layer_config[0], kernel_size=layer_config[1], activation=layer_config[2]))
        except ValueError:
            return None

        # The standard convolutional model has dense layers at its end for classification - let us make the same assumption TODO: should be a part of search space
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

    def evaluate_input_model(self, num_epochs, batch_size):
        # Maybe introduce validation data to stop training if the validation error starts increasing.
        X_train, X_test, y_train, y_test = self.input

        self.model.fit(x=X_train, y=y_train, epochs=num_epochs,
                       batch_size=batch_size)

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
        self.model_size = self.__evaluate_model_size()

    def better_accuracy(self, other_configuration):
        return self.accuracy > other_configuration.accuracy

    def better_precision(self, other_configuration):
        return self.precision > other_configuration.precision

    def better_recall(self, other_configuration):
        return self.recall > other_configuration.recall

    def better_model_size(self, other_configuration):
        return self.model_size < other_configuration.model_size

    def better_input_model(self, other_configuration):
        return np.any(np.array([self.better_accuracy(other_configuration), self.better_precision(
            other_configuration), self.better_recall(other_configuration), self.better_model_size(other_configuration)]))

    def __evaluate_model_size(self):
        save_directory = pathlib.Path("./tmp/")
        tf_model_file = save_directory/"tf_model"
        tf.saved_model.save(self.model, tf_model_file)

        converter = tf.lite.TFLiteConverter.from_saved_model(
            tf_model_file.resolve().as_posix())
        tflite_model = converter.convert()
        tflite_model_file = save_directory/"tflite_model"
        model_size = tflite_model_file.write_bytes(tflite_model)
        return model_size
