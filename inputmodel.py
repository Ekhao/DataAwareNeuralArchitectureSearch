import datasetloader
import searchspace

import tensorflow as tf
import sklearn.metrics
import numpy as np
import pathlib
import struct
import time


class InputModel:
    def __init__(self) -> None:
        pass

    # A "secondary" constructor to allow the creation of an InputModelClass for access to methods without loading datasets a creating neural network models.
    def initialize_input_model(self, input_configuration, model_configuration, search_space: searchspace.SearchSpace, dataset_loader: datasetloader.DatasetLoader, frame_size, hop_length, num_mel_banks, num_mfccs, num_target_classes, model_optimizer, model_loss_function, model_metrics, model_width_dense_layer, seed) -> None:
        self.input_configuration = input_configuration
        self.model_configuration = model_configuration
        self.seed = seed

        self.input = self.create_input(
            input_configuration=input_configuration, search_space=search_space, dataset_loader=dataset_loader, frame_size=frame_size, hop_length=hop_length, num_mel_banks=num_mel_banks, num_mfccs=num_mfccs)
        # We need to subscript the dataset two times.
        # First subscript is to choose the normal files (here we could also chose the abnormal files - doesnt matter)
        # Second subscript is to choose the first entry (all entries should have the same shape)
        self.model = self.create_model(
            model_configuration=model_configuration, search_space=search_space, input_shape=self.input[0][0].shape, num_target_classes=num_target_classes, model_optimizer=model_optimizer, model_loss_function=model_loss_function, model_metrics=model_metrics, model_width_dense_layer=model_width_dense_layer)

    def create_input(self, input_configuration: int, search_space: searchspace.SearchSpace, dataset_loader: datasetloader.DatasetLoader, frame_size, hop_length, num_mel_banks, num_mfccs) -> tuple:
        input_config = search_space.input_decode(input_configuration)

        normal_preprocessed, anomalous_preprocessed = dataset_loader.load_dataset(
            input_config[0], input_config[1], frame_size=frame_size, hop_length=hop_length, num_mel_banks=num_mel_banks, num_mfccs=num_mfccs)

        # Setting amount of normal and anomalous samples for weighing the model
        self.num_normal_samples = len(normal_preprocessed)
        self.num_anomalous_samples = len(anomalous_preprocessed)

        return dataset_loader.supervised_dataset(normal_preprocessed, anomalous_preprocessed)

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

    def evaluate_input_model(self, num_epochs, batch_size):
        # Maybe introduce validation data to stop training if the validation error starts increasing.
        X_train, X_test, y_train, y_test = self.input

        total_samples = self.num_normal_samples + self.num_anomalous_samples
        weight_for_0 = (1 / self.num_normal_samples) * (total_samples / 2.0)
        weight_for_1 = (1 / self.num_anomalous_samples) * (total_samples / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        self.model.fit(x=X_train, y=y_train, epochs=num_epochs,
                       batch_size=batch_size, class_weight=class_weight)

        start = time.perf_counter()
        print("Start predicting")

        y_hat = self.model.predict(X_test, batch_size=batch_size)

        print(
            f"Done predicting, start transforming. Time predict took: {time.perf_counter()-start}")

        # Transform the output one hot incoding into class indices
        y_hat = tf.math.top_k(input=y_hat, k=1).indices.numpy()[:, 0]

        print("Done transforming, calculating accuracy")

        # We would like to get accuracy, precision, recall and model size.
        self.accuracy = sklearn.metrics.accuracy_score(
            y_true=y_test, y_pred=y_hat)
        print("Done accuracy, calculating precision")
        self.precision = sklearn.metrics.precision_score(
            y_true=y_test, y_pred=y_hat)

        print("Done precision, calculating recall")
        self.recall = sklearn.metrics.recall_score(
            y_true=y_test, y_pred=y_hat)

        print("Done recall, calculating model_size")
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

    def free_input_model(self):
        del self.input
        del self.model
        return

    def __evaluate_model_size(self):
        start = time.perf_counter()

        print("Create directory...")
        unique_extension = self.seed
        save_directory = pathlib.Path("./tmp/")
        save_directory.mkdir(exist_ok=True)

        #tf_model_file = save_directory/f"tf_model-{unique_extension}"
        #tf.saved_model.save(self.model, tf_model_file)

        try:
            print(f"Convert1 model...")
            converter = tf.lite.TFLiteConverter.from_keras_model(
                self.model)
            convert1 = time.perf_counter()
            print(f"Converting1 model  took: {start-convert1}")
        except Exception as e:
            print(
                f"{e}\nSaving model size as max size + 1 (2000000001) for this to be identified later.")
            return 2000000001
        try:
            print(f"Convert2 model..")
            tflite_model = converter.convert()
            convert2 = time.perf_counter()
            print(f"Converting2 model took: {convert1-convert2}")
        except struct.error:
            return 2000000000
        except Exception as e:
            print(
                f"{e}\nSaving model size as max size + 1 (2000000001) for this to be identified later.")
            return 2000000001

        tflite_model_file = save_directory/f"tflite_model-{unique_extension}"

        print("Start saving model...")
        model_size = tflite_model_file.write_bytes(tflite_model)
        save = time.perf_counter()
        print(f"Saving model took: {convert2-save}")

        print("Start removing model...")
        tflite_model_file.unlink()
        remove = time.perf_counter()
        print(f"Removing model took: {save-remove}")

        return model_size
