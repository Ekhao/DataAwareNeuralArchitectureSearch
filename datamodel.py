# This python file contains logic for the DataModel class. The conceptual idea of the DataModel class is that an object of this class is an instance of the search space and a potential solution for the Data Aware Neural Architecture Search.

# Standard Library Imports
from __future__ import annotations
import pathlib
import struct
from typing import Any, Optional

# Third Party Imports
import tensorflow as tf
import numpy as np

# Local Imports
import datasetloader

Data = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class DataModel:
    # The primary constructor for the data model class. Assumes that all needed data has already been processed - e.g. data loaded according to data configuration and model created according to model configuration. This constructor is likely not used directly.
    def __init__(
        self,
        data: Data,
        data_configuration: tuple[Any, ...],
        model: tf.keras.Model,
        model_configuration: list[tuple[Any, ...]],
        num_samples_per_class: dict[int, int],
        seed=None,
    ) -> None:
        self.data = data
        self.data_configuration = data_configuration
        self.model = model
        self.model_configuration = model_configuration
        self.seed = seed
        self.num_samples_per_class = num_samples_per_class

    # A constructor to use when both data and model need to be created.
    @classmethod
    def from_data_configuration(
        cls,
        data_configuration: tuple[Any, ...],
        model_configuration: list[tuple[Any, ...]],
        dataset_loader: datasetloader.DatasetLoader,
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
        model_width_dense_layer: int,
        seed: Optional[int] = None,
        **data_options,
    ) -> DataModel:
        data = cls.create_data(
            data_configuration,
            dataset_loader,
            **data_options,
        )
        # For the data shape we need to subscript the dataset two times.
        # First subscript is to choose the training samples (here we could also chose the test samples - doesnt matter)
        # Second subscript is to choose the first entry (all entries should have the same shape)
        model = cls.create_model(
            model_configuration,
            data[0][0].shape,
            num_target_classes,
            model_optimizer,
            model_loss_function,
            model_width_dense_layer,
        )

        return cls(
            data,
            data_configuration,
            model,
            model_configuration,
            dataset_loader.num_samples_per_class(),
            seed,
        )

    # An alternative constructor to use when data is already loaded and only model needs to be created.
    @classmethod
    def from_preloaded_data(
        cls,
        data: Data,
        num_samples_per_class: dict[int, int],
        data_configuration: tuple[Any, ...],
        model_configuration: list[tuple[Any, ...]],
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
        model_width_dense_layer: int,
        seed: Optional[int] = None,
    ) -> DataModel:
        # For the data shape we need to subscript the dataset two times.
        # First subscript is to choose the normal files (here we could also chose the abnormal files - doesnt matter)
        # Second subscript is to choose the first entry (all entries should have the same shape)
        model = cls.create_model(
            model_configuration,
            data[0][0].shape,
            num_target_classes,
            model_optimizer,
            model_loss_function,
            model_width_dense_layer,
        )

        return cls(
            data,
            data_configuration,
            model,
            model_configuration,
            num_samples_per_class,
            seed,
        )

    @staticmethod
    def create_data(
        data_configuration: tuple[Any, ...],
        dataset_loader: datasetloader.DatasetLoader,
        **options,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dataset = dataset_loader.load_dataset(
            target_sr=data_configuration[0],
            preprocessing_type=data_configuration[1],
            frame_size=options.get("frame_size"),
            hop_length=options.get("hop_length"),
            num_mel_filters=options.get("num_mel_filters"),
            num_mfccs=options.get("num_mfccs"),
        )

        return dataset_loader.supervised_dataset(dataset)

    @staticmethod
    def create_model(
        model_configuration: list[tuple[Any, ...]],
        data_shape: tuple[int, ...],
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
        model_width_dense_layer: int,
    ) -> Optional[tf.keras.Model]:
        model = tf.keras.Sequential()

        # For the first layer we need to define the data shape
        model.add(
            tf.keras.layers.Conv2D(
                filters=model_configuration[0][0],
                kernel_size=model_configuration[0][1],
                activation=model_configuration[0][2],
                input_shape=data_shape,
            )
        )

        try:
            for layer_config in model_configuration[1:]:
                model.add(
                    tf.keras.layers.Conv2D(
                        filters=layer_config[0],
                        kernel_size=layer_config[1],
                        activation=layer_config[2],
                    )
                )
        except ValueError:
            return None

        # The standard convolutional model has dense layers at its end for classification - let us make the same assumption
        model.add(tf.keras.layers.Flatten())

        model.add(
            tf.keras.layers.Dense(
                model_width_dense_layer, activation=tf.keras.activations.relu
            )
        )

        # Output layer
        model.add(
            tf.keras.layers.Dense(
                num_target_classes, activation=tf.keras.activations.softmax
            )
        )

        model.compile(
            optimizer=model_optimizer,
            loss=model_loss_function,
            metrics=[
                tf.keras.metrics.Accuracy(),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        model.summary()

        return model

    def evaluate_data_model(self, num_epochs: int, batch_size: int) -> None:
        # Maybe introduce validation data to stop training if the validation error starts increasing.
        X_train, X_test, y_train, y_test = self.data

        # Turn y_train and y_test into one-hot encoded vectors.
        y_train = tf.one_hot(y_train, 2)
        y_test = tf.one_hot(y_test, 2)

        total_sample_length = 0
        for sample_length in self.num_samples_per_class.values():
            total_sample_length += sample_length

        class_weight = {}
        i = 0
        number_of_classes = len(self.num_samples_per_class)
        for sample_length in self.num_samples_per_class.values():
            class_weight[i] = (1 / sample_length) * (
                total_sample_length / number_of_classes
            )

        self.model.fit(
            x=X_train,
            y=y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            class_weight=class_weight,
        )

        self.model.evaluate(X_test, y_test, batch_size=batch_size)

        # We would like to get accuracy, precision, recall and model size.
        results = self.model.get_metrics_result()

        self.accuracy: float = results["accuracy"].numpy()
        self.precision: float = results["precision"].numpy()
        self.recall: float = results["recall"].numpy()
        self.model_size = self._evaluate_model_size()

    def better_accuracy(self, other_datamodel: DataModel) -> bool:
        return self.accuracy > other_datamodel.accuracy

    def better_precision(self, other_datamodel: DataModel) -> bool:
        return self.precision > other_datamodel.precision

    def better_recall(self, other_datamodel: DataModel) -> bool:
        return self.recall > other_datamodel.recall

    def better_model_size(self, other_datamodel: DataModel) -> bool:
        return self.model_size < other_datamodel.model_size

    def better_data_model(self, other_datamodel: DataModel) -> bool:
        return bool(
            np.any(
                np.array(
                    [
                        self.better_accuracy(other_datamodel),
                        self.better_precision(other_datamodel),
                        self.better_recall(other_datamodel),
                        self.better_model_size(other_datamodel),
                    ]
                )
            )
        )

    def free_data_model(self) -> None:
        del self.data
        del self.model
        return

    def _evaluate_model_size(self) -> int:
        unique_extension = self.seed
        save_directory = pathlib.Path("./tmp/")
        save_directory.mkdir(exist_ok=True)

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        except Exception as e:
            print(
                f"{e}\nSaving model size as max size + 1 (2000000001) for this to be identified later."
            )
            return 2000000001
        try:
            tflite_model = converter.convert()
        except struct.error:
            return 2000000000
        except Exception as e:
            print(
                f"{e}\nSaving model size as max size + 1 (2000000001) for this to be identified later."
            )
            return 2000000001

        tflite_model_file = save_directory / f"tflite_model-{unique_extension}"

        model_size = tflite_model_file.write_bytes(tflite_model)

        tflite_model_file.unlink()

        return model_size
