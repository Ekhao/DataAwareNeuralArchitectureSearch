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
import data


class DataModel:
    # The primary constructor for the data model class. Assumes that all needed data has already been processed - e.g. data loaded according to data configuration and model created according to model configuration. This constructor is likely not used directly.
    def __init__(
        self,
        data: data.Data,
        data_configuration: tuple[Any, ...],
        model: tf.keras.Model,
        model_configuration: list[tuple[Any, ...]],
        seed=None,
    ) -> None:
        self.data = data
        self.data_configuration = data_configuration
        self.model = model
        self.model_configuration = model_configuration
        self.seed = seed

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
        test_size: float,
        seed: Optional[int] = None,
        **data_options,
    ) -> DataModel:
        data = cls.create_data(
            data_configuration,
            dataset_loader,
            test_size,
            **data_options,
        )
        # For the data shape we need to subscript the dataset two times.
        # First subscript is to choose the training samples (here we could also chose the test samples - doesnt matter)
        # Second subscript is to choose the first entry (all entries should have the same shape)
        model = cls.create_model(
            model_configuration,
            data.X_train[0].shape,
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
            seed,
        )

    # An alternative constructor to use when data is already loaded and only model needs to be created.
    @classmethod
    def from_preloaded_data(
        cls,
        data: data.Data,
        data_configuration: tuple[Any, ...],
        model_configuration: list[tuple[Any, ...]],
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
        model_width_dense_layer: int,
        seed: Optional[int] = None,
    ) -> DataModel:
        model = cls.create_model(
            model_configuration,
            data.X_train[0].shape,
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
            seed,
        )

    @staticmethod
    def create_data(
        data_configuration: tuple[Any, ...],
        dataset_loader: datasetloader.DatasetLoader,
        test_size: float,
        **options,
    ) -> data.Data:
        dataset = dataset_loader.configure_dataset(
            target_sr=data_configuration[0],
            preprocessing_type=data_configuration[1],
            frame_size=options.get("frame_size"),
            hop_length=options.get("hop_length"),
            num_mel_filters=options.get("num_mel_filters"),
            num_mfccs=options.get("num_mfccs"),
        )

        return dataset_loader.supervised_dataset(dataset, test_size=test_size)

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

        model.add(tf.keras.Input(shape=data_shape))

        try:
            for layer_config in model_configuration:
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
        # Calculations to be used several times in function
        num_classes = max(self.data.y_train) + 1
        assert num_classes == max(self.data.y_test) + 1

        # Turn y_train and y_test into one-hot encoded vectors.
        y_train = tf.one_hot(self.data.y_train, num_classes)
        y_test = tf.one_hot(self.data.y_test, num_classes)

        # If y_val exists do the same for it
        if self.data.y_val != None:
            assert num_classes == max(self.data.y_val) + 1
            y_val = tf.one_hot(self.data.y_val, num_classes)

        total_sample_length = len(self.data.X_train)

        class_weight = {}
        for i in range(num_classes):
            class_sample_length = 0
            for label in self.data.y_train:
                if label == i:
                    class_sample_length += 1
            class_weight[i] = total_sample_length / class_sample_length

        self.model.fit(
            x=self.data.X_train,
            y=y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            class_weight=class_weight,
        )

        self.model.evaluate(self.data.X_test, y_test, batch_size=batch_size)

        # We would like to get accuracy, precision, recall and model size.
        results = self.model.get_metrics_result()

        self.accuracy: float = results["accuracy"]
        self.precision: float = results["precision"]
        self.recall: float = results["recall"]
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
