# This python file contains logic for the DataModel class. The conceptual idea of the DataModel class is that an object of this class is an instance of the search space and a potential solution for the Data Aware Neural Architecture Search.

# Standard Library Imports
from __future__ import annotations
import sys
from typing import Any, Optional

# Third Party Imports
import tensorflow as tf
import numpy as np

# Local Imports
import datasetloader
from data import Data
from configuration import Configuration


class DataModel:
    # The primary constructor for the data model class. Assumes that all needed data has already been processed - e.g. data loaded according to data configuration and model created according to model configuration. This constructor is likely not used directly.
    def __init__(
        self,
        configuration: Configuration,
        data: Data | tf.data.Dataset,
        model: tf.keras.Model,
        seed=None,
    ) -> None:
        self.configuration = configuration
        self.data = data
        self.model = model
        self.seed = seed

    # A constructor to use when both data and model need to be created.
    @classmethod
    def from_data_configuration(
        cls,
        configuration: Configuration,
        dataset_loader: datasetloader.DatasetLoader,
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
        model_width_dense_layer: int,
        max_memory_consumption: int,
        test_size: float,
        seed: Optional[int] = None,
        **data_options,
    ) -> DataModel:
        data = cls.create_data(
            configuration.data_configuration,
            dataset_loader,
            test_size,
            **data_options,
        )

        if isinstance(data.X_train, np.ndarray):
            model = cls.create_model(
                configuration.model_configuration,
                data.X_train[0].shape,
                num_target_classes,
                model_optimizer,
                model_loss_function,
                model_width_dense_layer,
                max_memory_consumption,
            )
        elif isinstance(data.X_train, tf.data.Dataset):
            model = cls.create_model(
                configuration.model_configuration,
                data.X_train.element_spec[0].shape[1:],
                num_target_classes,
                model_optimizer,
                model_loss_function,
                model_width_dense_layer,
                max_memory_consumption,
            )
        else:
            raise TypeError(
                "Generated data was neither a np.ndarray or a tf.data.Dataset."
            )

        return cls(
            configuration,
            data,
            model,
            seed,
        )

    # An alternative constructor to use when data is already loaded and only model needs to be created.
    @classmethod
    def from_preloaded_data(
        cls,
        configuration: Configuration,
        data: Data,
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
        model_width_dense_layer: int,
        max_memory_consumption: int,
        seed: Optional[int] = None,
    ) -> DataModel:

        if isinstance(data.X_train, np.ndarray):
            model = cls.create_model(
                configuration.model_configuration,
                data.X_train[0].shape,
                num_target_classes,
                model_optimizer,
                model_loss_function,
                model_width_dense_layer,
                max_memory_consumption,
            )
        if isinstance(data.X_train, tf.data.Dataset):
            model = cls.create_model(
                configuration.model_configuration,
                data.X_train.element_spec[0].shape,
                num_target_classes,
                model_optimizer,
                model_loss_function,
                model_width_dense_layer,
                max_memory_consumption,
            )

        return cls(
            configuration,
            data,
            model,
            seed,
        )

    @staticmethod
    def create_data(
        data_configuration: dict,
        dataset_loader: datasetloader.DatasetLoader,
        test_size: float,
        **options,
    ) -> Data:
        dataset = dataset_loader.configure_dataset(
            **data_configuration,
            **options,
        )

        return dataset_loader.supervised_dataset(dataset, test_size=test_size)

    @staticmethod
    def create_model(
        model_configuration: list[dict],
        data_shape: tuple[int, ...],
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
        model_width_dense_layer: int,
        max_memory_consumption,
    ) -> Optional[tf.keras.Model]:
        model = tf.keras.Sequential()

        # For the first layer we need to define the data shape

        model.add(tf.keras.Input(shape=data_shape))

        try:
            for layer_config in model_configuration:
                if layer_config["type"] == "conv_layer":
                    model.add(
                        tf.keras.layers.Conv2D(
                            filters=layer_config["filters"],
                            kernel_size=layer_config["kernel_size"],
                            activation=layer_config["activation"],
                        )
                    )
                else:
                    raise ValueError(
                        f"The layer type {layer_config['type']} is not yet supported"
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
            metrics=["Accuracy", "Precision", "Recall"],
        )

        model.summary()

        model_size = DataModel._get_model_size(model)

        if model_size > max_memory_consumption:
            print(
                f"Model size too large at {model_size} bytes and a max memory consumption of {max_memory_consumption} bytes."
            )
            model = None

        return model

    @staticmethod
    def _get_model_size(model):
        total_bytes = 0
        for layer in model.layers:
            for weight in layer.weights:
                # Get the weight values
                weight_values = weight.numpy()
                # Get the number of bytes for each weight tensor
                total_bytes += weight_values.nbytes
        return total_bytes

    @staticmethod
    def _get_data_size(data):
        if isinstance(data, np.ndarray):
            size_in_bytes = data[0].nbytes
        elif isinstance(data, tf.data.Dataset):
            sample = next(iter(data.as_numpy_iterator()))
            serialized_sample = tf.io.serialize_tensor(tf.convert_to_tensor(sample))
            size_in_bytes = sys.getsizeof(serialized_sample.numpy())
        return size_in_bytes

    def evaluate_data_model(self, num_epochs: int, batch_size: int) -> None:
        if isinstance(self.data.X_train, np.ndarray):
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
            self.model.evaluate(
                self.data.X_test, y_test, batch_size=batch_size, verbose=1
            )

        elif isinstance(self.data.X_train, tf.data.Dataset):
            self.model.fit(
                self.data.X_train,
                epochs=num_epochs,
                steps_per_epoch=100,
                validation_data=self.data.X_val,
            )
            self.model.evaluate(self.data.X_test, verbose=1)

        # We would like to get accuracy, precision, recall and model size.
        results = self.model.get_metrics_result()

        self.accuracy: float = results["Accuracy"]
        self.precision: float = results["Precision"]
        self.recall: float = results["Recall"]
        self.memory_consumption = self._get_model_size(
            self.model
        ) + self._get_data_size(self.data.X_train)

    def better_accuracy(self, other_datamodel: DataModel) -> bool:
        return self.accuracy > other_datamodel.accuracy

    def better_precision(self, other_datamodel: DataModel) -> bool:
        return self.precision > other_datamodel.precision

    def better_recall(self, other_datamodel: DataModel) -> bool:
        return self.recall > other_datamodel.recall

    def better_memory_consumption(self, other_datamodel: DataModel) -> bool:
        return self.memory_consumption < other_datamodel.memory_consumption

    def better_data_model(self, other_datamodel: DataModel) -> bool:
        return bool(
            np.any(
                np.array(
                    [
                        self.better_accuracy(other_datamodel),
                        self.better_precision(other_datamodel),
                        self.better_recall(other_datamodel),
                        self.better_memory_consumption(other_datamodel),
                    ]
                )
            )
        )

    def free_data_model(self) -> None:
        del self.data
        del self.model
        return
