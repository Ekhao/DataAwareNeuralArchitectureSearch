# This python file contains logic for the DataModel class. The conceptual idea of the DataModel class is that an object of this class is an instance of the search space and a potential solution for the Data Aware Neural Architecture Search.

# Standard Library Imports
from __future__ import annotations
from typing import Any, Optional

# Third Party Imports
import tensorflow as tf
import numpy as np

# Local Imports
import datasetloader
from data import Data
from configuration import Configuration

WAKE_VISION_STEPS_PER_EPOCH = 10


class DataModel:
    # The primary constructor for the data model class. Assumes that all needed data has already been processed - e.g. data loaded according to data configuration and model created according to model configuration. This constructor is likely not used directly.
    def __init__(
        self,
        configuration: Configuration,
        data: Data | tf.data.Dataset,
        model: tf.keras.Model,
        data_dtype_multiplier: int,
        model_dtype_multiplier: int,
        seed=None,
    ) -> None:
        self.configuration = configuration
        self.data = data
        self.model = model
        self.seed = seed
        self.data_dtype_multiplier = data_dtype_multiplier
        self.model_dtype_multiplier = model_dtype_multiplier

    @staticmethod
    def create_data(
        data_configuration: dict,
        dataset_loader: datasetloader.DatasetLoader,
        test_size: float,
        max_ram_consumption: int,
        data_dtype_multiplier: int,
        **options,
    ) -> Data:
        dataset = dataset_loader.configure_dataset(
            **data_configuration,
            **options,
        )

        dataset = dataset_loader.supervised_dataset(dataset, test_size=test_size)

        data_ram_consumption = DataModel._get_data_size(
            dataset.X_train, data_dtype_multiplier
        )

        if data_ram_consumption > max_ram_consumption:
            print(
                f"A single data input consumes more memory than maximum ram consumption. {data_ram_consumption} out of {max_ram_consumption}."
            )
            dataset = None

        return dataset

    @staticmethod
    def create_model(
        model_configuration: list[dict],
        data_shape: tuple[int, ...],
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
        model_width_dense_layer: int,
        max_ram_consumption: int,
        max_flash_consumption: int,
        data_dtype_multiplier: int,
        model_dtype_multiplier: int,
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

        # Check that model can fit in max flash consumption
        model_size = DataModel._get_model_size(model, model_dtype_multiplier)

        if model_size > max_flash_consumption:
            print(
                f"Model size too large at {model_size} bytes and a max flash consumption of {max_flash_consumption} bytes."
            )
            model = None

        # Check that any intermediate representation can fit in max ram consumption
        if model != None:
            max_tensor_memory = DataModel._get_max_internal_representation_size(
                model, data_dtype_multiplier
            )

            if max_tensor_memory > max_ram_consumption:
                print(
                    f"The maximum internal tensor representation will not be able to fit in maximum ram consumption. Using {max_tensor_memory} out of {max_ram_consumption}"
                )
                model = None

        return model

    @staticmethod
    def _get_model_size(model, model_dtype_multiplier: int):
        total_bytes = 0
        for layer in model.layers:
            for weight in layer.weights:
                # Get the weight values
                weight_values = weight.numpy()
                # Get the number of bytes for each weight tensor
                total_bytes += weight_values.size * model_dtype_multiplier
        return total_bytes

    @staticmethod
    def _get_data_size(data, data_dtype_multiplier: int):
        if isinstance(data, np.ndarray):
            size_of_sample = data[0].size
            size_in_bytes = size_of_sample * data_dtype_multiplier
        elif isinstance(data, tf.data.Dataset):
            data = data.unbatch()
            sample = next(iter(data.as_numpy_iterator()))
            size_of_sample = tf.size(
                sample[0]
            )  # 0 to select input features and not output targets
            size_in_bytes = size_of_sample * data_dtype_multiplier
        return size_in_bytes

    @staticmethod
    def _get_max_internal_representation_size(model, data_dtype_multiplier: int):
        max_tensor_memory = 0

        for layer in model.layers:
            input_shape = layer.input.shape
            output_shape = layer.output.shape
            input_size = np.prod(input_shape[1:])  # Exclude the batch size
            output_size = np.prod(output_shape[1:])
            input_memory = input_size * data_dtype_multiplier
            output_memory = output_size * data_dtype_multiplier
            tensor_memory = input_memory + output_memory
            max_tensor_memory = max(max_tensor_memory, tensor_memory)

        return max_tensor_memory

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
                verbose=2,
            )
            self.model.evaluate(
                self.data.X_test, y_test, batch_size=batch_size, verbose=2
            )

        elif isinstance(self.data.X_train, tf.data.Dataset):
            self.model.fit(
                self.data.X_train,
                epochs=num_epochs,
                steps_per_epoch=WAKE_VISION_STEPS_PER_EPOCH,
                validation_data=self.data.X_val,
                verbose=2,
            )
            self.model.evaluate(self.data.X_test, verbose=2)

        # We would like to get accuracy, precision, recall and model size.
        results = self.model.get_metrics_result()

        self.accuracy: float = results["Accuracy"]
        self.precision: float = results["Precision"]
        self.recall: float = results["Recall"]
        self.ram_consumption = max(
            self._get_data_size(self.data.X_train, self.data_dtype_multiplier),
            self._get_max_internal_representation_size(
                self.model, self.data_dtype_multiplier
            ),
        )
        self.flash_consumption = self._get_model_size(
            self.model, self.model_dtype_multiplier
        )

    def better_accuracy(self, other_datamodel: DataModel) -> bool:
        return self.accuracy > other_datamodel.accuracy

    def better_precision(self, other_datamodel: DataModel) -> bool:
        return self.precision > other_datamodel.precision

    def better_recall(self, other_datamodel: DataModel) -> bool:
        return self.recall > other_datamodel.recall

    def better_ram_consumption(self, other_datamodel: DataModel) -> bool:
        return self.ram_consumption < other_datamodel.ram_consumption

    def better_flash_consumption(self, other_datamodel: DataModel) -> bool:
        return self.flash_consumption < other_datamodel.flash_consumption

    def better_data_model(self, other_datamodel: DataModel) -> bool:
        return bool(
            np.any(
                np.array(
                    [
                        self.better_accuracy(other_datamodel),
                        self.better_precision(other_datamodel),
                        self.better_recall(other_datamodel),
                        self.better_ram_consumption(other_datamodel),
                        self.better_flash_consumption(other_datamodel),
                    ]
                )
            )
        )

    def free_data_model(self) -> None:
        del self.data
        del self.model
        return
