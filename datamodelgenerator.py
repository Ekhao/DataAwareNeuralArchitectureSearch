# This file contains the DataModelGenerator class which contains the main logic for the data aware neural architecture search. The DataModelGenerator class is responsible for making the search strategy create new configurations, creating data models according to those configurations, evaluating the data models and update the parameters of the search strategy according to this evaluation. Also saves the pareto frontier of data models.

# Standard Library Imports
import csv
import datetime
import pathlib

# Third Party Imports
import numpy as np
import tensorflow as tf

# Local Imports
import datasetloader
import supernet
from searchstrategy import SearchStrategy
from datamodel import DataModel


class DataModelGenerator:
    def __init__(
        self,
        num_target_classes: int,
        loss_function: tf.keras.losses.Loss,
        search_strategy: SearchStrategy,
        dataset_loader: datasetloader.DatasetLoader,
        test_size: float,
        optimizer: tf.keras.optimizers.Optimizer,
        width_dense_layer: int,
        num_epochs: int,
        batch_size: int,
        max_ram_consumption: int,
        max_flash_consumption: int,
        data_dtype_multiplier: int,
        model_dtype_multiplier: int,
        supernet_flag: bool,
        **data_options,
    ) -> None:
        self.num_target_classes = num_target_classes
        self.loss_function = loss_function
        self.search_strategy = search_strategy
        self.search_space = search_strategy.search_space
        self.optimizer = optimizer
        self.width_dense_layer = width_dense_layer
        self.dataset_loader = dataset_loader
        self.test_size = test_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.data_options = data_options
        self.seed = search_strategy.seed
        self.max_ram_consumption = max_ram_consumption
        self.max_flash_consumption = max_flash_consumption
        self.data_dtype_multiplier = data_dtype_multiplier
        self.model_dtype_multiplier = model_dtype_multiplier
        self.supernet_flag = supernet_flag

    def run_data_nas(self, num_of_models: int) -> list[DataModel]:
        pareto_optimal_models = []
        previous_data_configuration = None
        previous_data = None

        save_directory = pathlib.Path("./datamodel_logs/")
        save_directory.mkdir(exist_ok=True)
        csv_log_name = f"datamodel_logs/{datetime.datetime.now().isoformat()}.csv"
        with open(csv_log_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Model Number",
                    "Data Configuration",
                    "Model Configuration",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "Model Size",
                ]
            )

        # Create a dictionary to store supernets (one for each data configuration)
        if self.supernet_flag:
            supernets = {}

        for model_number in range(num_of_models):
            # Print that we are now running a new sample
            print("-" * 100)
            print(f"Starting model number {model_number}")

            # Get configuration from search strategy
            print("Generating configuration...")
            configuration = self.search_strategy.generate_configuration()

            print(
                f"Data configuration: {configuration.data_configuration}\nModel configuration: {configuration.model_configuration}"
            )

            print("Creating data and model from configuration...")
            if configuration.data_configuration != previous_data_configuration:
                data = DataModel.create_data(
                    configuration.data_configuration,
                    dataset_loader=self.dataset_loader,
                    test_size=self.test_size,
                    max_ram_consumption=self.max_ram_consumption,
                    data_dtype_multiplier=self.data_dtype_multiplier,
                    **self.data_options,
                )
            elif previous_data != None:
                data = previous_data
            else:
                raise RuntimeError(
                    "Configuration was same as previous but no previous data was loaded."
                )

            if data == None:
                print("Infeasible data generated. Skipping to next configuration...")
                continue

            if self.supernet_flag:
                if (
                    tuple(configuration.data_configuration.items()) in supernets
                ):  # This seems to not work and each input is considered the same - check out
                    supernet_instance = supernets[
                        tuple(configuration.data_configuration.items())
                    ]
                else:
                    if isinstance(data.X_train, np.ndarray):
                        raise NotImplementedError(
                            "Supernet functionality not yet implemented for data as numpy arrays"
                        )
                    elif isinstance(data.X_train, tf.data.Dataset):
                        supernet_instance = supernet.SuperNet(
                            data=data,
                            num_target_classes=self.num_target_classes,
                            model_optimizer=self.optimizer,
                            model_loss_function=self.loss_function,
                        )
                        supernets[tuple(configuration.data_configuration.items())] = (
                            supernet_instance
                        )
                    else:
                        raise TypeError(
                            "Generated data was neither a np.ndarray or tf.data.Dataset"
                        )
                model = supernet_instance.sample_subnet(
                    **configuration.model_configuration
                )
            else:
                if isinstance(data.X_train, np.ndarray):
                    data_shape = data.X_train[0].shape
                elif isinstance(data.X_train, tf.data.Dataset):
                    data_shape = data.X_train.element_spec[0].shape[1:]
                else:
                    raise TypeError(
                        "Generated data was neither a np.ndarray or tf.data.Dataset"
                    )

                model = DataModel.create_model(
                    model_configuration=configuration.model_configuration,
                    data_shape=data_shape,
                    num_target_classes=self.num_target_classes,
                    model_optimizer=self.optimizer,
                    model_loss_function=self.loss_function,
                    model_width_dense_layer=self.width_dense_layer,
                    max_ram_consumption=self.max_ram_consumption,
                    max_flash_consumption=self.max_flash_consumption,
                    data_dtype_multiplier=self.data_dtype_multiplier,
                    model_dtype_multiplier=self.model_dtype_multiplier,
                )

            if model == None:
                print("Infeasible model generated. Skipping to next configuration...")
                continue

            data_model = DataModel(
                configuration=configuration,
                data=data,
                model=model,
                data_dtype_multiplier=self.data_dtype_multiplier,
                model_dtype_multiplier=self.model_dtype_multiplier,
            )

            print("Evaluating performance of data and model")
            # Evaluate performance of data and model
            data_model.evaluate_data_model(self.num_epochs, self.batch_size)

            print(
                f"Model{model_number} metrics:\nAccuracy: {data_model.accuracy}\nPrecision: {data_model.precision}\nRecall: {data_model.recall}\nRam Consumption (bytes): {data_model.ram_consumption}\nFlash Consumption (bytes): {data_model.flash_consumption}"
            )

            print("Updating parameters of the search strategy...")
            # Update search strategy parameters
            self.search_strategy.update_parameters(data_model)

            print("Freeing loaded data and model to reduce memory consumption...")
            previous_data_configuration = data_model.configuration.data_configuration
            previous_data = data_model.data
            data_model.free_data_model()

            print("Saving DataModel and metrics in logs...")
            self._save_to_csv(csv_log_name, model_number, data_model)

            print("Saving DataModel for pareto front calculation")
            # Save the models that are pareto optimal
            pareto_optimal_models.append(data_model)

        pareto_optimal_models = self._prune_non_pareto_optimal_models(
            pareto_optimal_models
        )
        return pareto_optimal_models

    def _save_to_csv(
        self, csv_log_name: str, model_number: int, data_model: DataModel
    ) -> None:
        with open(csv_log_name, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    model_number,
                    data_model.configuration.data_configuration,
                    data_model.configuration.model_configuration,
                    data_model.accuracy,
                    data_model.precision,
                    data_model.recall,
                    data_model.ram_consumption,
                    data_model.flash_consumption,
                ]
            )

    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    @staticmethod
    def _prune_non_pareto_optimal_models(
        iterative_pareto_optimal_models: list[DataModel],
    ) -> list[DataModel]:
        iterative_pareto_optimal_models_numpy = np.array(
            iterative_pareto_optimal_models
        )
        is_optimal = np.ones(iterative_pareto_optimal_models_numpy.shape[0], dtype=bool)
        for i, model in enumerate(iterative_pareto_optimal_models_numpy):
            if is_optimal[i]:
                is_optimal[is_optimal] = np.array(
                    [
                        x.better_data_model(model)
                        for x in iterative_pareto_optimal_models_numpy[is_optimal]
                    ]
                )
                is_optimal[i] = True
        return list(iterative_pareto_optimal_models_numpy[is_optimal])
