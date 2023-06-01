# This file contains the DataModelGenerator class which contains the main logic for the data aware neural architecture search. The DataModelGenerator class is responsible for making the controller create new configurations, creating data models according to those configurations, evaluating the data models and update the parameters of the controller according to this evaluation. Also saves the pareto frontier of data models.

# Standard Library Imports
import csv
import datetime
import pathlib

# Third Party Imports
import numpy as np
import tensorflow as tf

# Local Imports
import datamodel
import datasetloader
from controller import Controller
from datamodel import DataModel


class DataModelGenerator:
    def __init__(
        self,
        num_target_classes: int,
        loss_function: tf.keras.losses.Loss,
        controller: Controller,
        dataset_loader: datasetloader.DatasetLoader,
        optimizer: tf.keras.optimizers.Optimizer,
        width_dense_layer: int,
        num_epochs: int,
        batch_size: int,
        number_of_normal_files: int,
        number_of_anomalous_files: int,
        frame_size: int,
        hop_length: int,
        num_mel_banks: int,
        num_mfccs: int,
    ) -> None:
        self.num_target_classes = num_target_classes
        self.loss_function = loss_function
        self.controller = controller
        self.search_space = controller.search_space
        self.optimizer = optimizer
        self.width_dense_layer = width_dense_layer
        self.dataset_loader = dataset_loader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.number_of_normal_files = number_of_normal_files
        self.number_of_anomalous_files = number_of_anomalous_files
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.num_mel_banks = num_mel_banks
        self.num_mfccs = num_mfccs
        self.seed = controller.seed

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

        for model_number in range(num_of_models):
            # Print that we are now running a new sample
            print("-" * 100)
            print(f"Starting model number {model_number}")

            # Get configuration from controller
            print("Generating model configuration...")
            (
                data_configuration,
                model_configuration,
            ) = self.controller.generate_configuration()

            print(
                f"Data configuration: {data_configuration}\nModel configuration: {model_configuration}"
            )

            print("Creating data and model from configuration...")
            if data_configuration != previous_data_configuration:
                # Create data and model from configuration
                data_model = datamodel.DataModel.from_data_configuration(
                    data_configuration,
                    model_configuration,
                    self.search_space,
                    self.dataset_loader,
                    self.frame_size,
                    self.hop_length,
                    self.num_mel_banks,
                    self.num_mfccs,
                    self.num_target_classes,
                    self.optimizer,
                    self.loss_function,
                    self.width_dense_layer,
                    self.seed,
                )
            elif previous_data != None:
                # Use previous data and create model from configuration
                data_model = datamodel.DataModel.from_preloaded_data(
                    previous_data,
                    self.number_of_normal_files,
                    self.number_of_anomalous_files,
                    data_configuration,
                    model_configuration,
                    self.search_space,
                    self.num_target_classes,
                    self.optimizer,
                    self.loss_function,
                    model_width_dense_layer=self.width_dense_layer,
                    seed=self.seed,
                )
            else:
                raise RuntimeError(
                    "Configuration was same as previous but no previous data was loaded. This should not happen."
                )

            # Some data and model configurations are infeasible. In this case the model created in the data model will be None.
            # If we create an infeasible datamodel we simply skip to proposing the next model
            if data_model.model == None:
                print("Infeasible model generated. Skipping to next configuration...")
                continue

            print("Evaluating performance of data and model")
            # Evaluate performance of data and model
            data_model.evaluate_data_model(self.num_epochs, self.batch_size)

            print(
                f"Model{model_number} metrics:\nAccuracy: {data_model.accuracy}\nPrecision: {data_model.precision}\nRecall: {data_model.recall}\nModel Size (bytes): {data_model.model_size}"
            )

            print("Updating parameters of the controller...")
            # Update controller parameters
            self.controller.update_parameters(data_model)

            print("Freeing loaded data and model to reduce memory consumption...")
            previous_data_configuration = data_model.data_configuration
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
                    data_model.data_configuration,
                    data_model.model_configuration,
                    data_model.accuracy,
                    data_model.precision,
                    data_model.recall,
                    data_model.model_size,
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
