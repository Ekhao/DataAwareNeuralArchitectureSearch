import numpy as np
import csv
import datetime
import pathlib

import datamodel
import datasetloader
import constants


class DataModelGenerator:
    def __init__(self, num_target_classes, loss_function, controller, dataset_loader: datasetloader.DatasetLoader, optimizer="Adam", metrics=["accuracy"], width_dense_layer=constants.WIDTH_OF_DENSE_LAYER, num_epochs=constants.NUM_EPOCHS, batch_size=constants.BATCH_SIZE, number_of_normal_files=constants.NUMBER_OF_NORMAL_FILES_TO_USE, number_of_anomalous_files=constants.NUMBER_OF_ANOMALOUS_FILES_TO_USE, path_to_normal_files=constants.PATH_TO_NORMAL_FILES, path_to_anomalous_files=constants.PATH_TO_ANOMALOUS_FILES, frame_size=constants.FRAME_SIZE, hop_length=constants.HOP_LENGTH, num_mel_banks=constants.NUMBER_OF_MEL_FILTER_BANKS, num_mfccs=constants.NUMBER_OF_MFCCS):
        self.num_target_classes = num_target_classes
        self.loss_function = loss_function
        self.controller = controller
        self.search_space = controller.search_space
        self.optimizer = optimizer
        self.metrics = metrics
        self.width_dense_layer = width_dense_layer
        self.dataset_loader = dataset_loader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.number_of_normal_files = number_of_normal_files
        self.number_of_anomalous_files = number_of_anomalous_files
        self.path_to_normal_files = path_to_normal_files
        self.path_to_anomalous_files = path_to_anomalous_files
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.num_mel_banks = num_mel_banks
        self.num_mfccs = num_mfccs
        self.seed = controller.seed

    def run_data_nas(self, num_of_models):
        pareto_optimal_models = []
        previous_data_configuration = None
        previous_data = None

        save_directory = pathlib.Path("./datamodel_logs/")
        save_directory.mkdir(exist_ok=True)
        csv_log_name = f"datamodel_logs/{datetime.datetime.now().isoformat()}.csv"
        with open(csv_log_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Model Number", "Data Configuration",
                            "Model Configuration", "Accuracy", "Precision", "Recall", "Model Size"])

        for model_number in range(num_of_models):
            # Print that we are now running a new sample
            print("-"*100)
            print(f"Starting model number {model_number}")

            # Get configuration from controller
            print("Generating model configuration...")
            data_configuration, model_configuration = self.controller.generate_configuration()

            print(
                f"Data configuration: {self.search_space.data_decode(data_configuration)}\nModel configuration: {self.search_space.model_decode(model_configuration)}")

            print("Creating data and model from configuration...")
            if data_configuration != previous_data_configuration:
                # Create data and model from configuration
                data_model = datamodel.DataModel()
                data_model.initialize_data_model(data_configuration=data_configuration, model_configuration=model_configuration, search_space=self.search_space, dataset_loader=self.dataset_loader, frame_size=self.frame_size, hop_length=self.hop_length,
                                                 num_mel_banks=self.num_mel_banks, num_mfccs=self.num_mfccs, num_target_classes=self.num_target_classes, model_optimizer=self.optimizer, model_loss_function=self.loss_function, model_metrics=self.metrics, model_width_dense_layer=self.width_dense_layer, seed=self.seed)
            else:
                data_model = datamodel.DataModel()
                data_model.alternate_initialize_data_model(previous_data, data_configuration, model_configuration, self.search_space, self.num_target_classes,
                                                           self.optimizer, self.model_loss_function, model_metrics=self.metrics, model_width_dense_layer=self.width_dense_layer, seed=self.seed)
                data_model.num_normal_samples = len(
                    self.dataset_loader.base_normal_audio)
                data_model.num_anomalous_samples = len(
                    self.dataset_loader.base_anomalous_audio)

            # Some data and model configurations are infeasible. In this case the model created in the data model will be None.
            # If we create an infeasible datamodel we simply skip to proposing the next model
            if data_model.model == None:
                print("Infeasible model generated. Skipping to next configuration...")
                continue

            print("Evaluating performance of data and model")
            # Evaluate performance of data and model
            data_model.evaluate_data_model(self.num_epochs, self.batch_size)

            print(
                f"Model{model_number} metrics:\nAccuracy: {data_model.accuracy}\nPrecision: {data_model.precision}\nRecall: {data_model.recall}\nModel Size (bytes): {data_model.model_size}")

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
            pareto_optimal_models)
        return pareto_optimal_models

    def _save_to_csv(self, csv_log_name, model_number, data_model):
        with open(csv_log_name, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([model_number, self.search_space.data_decode(
                data_model.data_configuration), self.search_space.model_decode(data_model.model_configuration), data_model.accuracy, data_model.precision, data_model.recall, data_model.model_size])

    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    @staticmethod
    def _prune_non_pareto_optimal_models(iterative_pareto_optimal_models):
        iterative_pareto_optimal_models = np.array(
            iterative_pareto_optimal_models)
        is_optimal = np.ones(
            iterative_pareto_optimal_models.shape[0], dtype=bool)
        for i, model in enumerate(iterative_pareto_optimal_models):
            if is_optimal[i]:
                is_optimal[is_optimal] = np.array([x.better_data_model(
                    model) for x in iterative_pareto_optimal_models[is_optimal]])
                is_optimal[i] = True
        return list(iterative_pareto_optimal_models[is_optimal])
