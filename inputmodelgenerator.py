import numpy as np
import csv
import datetime


import inputmodel
import datasetloader
import constants


class InputModelGenerator:
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

    def run_input_nas(self, num_of_models):
        pareto_optimal_models = []
        input = None
        csv_log_name = f"inputmodel_logs/{datetime.datetime.now().isoformat()}.csv"
        with open(csv_log_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Model Number", "Input Configuration",
                            "Model Configuration", "Accuracy", "Precision", "Recall", "Model Size"])
        for model_number in range(num_of_models):
            # Print that we are now running a new sample
            print("-"*100)
            print(f"Starting model number {model_number}")

            # Get configuration from controller
            print("Generating model configuration...")
            input_configuration, model_configuration = self.controller.generate_configuration()

            print(
                f"Input configuration: {self.search_space.input_decode(input_configuration)}\nModel configuration: {self.search_space.model_decode(model_configuration)}")

            print("Creating input and model from configuration...")
            # Create input and model from configuration
            if input == None:
                input_model = inputmodel.InputModel()
                input_model.initialize_input_model(input_configuration=input_configuration, model_configuration=model_configuration, search_space=self.search_space, dataset_loader=self.dataset_loader, frame_size=self.frame_size, hop_length=self.hop_length,
                                                   num_mel_banks=self.num_mel_banks, num_mfccs=self.num_mfccs, num_target_classes=self.num_target_classes, model_optimizer=self.optimizer, model_loss_function=self.loss_function, model_metrics=self.metrics, model_width_dense_layer=self.width_dense_layer, seed=self.seed)

                input = input_model.input
            else:
                input_model = inputmodel.InputModel()
                input_model.input = input
                input_model.model = input_model.create_model(
                    model_configuration, self.search_space, input[0][0].shape, self.num_target_classes, self.optimizer, self.loss_function, self.metrics, self.width_dense_layer)
                input_model.seed = self.seed
                input_model.input_configuration = input_configuration
                input_model.model_configuration = model_configuration

            # Some input and model configurations are infeasible. In this case the model created in the input model will be None.
            # If we create an infeasible inputmodel we simply skip to proposing the next model
            if input_model.model == None:
                print("Infeasible model generated. Skipping to next configuration...")
                continue

            print("Evaluating performance of input and model")
            # Evaluate performance of input and model
            input_model.evaluate_input_model(self.num_epochs, self.batch_size)

            print(
                f"Model{model_number} metrics:\nAccuracy: {input_model.accuracy}\nPrecision: {input_model.precision}\nRecall: {input_model.recall}\nModel Size (bytes): {input_model.model_size}")

            print("Updating parameters of the controller...")
            # Update controller parameters
            self.controller.update_parameters(input_model)

            print("Freeing loaded data and model to reduce memory consumption...")
            input_model.free_input_model()

            print("Saving InputModel and metrics in logs...")
            self.__save_to_csv(csv_log_name, model_number, input_model)

            print("Saving InputModel for pareto front calculation")
            # Save the models that are pareto optimal
            pareto_optimal_models.append(input_model)

        pareto_optimal_models = self.__prune_non_pareto_optimal_models(
            pareto_optimal_models)
        return pareto_optimal_models

    def __save_to_csv(self, csv_log_name, model_number, input_model):
        with open(csv_log_name, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([model_number, self.search_space.input_decode(
                input_model.input_configuration), self.search_space.model_decode(input_model.model_configuration), input_model.accuracy, input_model.precision, input_model.recall, input_model.model_size])

    # def save_pareto_optimal_models(self, current_input_model, pareto_optimal_models):
    #    new_list = pareto_optimal_models
    #    dominated = False
    #    for previous_input_model in pareto_optimal_models:
    #        if previous_input_model.better_input_model(current_input_model):
    #            dominated = True
    #            break
    #    if not dominated:
    #        new_list.append(current_input_model)
    #    return new_list

    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    def __prune_non_pareto_optimal_models(self, iterative_pareto_optimal_models):
        iterative_pareto_optimal_models = np.array(
            iterative_pareto_optimal_models)
        is_optimal = np.ones(
            iterative_pareto_optimal_models.shape[0], dtype=bool)
        for i, model in enumerate(iterative_pareto_optimal_models):
            if is_optimal[i]:
                is_optimal[is_optimal] = np.array([x.better_input_model(
                    model) for x in iterative_pareto_optimal_models[is_optimal]])
                is_optimal[i] = True
        return list(iterative_pareto_optimal_models[is_optimal])
