import tensorflow as tf

import randomcontroller
import searchspace
import inputmodel
import datasetloader
import constants


class InputModelGenerator:
    def __init__(self, num_target_classes, loss_function, controller=randomcontroller.RandomController(searchspace.SearchSpace(model_layer_search_space=constants.MODEL_LAYER_SEARCH_SPACE, input_search_space=constants.INPUT_SEARCH_SPACE)), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), dropout_rate=0.5, metrics=["accuracy"], width_dense_layer=constants.WIDTH_OF_DENSE_LAYER, dataset_loader=datasetloader.DatasetLoader(constants.PATH_TO_NORMAL_FILES, constants.PATH_TO_ANOMALOUS_FILES, constants.NUMBER_OF_NORMAL_FILES_TO_USE, constants.NUMBER_OF_ANOMALOUS_FILES_TO_USE, constants.DATASET_CHANNEL_TO_USE), num_epochs=constants.NUM_EPOCHS, batch_size=constants.BATCH_SIZE, number_of_normal_files=constants.NUMBER_OF_NORMAL_FILES_TO_USE, number_of_anomalous_files=constants.NUMBER_OF_ANOMALOUS_FILES_TO_USE, path_to_normal_files=constants.PATH_TO_NORMAL_FILES, path_to_anomalous_files=constants.PATH_TO_ANOMALOUS_FILES):
        self.num_target_classes = num_target_classes
        self.loss_function = loss_function
        self.controller = controller
        self.search_space = controller.search_space
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.metrics = metrics
        self.width_dense_layer = width_dense_layer
        self.dataset_loader = dataset_loader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.number_of_normal_files = number_of_normal_files
        self.number_of_anomalous_files = number_of_anomalous_files
        self.path_to_normal_files = path_to_normal_files
        self.path_to_anomalous_files = path_to_anomalous_files
        self.pareto_optimal_models = []

    def run_input_nas(self, num_of_samples):
        pareto_optimal_list = []
        for sample in range(num_of_samples):
            # Get configuration from controller
            input_configuration, model_configuration = self.controller.generate_configuration()

            # Create input and model from configuration
            input_model = inputmodel.InputModel(
                input_configuration=input_configuration, model_configuration=model_configuration, search_space=self.search_space, dataset_loader=self.dataset_loader, num_target_classes=self.num_target_classes, model_optimizer=self.optimizer, model_loss_function=self.loss_function, model_metrics=self.metrics, model_width_dense_layer=self.width_dense_layer)

            # Evaluate performance of model
            input_model.evaluate_input_model(self.num_epochs, self.batch_size)

            # Update controller parameters
            self.controller.update_parameters(input_model)

            print(
                f"Accuracy: {input_model.accuracy}\nPrecision: {input_model.precision}\nRecall: {input_model.recall}")

            # Save the models that are pareto optimal
            self.pareto_optimal_models = self.save_pareto_optimal_models(
                input_model)

    def save_pareto_optimal_models(self, current_input_model):
        new_list = self.pareto_optimal_models
        pareto_dominated = False
        for previous_input_model in self.pareto_optimal_models:
            if previous_input_model.better_input_model(current_input_model):
                dominated = True
                break
        if not dominated:
            new_list.append(current_input_model)
        return new_list
