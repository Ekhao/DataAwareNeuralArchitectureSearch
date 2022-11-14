import tensorflow as tf

import randomcontroller
import searchspace
import inputmodel
import datasetloader
import constants


class InputModelGenerator:
    def __init__(self, num_target_classes, loss_function, controller=randomcontroller.RandomController(searchspace.SearchSpace(model_layer_search_space=constants.MODEL_LAYER_SEARCH_SPACE, input_search_space=constants.INPUT_SEARCH_SPACE)), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), dropout_rate=0.5, metrics=["accuracy"], num_epochs=constants.NUM_EPOCHS, batch_size=constants.BATCH_SIZE, number_of_normal_files=constants.NUMBER_OF_NORMAL_FILES_TO_USE, number_of_anomalous_files=constants.NUMBER_OF_ANOMALOUS_FILES_TO_USE):
        self.num_target_classes = num_target_classes
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.metrics = metrics
        self.controller = controller
        self.search_space = controller.search_space
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.number_of_normal_files = number_of_normal_files
        self.number_of_anomalous_files = number_of_anomalous_files
        self.pareto_optimal_models = []

    def run_input_nas(self, num_of_samples):
        pareto_optimal_list = []
        for sample in range(num_of_samples):
            # Get configuration from controller
            input_configuration, model_layer_configuration = self.controller.generate_configuration()

            # Create input and model from configuration
            dataset, model = self.create_input_model(
                input_configuration, model_layer_configuration)

            # Create a configuration class from the generated configuration
            input_model = inputmodel.InputModel(dataset=dataset, model=model)

            # Evaluate performance of model
            input_model.evaluate_input_model(input_model)

            # Update controller parameters
            self.controller.update_parameters(input_model)

            print(
                f"Accuracy: {input_model.accuracy}\nPrecision: {input_model.precision}\nRecall: {input_model.recall}")

            # Save the models that are pareto optimal
            self.pareto_optimal_models = self.save_pareto_optimal_models(
                input_model)

    def create_input_model(self, input_number: int, model_layer_numbers: list[int]) -> tf.keras.Model:
        dataset = self.create_input(input_number)
        # We need to subscript the dataset two times.
        # First subscript is to choose the normal files (here we could also chose the abnormal files - doesnt matter)
        # Second subscript is to choose the first entry (all entries should have the same shape)
        model = self.create_model(model_layer_numbers, dataset[0][0].shape)
        return inputmodel.InputModel(input, model)

    def create_input(self, input_number: int, number_of_normal_files: int = constants.NUMBER_OF_NORMAL_FILES_TO_USE) -> tuple:
        input_config = self.search_space.input_decode(input_number)

        dataset_loader = datasetloader.DatasetLoader()
        normal_preprocessed, abnormal_preprocessed = dataset_loader.load_dataset(
            constants.PATH_TO_NORMAL_FILES, constants.PATH_TO_ANOMALOUS_FILES, input_config[0], input_config[1], self.number_of_normal_files, self.number_of_anomalous_files)

        return dataset_loader.supervised_dataset(normal_preprocessed, abnormal_preprocessed)

    def create_model(self, sequence: list[int], input_shape=tuple) -> tf.keras.Model:
        layer_configs = self.search_space.model_decode(sequence)

        model = tf.keras.Sequential()

        # For the first layer we need to define the input shape
        model.add(tf.keras.layers.Conv2D(
            filters=layer_configs[0][0], kernel_size=layer_configs[0][1], activation=layer_configs[0][2], input_shape=input_shape))

        for layer_config in layer_configs[1:]:
            model.add(tf.keras.layers.Conv2D(
                filters=layer_config[0], kernel_size=layer_config[1], activation=layer_config[2]))

        # The standard convolutional model has dense layers at its end for classification - let us make the same assumption TODO: should be a part of search space
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(
            constants.WIDTH_OF_DENSE_LAYER, activation=tf.keras.activations.relu))

        # Output layer
        model.add(tf.keras.layers.Dense(
            self.num_target_classes, activation=tf.keras.activations.softmax))

        model.compile(optimizer=self.optimizer,
                      loss=self.loss_function, metrics=self.metrics)

        return model

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
