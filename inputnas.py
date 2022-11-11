# We start with a simple and generic method to encode the search space. In the beginning we also only look for convolutional models as a proof of concept. (Let us stay in on the topic of audio processing) - I believe that edge impulse does the same

import itertools
import numpy as np
import tensorflow as tf
import datasetloader
import randomcontroller
from constants import *
import sklearn.metrics


def main():
    input_model_generator = InputModelGenerator(
        NUM_OUTPUT_CLASSES, LOSS_FUNCTION, controller=randomcontroller.RandomController(seed=42))
    input_model_generator.run_input_nas(num_of_samples=2)


class SearchSpace:
    def __init__(self, model_layer_search_space, input_search_space) -> None:
        self.input_search_space = self.search_space_gen(*input_search_space)
        # Maybe add dropout and output layers to the model search space
        self.model_layer_search_space = self.search_space_gen(
            *model_layer_search_space)

    def search_space_gen(self, *iterables) -> dict:

        # Encode combinations in the search space into numbers using dictionaries
        values = list(itertools.product(*iterables))
        keys = [key for key in range(len(values))]

        return dict(zip(keys, values))

    def input_decode(self, input_number: int) -> tuple:
        return self.input_search_space[input_number]

    def model_decode(self, sequence: list[int]) -> list[tuple]:
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(self.model_layer_search_space[key])

        return decoded_sequence


class InputModel:
    def __init__(self, input, model) -> None:
        self.model = model
        self.input = input

    def evaluate_input_model(self, generator):
        # Maybe introduce validation data to stop training if the validation error starts increasing.
        X_train, X_test, y_train, y_test = self.dataset

        self.model.fit(x=X_train, y=y_train, epochs=generator.num_epochs,
                       batch_size=generator.batch_size)

        y_hat = self.model.predict(X_test, batch_size=generator.batch_size)

        # Transform the output one hot incoding into class indices
        y_hat = tf.math.top_k(input=y_hat, k=1).indices.numpy()[:, 0]

        # We would like to get accuracy, precision, recall and model size.
        self.accuracy = sklearn.metrics.accuracy_score(
            y_true=y_test, y_pred=y_hat)
        self.precision = sklearn.metrics.precision_score(
            y_true=y_test, y_pred=y_hat)
        self.recall = sklearn.metrics.recall_score(
            y_true=y_test, y_pred=y_hat)

    def better_accuracy(self, other_configuration):
        return self.accuracy > other_configuration.accuracy

    def better_precision(self, other_configuration):
        return self.precision > other_configuration.precision

    def better_recall(self, other_configuration):
        return self.recall > other_configuration.recall

    def better_input_model(self, other_configuration):
        return np.any(np.array([self.better_accuracy(other_configuration), self.better_precision(
            other_configuration), self.better_recall(other_configuration)]))


class InputModelGenerator:
    def __init__(self, num_target_classes, loss_function, search_space=SearchSpace(
            model_layer_search_space=MODEL_LAYER_SEARCH_SPACE, input_search_space=INPUT_SEARCH_SPACE),  controller=randomcontroller.RandomController(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), dropout_rate=0.5, metrics=["accuracy"], num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, number_of_normal_files=NUMBER_OF_NORMAL_FILES_TO_USE, number_of_anomalous_files=NUMBER_OF_ANOMALOUS_FILES_TO_USE):
        self.num_target_classes = num_target_classes
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.metrics = metrics
        self.search_space = search_space
        self.controller = controller
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
            input_model = InputModel(dataset=dataset, model=model)

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
        return InputModel(input, model)

    def create_input(self, input_number: int, number_of_normal_files: int = NUMBER_OF_NORMAL_FILES_TO_USE) -> tuple:
        input_config = self.search_space.input_decode(input_number)

        dataset_loader = datasetloader.DatasetLoader()
        normal_preprocessed, abnormal_preprocessed = dataset_loader.load_dataset(
            PATH_TO_NORMAL_FILES, PATH_TO_ANOMALOUS_FILES, input_config[0], input_config[1], self.number_of_normal_files, self.number_of_anomalous_files)

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
            WIDTH_OF_DENSE_LAYER, activation=tf.keras.activations.relu))

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


if __name__ == "__main__":
    main()
