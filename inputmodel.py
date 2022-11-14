import tensorflow as tf
import sklearn.metrics
import numpy as np


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
