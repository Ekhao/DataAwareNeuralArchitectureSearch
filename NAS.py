# We start with a simple and generic method to encode the search space. In the beginning we also only look for convolutional models as a proof of concept. (Let us stay in on the topic of audio processing) - I believe that edge impulse does the same

import itertools
from pickletools import optimize
import tensorflow as tf


class SearchSpace:
    def __init__(self, model_layer_search_space, input_search_space) -> None:
        self.input_search_space = self.search_space_gen(*input_search_space)
        self.model_layer_search_space = self.search_space_gen(*model_layer_search_space) # Maybe add dropout and output layers to the model search space


    def search_space_gen(self, *iterables) -> dict:

        #Encode combinations in the search space into numbers using dictionaries
        values = list(itertools.product(*iterables))
        keys = [key for key in range(len(values))]

        return dict(zip(keys, values))
    
    def input_decode(self, number: int) -> tuple:
        return self.input_search_space[number]

    def model_decode(self, sequence: list[int]) -> list[tuple]:
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(self.model_layer_search_space[key])
        
        return decoded_sequence

class ModelGenerator:
    def __init__(self, target_classes, loss_function, optimizer = tf.keras.optimizers.Adam,
                 learning_rate = 0.001, dropout_rate = 0.5, metrics = ["accuracy"]): #TODO: Maybe add decay, momentum and accuracy
        self.target_classes = target_classes
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.metrics = metrics
        self.search_space = SearchSpace(model_layer_search_space=([8,16,32,64,128],[3,5],["relu", "sigmoid"]),input_search_space=([48000, 24000, 12000, 6000, 3000, 1500, 750, 325],["spectrogram", "mfe", "mfcc"],[32, 16, 8, 4, 2])) #TODO: This shouldn't be hard coded here.
    
    def create_model(self, sequence: list[int], input_shape = tuple) -> tf.keras.Model:
        layer_configs = self.search_space.model_decode(sequence)

        model = tf.keras.Sequential()

        # For the first layer we need to define the input shape
        model.add(tf.keras.layers.Conv2D(filters = layer_configs[0][0], kernel_size = layer_configs[0][1], activation = layer_configs[0][2], input_shape=input_shape))

        for layer_config in layer_configs[1:]:
            model.add(tf.keras.layers.Conv2D(filters = layer_config[0], kernel_size = layer_config[1], activation = layer_config[2]))
        
        # The standard convolutional model has dense layers at its end for classification - let us make the same assumption TODO: should be a part of search space
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(128, activation="relu"))

        # Output layer
        model.add(tf.keras.layers.Dense(self.target_classes))

        model.compile(optimizer=self.optimizer(learning_rate=self.learning_rate), loss=self.loss_function, metrics=self.metrics)

        return model


model_generator = ModelGenerator(2, tf.keras.losses.CategoricalCrossentropy)
model = model_generator.create_model([6,8,2], (32,32,3))
print(model.summary())