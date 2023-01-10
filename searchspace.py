import itertools


class SearchSpace:
    def __init__(self, model_layer_search_space, data_search_space) -> None:
        self.data_search_space_options = data_search_space
        self.model_layer_search_space_options = model_layer_search_space

        self.data_search_space_enumerated = self.search_space_enumerator(
            *self.data_search_space_options)
        # Maybe add dropout and output layers to the model search space
        self.model_layer_search_space_enumerated = self.search_space_enumerator(
            *self.model_layer_search_space_options)

    def search_space_enumerator(self, *iterables) -> dict:

        # Encode combinations in the search space into numbers using dictionaries
        values = list(itertools.product(*iterables))
        keys = [key for key in range(len(values))]

        return dict(zip(keys, values))

    def data_decode(self, data_number: int) -> tuple:
        return self.data_search_space_enumerated[data_number]

    def data_encode(self, data_configuration: tuple) -> int:
        return list(self.data_search_space_enumerated.keys())[list(self.data_search_space_enumerated.values()).index(data_configuration)]

    def model_layer_decode(self, model_layer_number: int) -> tuple:
        return self.model_layer_search_space_enumerated[model_layer_number]

    def model_layer_encode(self, model_layer_configuration: tuple) -> int:
        return list(self.model_layer_search_space_enumerated.keys())[list(self.model_layer_search_space_enumerated.values()).index(model_layer_configuration)]

    def model_decode(self, model_layer_number_sequence: list[int]) -> list[tuple]:
        decoded_sequence = []
        for number in model_layer_number_sequence:
            decoded_sequence.append(self.model_layer_decode(number))
        return decoded_sequence

    def model_encode(self, model_layer_configuration_sequence: list[int]) -> list[tuple]:
        encoded_sequence = []
        for configuration in model_layer_configuration_sequence:
            encoded_sequence.append(self.model_layer_encode(configuration))
        return encoded_sequence
