import itertools


class SearchSpace:
    def __init__(self, model_layer_search_space, input_search_space) -> None:
        self.input_search_space_options = input_search_space
        self.model_layer_search_space_options = model_layer_search_space

        self.input_search_space_enumerated = self.search_space_enumerator(
            *input_search_space)
        # Maybe add dropout and output layers to the model search space
        self.model_layer_search_space_enumerated = self.search_space_enumerator(
            *model_layer_search_space)

    def search_space_enumerator(self, *iterables) -> dict:

        # Encode combinations in the search space into numbers using dictionaries
        values = list(itertools.product(*iterables))
        keys = [key for key in range(len(values))]

        return dict(zip(keys, values))

    def input_decode(self, input_number: int) -> tuple:
        return self.input_search_space_enumerated[input_number]

    def input_encode(self, input_configuration: tuple) -> int:
        return list(self.input_search_space_enumerated.keys())[list(self.input_search_space_enumerated.values()).index(input_configuration)]

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
