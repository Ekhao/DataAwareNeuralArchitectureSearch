import itertools


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
