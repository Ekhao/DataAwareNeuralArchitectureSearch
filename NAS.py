# Should we start with only searching for CNN architectures? We can use the binary encoding learnt from reading some papers in that case.
# Or the more general approach from paperspace?

import itertools
from random import sample

class SearchSpace():
    def __init__(self, model_layer_search_space, input_search_space) -> None:
        self.input_search_space = self.search_space_gen(*input_search_space)
        self.model_layer_search_space = self.search_space_gen(*model_layer_search_space)


    def search_space_gen(self, *iterables) -> dict:

        #Encode combinations in the search space into numbers using dictionaries
        values = list(itertools.product(*iterables))
        keys = [key for key in range(len(values))]

        return dict(zip(keys, values))

search_space = SearchSpace(model_layer_search_space=([8,16,32,64,128],["relu", "tanh", "sigmoid", "elu"]),input_search_space=([48000, 24000, 12000, 6000, 3000, 1500, 750, 325],["spectrogram", "mfe", "mfcc"],[32, 16, 8, 4, 2]))
print(search_space.model_layer_search_space, search_space.input_search_space)




# Create sequences of configurations

# Convert these into actual machine learning models