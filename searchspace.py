# This class implements the search space of the Data Aware Neural Architecture Search.

# Standard Library Imports
from typing import Any


class SearchSpace:
    def __init__(self, data_granularity_search_space: list[Any], model_layer_search_space: list[Any],) -> None:
        self.data_granularity_search_space = data_granularity_search_space
        self.model_layer_search_space = model_layer_search_space
