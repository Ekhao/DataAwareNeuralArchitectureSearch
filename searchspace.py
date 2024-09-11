# This class implements the search space of the Data Aware Neural Architecture Search.

# Standard Library Imports
from typing import Any


class SearchSpace:
    def __init__(
        self,
        data_search_space: dict[list],
        model_search_space: dict[dict[list]],
    ) -> None:
        self.data_search_space = data_search_space
        self.model_search_space = model_search_space
