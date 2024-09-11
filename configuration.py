# A configuration represents a choice of data and model parameters that can be used to create an instance of a datamodel.


class Configuration:
    def __init__(
        self,
        data_configuration: dict,
        model_configuration: list[dict],
    ):
        self.data_configuration: dict = data_configuration
        self.model_configuration: list[dict] = model_configuration
