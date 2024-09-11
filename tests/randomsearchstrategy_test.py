# Standard Library Imports
import unittest

# Local Imports
import search_strategies.randomsearchstrategy as randomsearchstrategy
import searchspace
from configuration import Configuration


class RandomSearchStrategyTestCase(unittest.TestCase):
    def test_generate_configuration(self):
        self.maxDiff = None
        search_space = searchspace.SearchSpace(
            data_search_space={
                "sample_rate": [48000, 24000, 12000, 6000, 3000, 1500, 750, 325],
                "audio_representation": ["spectrogram", "mel-spectrogram", "mfcc"],
            },
            model_search_space={
                "conv_layer": {
                    "filters": [2, 4, 8, 16, 32, 64, 128],
                    "kernel_size": [3, 5],
                    "activation": ["relu", "sigmoid"],
                }
            },
        )
        random_search_strategy = randomsearchstrategy.RandomSearchStrategy(
            search_space, 5, 41
        )
        config = random_search_strategy.generate_configuration()
        self.assertEqual(
            vars(config),
            vars(
                Configuration(
                    data_configuration={
                        "sample_rate": 750,
                        "audio_representation": "mel-spectrogram",
                    },
                    model_configuration=[
                        {
                            "activation": "sigmoid",
                            "filters": 16,
                            "kernel_size": 5,
                            "type": "conv_layer",
                        },
                        {
                            "activation": "relu",
                            "filters": 128,
                            "kernel_size": 3,
                            "type": "conv_layer",
                        },
                    ],
                )
            ),
        )
