# Standard Library Imports
import unittest

# Local Imports
import randomsearchstrategy
import searchspace


class RandomSearchStrategyTestCase(unittest.TestCase):
    def test_generate_configuration(self):
        search_space = searchspace.SearchSpace(
            [
                [48000, 24000, 12000, 6000, 3000, 1500, 750, 325],
                ["spectrogram", "mel-spectrogram", "mfcc"],
            ],
            [[2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]],
        )
        random_search_strategy = randomsearchstrategy.RandomSearchStrategy(
            search_space, 5, 41
        )
        config = random_search_strategy.generate_configuration()
        self.assertEqual(
            config, ((750, "mel-spectrogram"), [(4, 5, "sigmoid"), (32, 5, "sigmoid")])
        )
