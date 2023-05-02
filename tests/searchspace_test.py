# Standard Library Imports
import unittest

# Local Imports
import searchspace


class SearchSpaceTestCase(unittest.TestCase):

    def test_search_space_gen(self):
        search_space = searchspace.SearchSpace(
            [["Spectrogram", "MFCC"], [48000, 24000]], [[3, 5, 7], ["relu"]])
        self.assertEqual(search_space.data_granularity_search_space, [[
                         "Spectrogram", "MFCC"], [48000, 24000]])
        self.assertEqual(search_space.model_layer_search_space, [
                         [3, 5, 7], ["relu"]])
