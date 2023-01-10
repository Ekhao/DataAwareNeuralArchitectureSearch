import unittest

import randomcontroller
import searchspace


class RandomControllerTestCase(unittest.TestCase):
    def test_combination_function(self):
        search_space = searchspace.SearchSpace((
            [1, 2, 3, 4, 5, 6], ["a", "b", "c"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        random_controller = randomcontroller.RandomController(
            search_space=search_space, seed=41)
        num_combinations = len(
            random_controller.search_space.model_layer_search_space_enumerated)
        self.assertEqual(num_combinations, 18)

    def test_generate_configuration(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                               48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        random_controller = randomcontroller.RandomController(
            search_space=search_space, seed=41)
        config = random_controller.generate_configuration()
        self.assertEqual(config, (12, [7, 5, 12]))
