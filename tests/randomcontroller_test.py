import unittest

import randomcontroller
import searchspace


class RandomControllerTestCase(unittest.TestCase):
    def test_combination_function(self):
        random_controller = randomcontroller.RandomController(
            search_space=searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=41)
        num_combinations = random_controller.get_number_of_search_space_combinations((
            [1, 2, 3, 4, 5, 6], ["a", "b", "c"]))
        self.assertEqual(num_combinations, 18)

    def test_generate_configuration(self):
        random_controller = randomcontroller.RandomController(
            search_space=searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=41)
        config = random_controller.generate_configuration()
        self.assertEqual(config, (12, [7, 5, 12]))
