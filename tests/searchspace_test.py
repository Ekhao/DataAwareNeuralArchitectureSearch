import unittest

import searchspace


class SearchSpaceTestCase(unittest.TestCase):

    def test_search_space_gen(self):
        search_space = searchspace.SearchSpace(
            ([3, 5, 7], ["relu"]), (["Spectrogram", "MFCC"], [48000, 24000]))
        search_space.initialize_search_space()
        self.assertEqual(search_space.input_search_space_enumerated, {0: ("Spectrogram", 48000), 1: (
            "Spectrogram", 24000), 2: ("MFCC", 48000), 3: ("MFCC", 24000)})
        self.assertEqual(search_space.model_layer_search_space_enumerated, {
            0: (3, "relu"), 1: (5, "relu"), 2: (7, "relu")})

    def test_input_decode(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        decoded_input = search_space.input_decode(7)
        self.assertEqual(decoded_input, (12000, "mel-spectrogram"))

    def test_input_encode(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        encoded_input = search_space.input_encode((12000, "mel-spectrogram"))
        self.assertEqual(encoded_input, 7)

    def test_model_layer_decode(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        decoded_model_layer = search_space.model_layer_decode(15)
        self.assertEqual(decoded_model_layer, (16, 5, "sigmoid"))

    def test_model_layer_encode(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        encoded_model_layer = search_space.model_layer_encode(
            (16, 5, "sigmoid"))
        self.assertEqual(encoded_model_layer, 15)

    def test_model_decode(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        decoded_model = search_space.model_decode([27, 15, 7, 3])
        self.assertEqual(decoded_model, [
            (128, 5, "sigmoid"), (16, 5, "sigmoid"), (4, 5, "sigmoid"), (2, 5, "sigmoid")])

    def test_model_encode(self):
        search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
            48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
        search_space.initialize_search_space()
        encoded_model = search_space.model_encode([
            (128, 5, "sigmoid"), (16, 5, "sigmoid"), (4, 5, "sigmoid"), (2, 5, "sigmoid")])
        self.assertEqual(encoded_model, [27, 15, 7, 3])
