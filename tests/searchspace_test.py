import searchspace


def test_search_space_gen():
    search_space = searchspace.SearchSpace(
        ([3, 5, 7], ["relu"]), (["Spectrogram", "MFCC"], [48000, 24000]))
    assert search_space.input_search_space == {0: ("Spectrogram", 48000), 1: (
        "Spectrogram", 24000), 2: ("MFCC", 48000), 3: ("MFCC", 24000)} and search_space.model_layer_search_space == {
        0: (3, "relu"), 1: (5, "relu"), 2: (7, "relu")}


def test_input_decode():
    search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                           48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
    decoded_input = search_space.input_decode(7)
    assert decoded_input == (12000, "mel-spectrogram")


def test_input_encode():
    search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                           48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
    encoded_input = search_space.input_encode((12000, "mel-spectrogram"))
    assert encoded_input == 7


def test_model_layer_decode():
    search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                           48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
    decoded_model_layer = search_space.model_layer_decode(15)
    assert decoded_model_layer == (16, 5, "sigmoid")


def test_model_layer_encode():
    search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                           48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
    encoded_model_layer = search_space.model_layer_encode((16, 5, "sigmoid"))
    assert encoded_model_layer == 15


def test_model_decode():
    search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                           48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
    decoded_model = search_space.model_decode([27, 15, 7, 3])
    assert decoded_model == [
        (128, 5, "sigmoid"), (16, 5, "sigmoid"), (4, 5, "sigmoid"), (2, 5, "sigmoid")]


def test_model_encode():
    search_space = searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([
                                           48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"]))
    encoded_model = search_space.model_encode([
        (128, 5, "sigmoid"), (16, 5, "sigmoid"), (4, 5, "sigmoid"), (2, 5, "sigmoid")])
    assert encoded_model == [27, 15, 7, 3]
