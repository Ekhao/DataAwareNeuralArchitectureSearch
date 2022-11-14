import searchspace


def test_search_space_gen():
    search_space = searchspace.SearchSpace(
        ([3, 5, 7], ["relu"]), (["Spectrogram", "MFCC"], [48000, 24000]))
    assert search_space.input_search_space == {0: ("Spectrogram", 48000), 1: (
        "Spectrogram", 24000), 2: ("MFCC", 48000), 3: ("MFCC", 24000)} and search_space.model_layer_search_space == {
        0: (3, "relu"), 1: (5, "relu"), 2: (7, "relu")}
