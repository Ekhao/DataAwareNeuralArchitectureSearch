import NAS


def test_search_space_gen():
    search_space = NAS.SearchSpace(
        ([3, 5], ["relu", "elu"]), (["Spectrogram", "MFCC"], [48000, 24000]))
    assert search_space.input_search_space == {0: ("Spectrogram", 48000), 1: (
        "Spectrogram", 24000), 2: ("MFCC", 48000), 3: ("MFCC", 24000)}
