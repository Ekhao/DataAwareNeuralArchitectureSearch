import randomcontroller
import searchspace


def test_combination_function():
    random_controller = randomcontroller.RandomController(
        search_space=searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=41)
    num_combinations = random_controller.get_number_of_search_space_combinations((
        [1, 2, 3, 4, 5, 6], ["a", "b", "c"]))
    assert num_combinations == 18


def test_generate_configuration():
    random_controller = randomcontroller.RandomController(
        search_space=searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=41)
    config = random_controller.generate_configuration()
    assert config == (12, [7, 5, 12])
