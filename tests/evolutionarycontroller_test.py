import evolutionarycontroller
import searchspace


def test_init():
    evolutionary_controller = evolutionarycontroller.EvolutionaryController(
        search_space=searchspace.SearchSpace(([2, 4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
    assert evolutionary_controller.unevaluated_population.get() == (
        2, [6]) and evolutionary_controller.unevaluated_population.get() == (4, [9])


def test_generate_configuration():
    evolutionary_controller = evolutionarycontroller.EvolutionaryController(
        search_space=searchspace.SearchSpace(([2,   4, 8, 16, 32, 64, 128], [3, 5], ["relu", "sigmoid"]), ([48000, 24000, 12000, 6000, 3000, 1500, 750, 325], ["spectrogram", "mel-spectrogram", "mfcc"])), seed=32, trivial_initialization=True, population_size=2)
    input_model = evolutionary_controller.generate_configuration()
    assert input_model == (2, [6])
