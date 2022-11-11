import evolutionarycontroller


def test_init():
    evolutionary_controller = evolutionarycontroller.EvolutionaryController(
        seed=32, trivial_initialization=True, population_size=2)
    assert evolutionary_controller.unevaluated_population.get() == (
        2, [6]) and evolutionary_controller.unevaluated_population.get() == (4, [9])


def test_generate_configuration():
    evolutionary_controller = evolutionarycontroller.EvolutionaryController(
        seed=32, trivial_initialization=True, population_size=2)
    input_model = evolutionary_controller.generate_configuration()
    assert input_model == (2, [6])
