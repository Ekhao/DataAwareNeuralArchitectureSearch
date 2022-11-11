import randomcontroller


def test_combination_function():
    random_controller = randomcontroller.RandomController(seed=41)
    num_combinations = random_controller.get_number_of_search_space_combinations((
        [1, 2, 3, 4, 5, 6], ["a", "b", "c"]))
    assert num_combinations == 18


def test_generate_configuration():
    random_controller = randomcontroller.RandomController(seed=41)
    config = random_controller.generate_configuration()
    assert config == (12, [7, 5, 12])
