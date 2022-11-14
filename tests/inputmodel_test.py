import inputmodel


def test_better_configuration():
    input_model1 = inputmodel.InputModel(None, None)
    input_model1.accuracy = 0.98
    input_model1.precision = 0.7
    input_model1.recall = 0.99
    input_model2 = inputmodel.InputModel(None, None)
    input_model2.accuracy = 0.9
    input_model2.precision = 0.69
    input_model2.recall = 0.89
    assert input_model1.better_input_model(input_model2)


def test_not_better_configuration():
    input_model1 = inputmodel.InputModel(None, None)
    input_model1.accuracy = 0.60
    input_model1.precision = 0.56
    input_model1.recall = 0.82
    input_model2 = inputmodel.InputModel(None, None)
    input_model2.accuracy = 0.02
    input_model2.precision = 0.55
    input_model2.recall = 0.82
    assert not input_model2.better_input_model(input_model1)
