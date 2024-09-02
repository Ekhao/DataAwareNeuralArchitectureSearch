# Standard Library Imports
import unittest

# Local Imports
import datamodelgenerator
import datamodel


class DataModelGeneratorTestCase(unittest.TestCase):

    def test_adding_pareto_optimal_model(self):
        data_model1 = datamodel.DataModel(None, None, None, None)
        data_model2 = datamodel.DataModel(None, None, None, None)
        data_model3 = datamodel.DataModel(None, None, None, None)

        data_model1.data_configuration = 5
        data_model1.model_configuration = [5, 8, 14]
        data_model1.accuracy = 0.60
        data_model1.precision = 0.56
        data_model1.recall = 0.82
        data_model1.model_size = 456356

        data_model2.data_configuration = 13
        data_model2.model_configuration = [14, 10]
        data_model2.accuracy = 0.61
        data_model2.precision = 0.4
        data_model2.recall = 0.99
        data_model2.model_size = 358532

        data_model3.data_configuration = 20
        data_model3.model_configuration = [5]
        data_model3.accuracy = 0.59
        data_model3.precision = 0.57
        data_model3.recall = 0.81
        data_model3.model_size = 786565

        pareto_optimal_models = [data_model1, data_model2, data_model3]

        data_model4 = datamodel.DataModel(None, None, None, None)
        data_model4.data_configuration = 2
        data_model4.model_configuration = [1, 9, 5, 2]
        data_model4.accuracy = 0.02
        data_model4.precision = 0.58
        data_model4.recall = 0.82
        data_model4.model_size = 786565
        pareto_optimal_models.append(data_model4)

        datamodelgenerator.DataModelGenerator._prune_non_pareto_optimal_models(
            pareto_optimal_models
        )

        self.assertEqual(
            pareto_optimal_models, [data_model1, data_model2, data_model3, data_model4]
        )

    def test_adding_non_pareto_optimal_model(self):
        data_model1 = datamodel.DataModel(None, None, None, None)
        data_model2 = datamodel.DataModel(None, None, None, None)
        data_model3 = datamodel.DataModel(None, None, None, None)

        data_model1.accuracy = 0.60
        data_model1.precision = 0.56
        data_model1.recall = 0.82
        data_model1.model_size = 456356

        data_model2.accuracy = 0.61
        data_model2.precision = 0.4
        data_model2.recall = 0.99
        data_model2.model_size = 358532

        data_model3.accuracy = 0.59
        data_model3.precision = 0.57
        data_model3.recall = 0.81
        data_model3.model_size = 786565

        pareto_optimal_models = [data_model1, data_model2, data_model3]

        data_model4 = datamodel.DataModel(None, None, None, None)
        data_model4.accuracy = 0.02
        data_model4.precision = 0.55
        data_model4.recall = 0.82
        data_model4.model_size = 786565
        datamodelgenerator.DataModelGenerator._prune_non_pareto_optimal_models(
            pareto_optimal_models
        )
        self.assertEqual(pareto_optimal_models, [data_model1, data_model2, data_model3])

    def test_prune_non_pareto_optimal_model(self):
        data_model1 = datamodel.DataModel(None, None, None, None)
        data_model2 = datamodel.DataModel(None, None, None, None)
        data_model3 = datamodel.DataModel(None, None, None, None)

        data_model1.accuracy = 0.60
        data_model1.precision = 0.56
        data_model1.recall = 0.82
        data_model1.model_size = 456356

        data_model2.accuracy = 0.61
        data_model2.precision = 0.4
        data_model2.recall = 0.99
        data_model2.model_size = 358532

        data_model3.accuracy = 0.59
        data_model3.precision = 0.57
        data_model3.recall = 0.81
        data_model3.model_size = 786565

        data_model4 = datamodel.DataModel(None, None, None, None)
        data_model4.accuracy = 0.02
        data_model4.precision = 0.55
        data_model4.recall = 0.82
        data_model4.model_size = 786565

        pareto_optimal_models = [data_model1, data_model2, data_model3, data_model4]

        pareto_optimal_models = (
            datamodelgenerator.DataModelGenerator._prune_non_pareto_optimal_models(
                pareto_optimal_models
            )
        )
        self.assertEqual(pareto_optimal_models, [data_model1, data_model2, data_model3])

    def test_prune_non_pareto_optimal_model_sequential(self):
        data_model1 = datamodel.DataModel(None, None, None, None)
        data_model2 = datamodel.DataModel(None, None, None, None)
        data_model3 = datamodel.DataModel(None, None, None, None)

        data_model1.accuracy = 0.60
        data_model1.precision = 0.8
        data_model1.recall = 0.82
        data_model1.model_size = 456356

        data_model2.accuracy = 0.61
        data_model2.precision = 0.4
        data_model2.recall = 0.85
        data_model2.model_size = 758532

        data_model3.accuracy = 0.62
        data_model3.precision = 0.57
        data_model3.recall = 0.86
        data_model3.model_size = 358532

        data_model4 = datamodel.DataModel(None, None, None, None)
        data_model4.accuracy = 0.60
        data_model4.precision = 0.58
        data_model4.recall = 0.82
        data_model4.model_size = 786321

        pareto_optimal_models = [data_model1, data_model2, data_model3, data_model4]

        pareto_optimal_models = (
            datamodelgenerator.DataModelGenerator._prune_non_pareto_optimal_models(
                pareto_optimal_models
            )
        )
        self.assertEqual(pareto_optimal_models, [data_model1, data_model3])
