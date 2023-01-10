# This python file contains the main function of the Data Aware Neural Architecture Search. It is responsible for parsing the command line arguments, initializing the search space, loading the data and initializing the controller. It then calls the controller to perform the search and returns the pareto front.

# Standard Library Imports
import argparse

# Third Party Imports
import tensorflow as tf

# Local Imports
import searchspace
import datamodelgenerator
import datasetloader
import constants
import randomcontroller
import evolutionarycontroller


def main():
    argparser = argparse.ArgumentParser(
        description="A simple implementation of Data Aware NAS. Note the constants.py file that can be used to control additional settings.")
    argparser.add_argument("-n", "--path_normal_files",
                           help="The filepath to the directory containing normal files. Can also be set in configuration.py.", default=None)
    argparser.add_argument("-a", "--path_anomalous_files",
                           help="The filepath to the directory containing anoamlous files. Can also be set in configuration.py.", default=None)
    argparser.add_argument("-ns", "--path_noise_files",
                           help="The filepath to the directory containing noise files. Can also be set in configuration.py.", default=None)
    argparser.add_argument("-cns", "--case_noise_files",
                           help="The case to load noise files from. Can also be set in configuration.py.", default="case1")
    argparser.add_argument(
        "-c", "--controller", help="The controller to use to direct the search. Also known as search strategy. Supported options are \"evolution\" and \"random\".", choices=["evolution", "random"], default="evolution")
    argparser.add_argument("-i", "--initialization",
                           help="How to initialize the first generation of models for the \"evolution\" controller. Supported options are \"trivial\" and \"random\". Does nothing if paired with the \"random\" controller.", choices=["trivial", "random"], default="trivial")
    argparser.add_argument(
        "-nm", "--num_models", help="The number of models to evaluate before returning the pareto front.", type=int, default=10)
    argparser.add_argument(
        "-s", "--seed", help="A seed to be given to the random number generators of the program.", type=int, default=None)
    args = argparser.parse_args()

    # The following block of code enables memory growth for the GPU during runtime.
    # It is suspected that this helps avoiding out of memory errors.
    # https://www.tensorflow.org/guide/gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print("Initializing search space...")
    search_space = searchspace.SearchSpace(
        constants.MODEL_LAYER_SEARCH_SPACE, constants.DATA_SEARCH_SPACE)

    print("Loading dataset files from persistent storage...")
    if args.path_normal_files == None:
        args.path_normal_files = constants.PATH_TO_NORMAL_FILES
    if args.path_anomalous_files == None:
        args.path_anomalous_files = constants.PATH_TO_ANOMALOUS_FILES
    if args.path_noise_files == None:
        args.path_noise_files = constants.PATH_TO_NOISE_FILES

    dataset_loader = datasetloader.DatasetLoader(args.path_normal_files, args.path_anomalous_files, args.path_noise_files, "case1",
                                                 constants.NUMBER_OF_NORMAL_FILES_TO_USE, constants.NUMBER_OF_ANOMALOUS_FILES_TO_USE, constants.DATASET_CHANNEL_TO_USE)

    print("Initializing controller...")
    if args.controller == "evolution":
        controller = evolutionarycontroller.EvolutionaryController(
            search_space, args.seed)
        controller.initialize_controller(
            trivial_initialization=args.initialization == "trivial")
    else:
        controller = randomcontroller.RandomController(search_space, args.seed)

    data_model_generator = datamodelgenerator.DataModelGenerator(
        constants.NUM_OUTPUT_CLASSES, constants.LOSS_FUNCTION, controller=controller, dataset_loader=dataset_loader)
    pareto_front = data_model_generator.run_data_nas(
        num_of_models=args.num_models)

    print("Models on pareto front: ")
    for data_model in pareto_front:
        print(search_space.data_decode(
            data_model.data_configuration))
        print(search_space.model_decode(
            data_model.model_configuration))
        print(f"Accuracy: {data_model.accuracy}, Precision: {data_model.precision}, Recall: {data_model.recall}, Model Size (in bytes): {data_model.model_size}.")
        print("-"*200)


if __name__ == "__main__":
    main()
