# In this simple implementation the NAS search space only consists of convolutional models.

import inputmodelgenerator
import datasetloader
import constants
import randomcontroller
import evolutionarycontroller

import argparse
import tensorflow as tf


def main():
    argparser = argparse.ArgumentParser(
        description="A simple implementation of Data Aware NAS. Note the constants.py file that can be used to control additional settings.")
    argparser.add_argument("-n", "--path_normal_files",
                           help="The filepath to the directory containing normal files. Can also be set in configuration.py.", default=None)
    argparser.add_argument("-a", "--path_anomalous_files",
                           help="The filepath to the directory containing anoamlous files. Can also be set in configuration.py.", default=None)
    argparser.add_argument(
        "-c", "--controller", help="The controller to use to direct the search. Also known as search strategy. Supported options are \"evolution\" and \"random\".", choices=["evolution", "random"], default="evolution")
    argparser.add_argument("-i", "--initialization",
                           help="How to initialize the first generation of models for the \"evolution\" controller. Supported options are \"trivial\" and \"random\". Does nothing if paired with the \"random\" controller.", choices=["trivial", "random"], default="trivial")
    argparser.add_argument(
        "-nm", "--num_models", help="The number of models to evaluate before returning the pareto front.", type=int, default=10)
    argparser.add_argument(
        "-s", "--seed", help="A seed to be given to the random number generators of the program.", type=int, default=None)
    args = argparser.parse_args()

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
    constants.SEARCH_SPACE.initialize_search_space()

    print("Loading dataset files from persistent storage...")
    if args.path_normal_files == None:
        args.path_normal_files = constants.PATH_TO_NORMAL_FILES
    if args.path_anomalous_files == None:
        args.path_anomalous_files = constants.PATH_TO_ANOMALOUS_FILES

    dataset_loader = datasetloader.DatasetLoader(args.path_normal_files, args.path_normal_files,
                                                 constants.NUMBER_OF_NORMAL_FILES_TO_USE, constants.NUMBER_OF_ANOMALOUS_FILES_TO_USE, constants.DATASET_CHANNEL_TO_USE)

    print("Initializing controller...")
    if args.controller == "evolution":
        controller = evolutionarycontroller.EvolutionaryController(
            search_space=constants.SEARCH_SPACE, seed=args.seed)
        controller.initialize_controller(
            trivial_initialization=args.initialization == "trivial")
    else:
        controller = randomcontroller.RandomController(
            constants.SEARCH_SPACE, seed=args.seed)

    input_model_generator = inputmodelgenerator.InputModelGenerator(
        constants.NUM_OUTPUT_CLASSES, constants.LOSS_FUNCTION, controller=controller, dataset_loader=dataset_loader)
    input_model_generator.run_input_nas(num_of_models=args.num_models)


if __name__ == "__main__":
    main()
