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
    # Parse the command line arguments
    argparser = argparse.ArgumentParser(
        description="A simple implementation of a Data Aware NAS. Note the constants.py file also can be used to control settings. The command line arguments take precedence over the constants.py file.")

    # General Parameters
    argparser.add_argument(
        "-nm", "--num_models", help="The number of models to evaluate before returning the pareto front.", type=int)
    argparser.add_argument(
        "-s", "--seed", help="A seed to be given to the random number generators of the program.", type=int)

    # Joblib Parameters
    argparser.add_argument("-nc", "--num_cores_to_use",
                           help="The number of cpu cores to use for parallel processing. Mainly used for data loading. Can be set to -1 to use all available cores.", type=int)

    # Search Space Parameters
    argparser.add_argument("-dss", "--data_search_space",
                           help="The search space to use for data preprocessing.")
    argparser.add_argument("-mlss", "--model_layer_search_space",
                           help="The search space to use for model layers.")

    # Model Parameters
    argparser.add_argument("-o", "--optimizer",
                           help="The optimizer to use for training the models. Give as a string corresponding to the alias of a TensorFlow optimizer.")
    argparser.add_argument("-lf", "--loss_function",
                           help="The loss function to use for training the models. Give as a string corresponding to the alias of a TensorFlow loss function.")
    argparser.add_argument("-m", "--metrics",
                           help="The metrics to use for training the models. Give as a [\"...\"] formatted list of TensorFlow metric aliases.")
    argparser.add_argument("-no", "--num_output_classes",
                           help="The number of outputs classes that the created models should have.", type=int)
    argparser.add_argument("-wd", "--width_of_dense_layer",
                           help="The width of the dense layer after the convolutional layers in the model. Be aware that this argument can cause an explosion of model parameters.")

    # Dataset Parameters
    argparser.add_argument("-n", "--path_normal_files",
                           help="The filepath to the directory containing normal files.")
    argparser.add_argument("-a", "--path_anomalous_files",
                           help="The filepath to the directory containing anoamlous files.")
    argparser.add_argument("-ns", "--path_noise_files",
                           help="The filepath to the directory containing noise files.")
    argparser.add_argument("-cns", "--case_noise_files",
                           help="The case number to use for noise files.")
    argparser.add_argument("-nn", "--num_normal_files_to_use",
                           help="The number of normal files to use for training.", type=int)
    argparser.add_argument("-na", "--num_anomalous_files_to_use",
                           help="The number of anomalous files to use for training.", type=int)
    argparser.add_argument("-ch", "--dataset_channel_to_use",
                           help="The dataset channel to use for training.", type=int)
    argparser.add_argument("-sg", "--sound_gain",
                           help="The gain to apply to the sound files.", type=float)
    argparser.add_argument("-ng", "--noise_gain",
                           help="The gain to apply to the noise files.", type=float)

    # Audio Preprocessing Parameters
    argparser.add_argument("-fs", "--frame_size",
                           help="The frame size to use for preprocessing.", type=int)
    argparser.add_argument("-hl", "--hop_length",
                           help="The hop length to use for preprocessing.", type=int)
    argparser.add_argument("-nmf", "--num_mel_filters",
                           help="The number of mel filters to use for preprocessing.", type=int)
    argparser.add_argument("-nmfcc", "--num_mfccs",
                           help="The number of mfccs to use for preprocessing.", type=int)

    # Controller Parameters
    argparser.add_argument("-c", "--controller",
                           help="The controller to use to direct the search. Also known as search strategy. Supported options are \"evolution\" and \"random\".", choices=["evolution", "random"])
    argparser.add_argument("-i", "--initialization",
                           help="How to initialize the first generation of models for the \"evolution\" controller. Supported options are \"trivial\" and \"random\". Does nothing if paired with the \"random\" controller.", choices=["trivial", "random"])
    argparser.add_argument("-ml", "--max_num_layers",
                           help="The maximum number of layers to use for the models.", type=int)

    # Evaluation Parameters
    argparser.add_argument("-ne", "--num_epochs",
                           help="The number of epochs to train a model for before evaluation", type=int)
    argparser.add_argument("-bs", "--batch_size",
                           help="The batch size to use for training.", type=int)
    argparser.add_argument(
        "-msa", "--model_size_approximate_range", type=int)

    # Evolutionary Parameters
    argparser.add_argument("-ps", "--population_size",
                           help="The population size to use for the evolutionary controller. This argument is ignored if the randomcontroller is used.", type=int)
    argparser.add_argument("-ur", "--population_update_ratio",
                           help="The ratio of the population to be discarded and regenerated when the population is updated. This argument is ignored if the random controller is used.", type=float)
    argparser.add_argument("-cr", "--crossover_ratio",
                           help="The ratio of the updated population to be generated by crossover. The rest of the updated population is generated by mutations. This argument is ignored if the random controller is used.", type=float)

    args = argparser.parse_args()

    # Set options according to command line arguments and config file
    if not args.num_models:
        args.num_models = constants.NUM_MODELS
    if not args.seed:
        args.seed = constants.SEED
    if not args.num_cores_to_use:
        args.num_cores_to_use = constants.NUM_CORES_TO_USE
    if not args.data_search_space:
        args.data_search_space = constants.DATA_SEARCH_SPACE
    if not args.model_layer_search_space:
        args.model_layer_search_space = constants.MODEL_LAYER_SEARCH_SPACE
    if not args.optimizer:
        args.optimizer = constants.OPTIMIZER
    if not args.loss_function:
        args.loss_function = constants.LOSS_FUNCTION
    if not args.metrics:
        args.metrics = constants.METRICS
    if not args.num_output_classes:
        args.num_output_classes = constants.NUM_OUTPUT_CLASSES
    if not args.width_of_dense_layer:
        args.width_of_dense_layer = constants.WIDTH_OF_DENSE_LAYER
    if not args.path_normal_files:
        args.path_normal_files = constants.PATH_NORMAL_FILES
    if not args.path_anomalous_files:
        args.path_anomalous_files = constants.PATH_ANOMALOUS_FILES
    if not args.path_noise_files:
        args.path_noise_files = constants.PATH_NOISE_FILES
    if not args.case_noise_files:
        args.case_noise_files = constants.CASE_NOISE_FILES
    if not args.num_normal_files_to_use:
        args.num_normal_files_to_use = constants.NUM_NORMAL_FILES_TO_USE
    if not args.num_anomalous_files_to_use:
        args.num_anomalous_files_to_use = constants.NUM_ANOMALOUS_FILES_TO_USE
    if not args.dataset_channel_to_use:
        args.dataset_channel_to_use = constants.DATASET_CHANNEL_TO_USE
    if not args.sound_gain:
        args.sound_gain = constants.SOUND_GAIN
    if not args.noise_gain:
        args.noise_gain = constants.NOISE_GAIN
    if not args.frame_size:
        args.frame_size = constants.FRAME_SIZE
    if not args.hop_length:
        args.hop_length = constants.HOP_LENGTH
    if not args.num_mel_filters:
        args.num_mel_filters = constants.NUM_MEL_FILTERS
    if not args.num_mfccs:
        args.num_mfccs = constants.NUM_MFCCS
    if not args.controller:
        args.controller = constants.CONTROLLER
    if not args.initialization:
        args.initialization = constants.INITIALIZATION
    if not args.max_num_layers:
        args.max_num_layers = constants.MAX_NUM_LAYERS
    if not args.num_epochs:
        args.num_epochs = constants.NUM_EPOCHS
    if not args.batch_size:
        args.batch_size = constants.BATCH_SIZE
    if not args.model_size_approximate_range:
        args.model_size_approximate_range = constants.MODEL_SIZE_APPROXIMATE_RANGE
    if not args.population_size:
        args.population_size = constants.POPULATION_SIZE
    if not args.population_update_ratio:
        args.population_update_ratio = constants.POPULATION_UPDATE_RATIO
    if not args.crossover_ratio:
        args.crossover_ratio = constants.CROSSOVER_RATIO

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
        args.model_layer_search_space, args.data_search_space)

    print("Loading dataset files from persistent storage...")
    dataset_loader = datasetloader.DatasetLoader(args.path_normal_files, args.path_anomalous_files, args.path_noise_files, args.case_noise_files,
                                                 args.num_normal_files_to_use, args.num_anomalous_files_to_use, args.dataset_channel_to_use, args.num_cores_to_use, args.sound_gain, args.noise_gain, constants.AUDIO_SECONDS_TO_LOAD)

    print("Initializing controller...")
    if args.controller == "evolution":
        controller = evolutionarycontroller.EvolutionaryController(
            search_space, args.population_size, args.max_num_layers, args.population_update_ratio, args.crossover_ratio, args.model_size_approximate_range, args.seed)
        controller.initialize_controller(args.initialization == "trivial")
    else:
        controller = randomcontroller.RandomController(
            search_space, args.seed, args.max_num_layers)

    # Run the Data Aware NAS
    data_model_generator = datamodelgenerator.DataModelGenerator(
        args.num_output_classes, args.loss_function, controller, dataset_loader, args.optimizer, args.metrics, args.width_of_dense_layer, args.num_epochs, args.batch_size, args.num_normal_files_to_use, args.num_anomalous_files_to_use, args.path_normal_files, args.path_anomalous_files, args.frame_size, args.hop_length, args.num_mel_filters, args.num_mfccs)
    pareto_front = data_model_generator.run_data_nas(args.num_models)

    # Print out results
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
