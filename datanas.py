# This python file contains the main function of the Data Aware Neural Architecture Search. It is responsible for parsing the command line arguments, initializing the search space, loading the data and initializing the controller. It then calls the controller to perform the search and returns the pareto front.

# Standard Library Imports
import argparse
import json

# Third Party Imports
import tensorflow as tf

# Local Imports
import searchspace
import datamodelgenerator
import datasetloader
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
    argparser.add_argument("-l", "--loss",
                           help="The loss function to use for training the models. Give as a string corresponding to the alias of a TensorFlow loss function.")
    argparser.add_argument("-no", "--num_output_classes",
                           help="The number of outputs classes that the created models should have.", type=int)
    argparser.add_argument("-wd", "--width_dense_layer",
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
    argparser.add_argument("-nn", "--num_normal_files",
                           help="The number of normal files to use for training.", type=int)
    argparser.add_argument("-na", "--num_anomalous_files",
                           help="The number of anomalous files to use for training.", type=int)
    argparser.add_argument("-ch", "--dataset_channel",
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
        "-ams", "--approximate_model_size", help="An approximate size of the models to be generated. Is used to decide whether a generated model is scored well or poor on its model size.", type=int)

    # Evolutionary Parameters
    argparser.add_argument("-ps", "--population_size",
                           help="The population size to use for the evolutionary controller. This argument is ignored if the randomcontroller is used.", type=int)
    argparser.add_argument("-ur", "--population_update_ratio",
                           help="The ratio of the population to be discarded and regenerated when the population is updated. This argument is ignored if the random controller is used.", type=float)
    argparser.add_argument("-cr", "--crossover_ratio",
                           help="The ratio of the updated population to be generated by crossover. The rest of the updated population is generated by mutations. This argument is ignored if the random controller is used.", type=float)

    args = argparser.parse_args()

    # Parse config file
    config_file = open("config.json", "r")
    config = json.load(config_file)

    config = config["datanas-config"]
    general_config = config["general-config"]
    joblib_config = config["joblib-config"]
    search_space_config = config["search-space-config"]
    model_config = config["model-config"]
    dataset_config = config["dataset-config"]
    preprocessing_config = config["preprocessing-config"]
    controller_config = config["controller-config"]
    evaluation_config = config["evaluation-config"]
    evolutionary_config = config["evolutionary-config"]

    # Set options according to command line arguments and config file
    if not args.num_models:
        args.num_models = general_config["num-models"]
    if not args.seed:
        args.seed = general_config["seed"]
    if not args.num_cores_to_use:
        args.num_cores_to_use = joblib_config["num-cores-to-use"]
    if not args.data_search_space:
        args.data_search_space = search_space_config["data-search-space"]
    if not args.model_layer_search_space:
        args.model_layer_search_space = search_space_config["model-layer-search-space"]
    if not args.optimizer:
        args.optimizer = model_config["optimizer"]
    if not args.loss:
        args.loss = model_config["loss"]
    if not args.num_output_classes:
        args.num_output_classes = model_config["num-output-classes"]
    if not args.width_dense_layer:
        args.width_dense_layer = model_config["width-dense-layer"]
    if not args.path_normal_files:
        args.path_normal_files = dataset_config["path-normal-files"]
    if not args.path_anomalous_files:
        args.path_anomalous_files = dataset_config["path-anomalous-files"]
    if not args.path_noise_files:
        args.path_noise_files = dataset_config["path-noise-files"]
    if not args.case_noise_files:
        args.case_noise_files = dataset_config["case-noise-files"]
    if not args.num_normal_files:
        args.num_normal_files = dataset_config["num-normal-files"]
    if not args.num_anomalous_files:
        args.num_anomalous_files = dataset_config["num-anomalous-files"]
    if not args.dataset_channel:
        args.dataset_channel = dataset_config["dataset-channel"]
    if not args.sound_gain:
        args.sound_gain = dataset_config["sound-gain"]
    if not args.noise_gain:
        args.noise_gain = dataset_config["noise-gain"]
    if not args.frame_size:
        args.frame_size = preprocessing_config["frame-size"]
    if not args.hop_length:
        args.hop_length = preprocessing_config["hop-length"]
    if not args.num_mel_filters:
        args.num_mel_filters = preprocessing_config["num-mel-filters"]
    if not args.num_mfccs:
        args.num_mfccs = preprocessing_config["num-mfccs"]
    if not args.controller:
        args.controller = controller_config["controller"]
    if not args.initialization:
        args.initialization = controller_config["initialization"]
    if not args.max_num_layers:
        args.max_num_layers = controller_config["max-num-layers"]
    if not args.num_epochs:
        args.num_epochs = evaluation_config["num-epochs"]
    if not args.batch_size:
        args.batch_size = evaluation_config["batch-size"]
    if not args.approximate_model_size:
        args.approximate_model_size = evaluation_config["approximate-model-size"]
    if not args.population_size:
        args.population_size = evolutionary_config["population-size"]
    if not args.population_update_ratio:
        args.population_update_ratio = evolutionary_config["population-update-ratio"]
    if not args.crossover_ratio:
        args.crossover_ratio = evolutionary_config["crossover-ratio"]

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
        args.data_search_space, args.model_layer_search_space)

    print("Loading dataset files from persistent storage...")
    dataset_loader = datasetloader.DatasetLoader(args.path_normal_files, args.path_anomalous_files, args.path_noise_files, args.case_noise_files,
                                                 args.num_normal_files, args.num_anomalous_files, args.dataset_channel, args.num_cores_to_use, args.sound_gain, args.noise_gain, config["audio-seconds-to-load"])

    print("Initializing controller...")
    if args.controller == "evolution":
        controller = evolutionarycontroller.EvolutionaryController(
            search_space, args.population_size, args.max_num_layers, args.population_update_ratio, args.crossover_ratio, args.approximate_model_size, args.seed)
        controller.initialize_controller(args.initialization == "trivial")
    else:
        controller = randomcontroller.RandomController(
            search_space, args.max_num_layers, args.seed)

    # Run the Data Aware NAS
    data_model_generator = datamodelgenerator.DataModelGenerator(
        args.num_output_classes, args.loss, controller, dataset_loader, args.optimizer, args.width_dense_layer, args.num_epochs, args.batch_size, args.num_normal_files, args.num_anomalous_files, args.path_normal_files, args.path_anomalous_files, args.frame_size, args.hop_length, args.num_mel_filters, args.num_mfccs)
    pareto_front = data_model_generator.run_data_nas(args.num_models)

    # Print out results
    print("Models on pareto front: ")
    for data_model in pareto_front:
        print(data_model.data_configuration)
        print(data_model.model_configuration)
        print(f"Accuracy: {data_model.accuracy}, Precision: {data_model.precision}, Recall: {data_model.recall}, Model Size (in bytes): {data_model.model_size}.")
        print("-"*200)


if __name__ == "__main__":
    main()
