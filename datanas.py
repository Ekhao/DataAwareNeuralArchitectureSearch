# This python file contains the main function of the Data Aware Neural Architecture Search. It is responsible for parsing the command line arguments, initializing the search space, loading the data and initializing the search strategy. It then calls the search strategy to perform the search and returns the pareto front.

# Standard Library Imports
import argparse
import json
import datetime
import csv

# Local Imports
import searchspace
import datamodelgenerator
import dataset_loaders.toyconveyordatasetloader
import dataset_loaders.wakevisiondatasetloader
import search_strategies.randomsearchstrategy as randomsearchstrategy
import search_strategies.evolutionarysearchstrategy as evolutionarysearchstrategy
import search_strategies.supernetevolutionary as supernetevolutionary


def main():
    # Parse the command line arguments
    argparser = argparse.ArgumentParser(
        description="A simple implementation of a Data Aware NAS. Note the constants.py file also can be used to control settings. The command line arguments take precedence over the constants.py file. The search space should be configured in the config.json file."
    )

    # General Parameters
    argparser.add_argument(
        "-s",
        "--seed",
        help="A seed to be given to the random number generators of the program.",
        type=int,
    )
    argparser.add_argument(
        "-snet",
        "--supernet",
        action="store_true",
        help="A flag to set wheter the neural architecture search should use supernets and weight sharing. This should generally speed up the neural architecture search.",
    )

    # Joblib Parameters
    argparser.add_argument(
        "-nc",
        "--num_cores_to_use",
        help="The number of cpu cores to use for parallel processing. Mainly used for data loading. Can be set to -1 to use all available cores.",
        type=int,
    )

    # Model Parameters
    argparser.add_argument(
        "-o",
        "--optimizer",
        help="The optimizer to use for training the models. Give as a string corresponding to the alias of a TensorFlow optimizer.",
    )
    argparser.add_argument(
        "-l",
        "--loss",
        help="The loss function to use for training the models. Give as a string corresponding to the alias of a TensorFlow loss function.",
    )
    argparser.add_argument(
        "-no",
        "--num_output_classes",
        help="The number of outputs classes that the created models should have.",
        type=int,
    )
    argparser.add_argument(
        "-wd",
        "--width_dense_layer",
        help="The width of the dense layer after the convolutional layers in the model. Be aware that this argument can cause an explosion of model parameters.",
    )

    # Dataset Parameters
    argparser.add_argument(
        "-dn",
        "--dataset_name",
        help="The name of the dataset to use for training. An appropriate loader for that dataset need to be defined. If the dataset needs additional options then specify these in the config file.",
    )

    # Search Strategy Parameters
    argparser.add_argument(
        "-ss",
        "--search_strategy",
        help='The search strategy to use to direct the search. Supported options are "evolution", "random" and "supernet_evo". "supernet_evo" is only supported for supernet searches, while the other strategies are supported for traditional searches.',
        choices=["evolution", "random", "supernet_evo"],
    )
    argparser.add_argument(
        "-i",
        "--initialization",
        help='How to initialize the first generation of models for the "evolution" search strategy. Supported options are "trivial" and "random". Does nothing if paired with the "random" search strategy.',
        choices=["trivial", "random"],
    )
    argparser.add_argument(
        "-ml",
        "--max_num_layers",
        help="The maximum number of layers to use for the models.",
        type=int,
    )

    # Evaluation Parameters
    argparser.add_argument(
        "-ne",
        "--num_epochs",
        help="The number of epochs to train a model for before evaluation",
        type=int,
    )
    argparser.add_argument(
        "-bs", "--batch_size", help="The batch size to use for training.", type=int
    )
    argparser.add_argument(
        "-mrc",
        "--max_ram_consumption",
        help="The maximum ram consumption allowed by data and intermediate representations",
        type=int,
    )
    argparser.add_argument(
        "-mfc",
        "--max_flash_consumption",
        help="The maximum flash consumption allowed for models",
    )
    argparser.add_argument(
        "-ddm",
        "--data_dtype_multiplier",
        help="The amount of bytes that the datatype which input data is stored in is expected to take",
    )
    argparser.add_argument(
        "-mdm",
        "--model_dtype_multiplier",
        help="The amount of bytes that the datatype which model parameters are stored in is expected to take",
    )

    # Evolutionary Parameters
    argparser.add_argument(
        "-ps",
        "--population_size",
        help="The population size to use for the evolutionary search strategy. This argument is ignored if the random search strategy is used.",
        type=int,
    )
    argparser.add_argument(
        "-ur",
        "--population_update_ratio",
        help="The ratio of the population to be discarded and regenerated when the population is updated. This argument is ignored if the random search strategy is used.",
        type=float,
    )
    argparser.add_argument(
        "-cr",
        "--crossover_ratio",
        help="The ratio of the updated population to be generated by crossover. The rest of the updated population is generated by mutations. This argument is ignored if the random search strategy is used.",
        type=float,
    )

    args = argparser.parse_args()

    # Parse config file
    config_file = open("config.json", "r")
    config = json.load(config_file)

    config = config["datanas_config"]
    general_config = config["general_config"]
    joblib_config = config["joblib_config"]
    search_space_config = config["search_space_config"]
    model_config = config["model_config"]
    dataset_config = config["dataset_config"]
    search_strategy_config = config["search_strategy_config"]
    evaluation_config = config["evaluation_config"]
    evolutionary_config = config["evolutionary_config"]

    # Set options according to command line arguments and config file
    if not args.seed:
        args.seed = general_config["seed"]
    if not args.supernet:
        args.supernet = general_config["supernet"]
    if not args.num_cores_to_use:
        args.num_cores_to_use = joblib_config["num_cores_to_use"]
    if not args.optimizer:
        args.optimizer = model_config["optimizer"]
    if not args.loss:
        args.loss = model_config["loss"]
    if not args.num_output_classes:
        args.num_output_classes = model_config["num_output_classes"]
    if not args.width_dense_layer:
        args.width_dense_layer = model_config["width_dense_layer"]
    if not args.dataset_name:
        args.dataset_name = dataset_config["dataset_name"]
    args.dataset_options = dataset_config["dataset_options"]
    if not args.search_strategy:
        args.search_strategy = search_strategy_config["search_strategy"]
    if not args.initialization:
        args.initialization = search_strategy_config["initialization"]
    if not args.max_num_layers:
        args.max_num_layers = search_strategy_config["max_num_layers"]
    if not args.num_epochs:
        args.num_epochs = evaluation_config["num_epochs"]
    if not args.batch_size:
        args.batch_size = evaluation_config["batch_size"]
    if not args.max_ram_consumption:
        args.max_ram_consumption = evaluation_config["max_ram_consumption"]
    if not args.max_flash_consumption:
        args.max_flash_consumption = evaluation_config["max_flash_consumption"]
    if not args.data_dtype_multiplier:
        args.data_dtype_multiplier = evaluation_config["data_dtype_multiplier"]
    if not args.model_dtype_multiplier:
        args.model_dtype_multiplier = evaluation_config["model_dtype_multiplier"]
    if not args.population_size:
        args.population_size = evolutionary_config["population_size"]
    if not args.population_update_ratio:
        args.population_update_ratio = evolutionary_config["population_update_ratio"]
    if not args.crossover_ratio:
        args.crossover_ratio = evolutionary_config["crossover_ratio"]

    # Add the search spaces from the config file to arguments:
    args.data_search_space = search_space_config["data_search_space"]
    args.model_search_space = search_space_config["model_search_space"]

    print("Initializing search space...")
    search_space = searchspace.SearchSpace(
        args.data_search_space, args.model_search_space
    )

    print("Loading dataset files from persistent storage...")
    if args.dataset_name == "ToyConveyor":
        dataset_loader = (
            dataset_loaders.toyconveyordatasetloader.ToyConveyorDatasetLoader(
                args.num_cores_to_use,
                args.dataset_options,
            )
        )
    elif args.dataset_name == "wake_vision":
        dataset_loader = (
            dataset_loaders.wakevisiondatasetloader.WakeVisionDatasetLoader()
        )
    else:
        raise ValueError(f'No dataset loader defined for "{args.dataset_name}".')

    print("Initializing search strategy...")
    if args.search_strategy == "evolution":
        search_strategy = evolutionarysearchstrategy.EvolutionarySearchStrategy(
            search_space,
            args.population_size,
            args.max_num_layers,
            args.population_update_ratio,
            args.crossover_ratio,
            args.max_ram_consumption,
            args.max_flash_consumption,
            args.seed,
        )
        search_strategy.initialize_search_strategy(args.initialization == "trivial")
    elif args.search_strategy == "random":
        search_strategy = randomsearchstrategy.RandomSearchStrategy(
            search_space, args.max_num_layers, args.seed
        )
    elif args.search_strategy == "supernet_evo":
        search_strategy = supernetevolutionary.SuperNetEvolutionary(
            search_space=search_space,
            population_size=args.population_size,
            population_update_ratio=args.population_update_ratio,
            crossover_ratio=args.crossover_ratio,
            max_ram_consumption=args.max_ram_consumption,
            max_flash_consumption=args.max_flash_consumption,
            seed=args.seed,
        )
    else:
        raise ValueError(f'No "{args.search_strategy}" defined".')

    # Run the Data Aware NAS
    data_model_generator = datamodelgenerator.DataModelGenerator(
        num_target_classes=args.num_output_classes,
        loss_function=args.loss,
        search_strategy=search_strategy,
        dataset_loader=dataset_loader,
        optimizer=args.optimizer,
        width_dense_layer=args.width_dense_layer,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_ram_consumption=args.max_ram_consumption,
        max_flash_consumption=args.max_flash_consumption,
        data_dtype_multiplier=args.data_dtype_multiplier,
        model_dtype_multiplier=args.model_dtype_multiplier,
        supernet_flag=args.supernet,
        **args.dataset_options,
    )
    pareto_front = data_model_generator.run_data_nas()

    # Save pareto results
    pareto_csv_log_name = (
        f"datamodel_logs/{datetime.datetime.now().isoformat()}_pareto.csv"
    )
    with open(pareto_csv_log_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Model Number",
                "Data Configuration",
                "Model Configuration",
                "Accuracy",
                "Precision",
                "Recall",
                "Ram Consumption",
                "Flash Consumption",
            ]
        )
        for data_model in pareto_front:
            writer.writerow(
                [
                    data_model.model_number,
                    data_model.configuration.data_configuration,
                    data_model.configuration.model_configuration,
                    data_model.accuracy,
                    data_model.precision,
                    data_model.recall,
                    data_model.ram_consumption,
                    data_model.flash_consumption,
                ]
            )


if __name__ == "__main__":
    main()
