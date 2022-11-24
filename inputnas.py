# In the beginning we also only look for convolutional models as a proof of concept. (Let us stay in on the topic of audio processing) - I believe that edge impulse does the same

import inputmodelgenerator
import constants
import randomcontroller
import searchspace


def main():
    input_model_generator = inputmodelgenerator.InputModelGenerator(
        constants.NUM_OUTPUT_CLASSES, constants.LOSS_FUNCTION, controller=randomcontroller.RandomController(searchspace.SearchSpace(model_layer_search_space=constants.MODEL_LAYER_SEARCH_SPACE, input_search_space=constants.INPUT_SEARCH_SPACE), seed=42))
    input_model_generator.run_input_nas(num_of_samples=2)


if __name__ == "__main__":
    main()
