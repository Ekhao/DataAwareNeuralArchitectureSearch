# A class to store information about a supervised dataset

# Standard Library Imports
from typing import Optional

# Third Party Imports
import numpy as np
import tensorflow as tf


class Data:
    def __init__(
        self,
        X_train: np.ndarray | tf.data.Dataset,
        X_test: np.ndarray | tf.data.Dataset,
        y_train: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray | tf.data.Dataset] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
