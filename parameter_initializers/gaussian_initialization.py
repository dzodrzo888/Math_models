"""This module is used to initialize weights using the Gaussian initialization technique."""
import numpy as np
from .initializor_base import InitializerBaseModel


class GaussianInitialization(InitializerBaseModel):
    """
    A class used to initialize weights using the Gaussian initialization technique.

    Args:
        InitializerBaseModel (_type_): _description_
    """

    def initialize_parameters(self, X: np.ndarray):
        """
        Initializes parameters using gaussian initialization.

        Args:
            n_features (int): _description_
        """
        _, n_features = X.shape

        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0

if __name__ == "__main__":
    random_var_init = GaussianInitialization()
    X = np.random.rand(1, 7)
    random_var_init.initialize_parameters(X)
    print(random_var_init.weights)
