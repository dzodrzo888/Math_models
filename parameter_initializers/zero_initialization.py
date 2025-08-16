"""This module is used to initialize weights using the Zero initialization technique."""
import numpy as np
from .initializor_base import InitializerBaseModel

class ZeroInitialization(InitializerBaseModel):
    """
    A class used to initialize weights using the Zero initialization technique.

    Args:
        InitializerBaseModel (_type_): _description_
    """

    def initialize_parameters(self, X: np.ndarray):
        """
        Initialize parametrs using zero technique (weights, bias)

        Args:
            X (np.ndarray): X feature.
        """
        _, n_features = X.shape

        self.weights = np.zeros(n_features)

        self.bias = 0.0

if __name__ == "__main__":
    zero_initializer = ZeroInitialization()
    X = np.random.rand(1, 7)
    zero_initializer.initialize_parameters(X=X)
    print(zero_initializer.weights)