"""This module is used to initialize weights using the He initialization technique"""
import numpy as np
from .initializor_base import InitializerBaseModel

class HeInitialization(InitializerBaseModel):
    """
    A class used to initialize weights using the He initialization technique.

    Args:
        InitializerBaseModel (ABC): Abstract model cls
    """

    def initialize_parameters(self, X: np.ndarray):
        """
        Initialize parametrs using He technique (weights, bias)

        Args:
            X (np.ndarray): X feature.
        """

        _, n_features = X.shape

        limit = np.sqrt(2/n_features)

        self.weights = np.random.normal(loc=0.0, scale=limit, size=n_features)

        self.bias=0.0

if __name__ == "__main__":
    he_initilizor = HeInitialization()
    X = np.random.rand(1, 7)
    he_initilizor.initialize_parameters(X=X)

    print(he_initilizor.weights)