import numpy as np
from .initializor_base import InitializerBaseModel

class XavierInitialization(InitializerBaseModel):
    """
    A class used to initialize weights using the Xavier initialization technique.

    Args:
        InitializerBaseModel (ABC): Abstract model cls
    """

    def initialize_parameters(self, X: np.ndarray, **kwargs):
        """
        Initialize parametrs using Xavier technique (weights, bias)

        Args:
            X (np.ndarray): X feature.
        """

        _, n_features = X.shape

        n_output = kwargs.get("n_output", 1)

        limit = np.sqrt(2 / (n_features + n_output))

        self.weights = np.random.normal(loc=0.0, scale=limit, size=n_features)

if __name__ == "__main__":
    xavier_initializer = XavierInitialization()
    X = np.random.rand(1, 7)
    xavier_initializer.initialize_parameters(X, kwargs={"n_output": 9})

    print(xavier_initializer.weights)