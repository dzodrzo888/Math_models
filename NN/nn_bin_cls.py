import numpy as np

class BinaryNN:
    """
    This class is used to create a simple binary classification neural network.
    """
    def __init__(self):
        ...

    def compute_predictions(self, X: np.array, w: float, b: float) -> np.array:
        """
        Calculates predictions.

        Args:
            x (np.array): Input values.
            w (float): Weights.
            b (float): Bias.

        Returns:
            z(np.array): Prediction
        """
        z = np.dot(X, w) + b

        return z

    def sigmoid_calc(self, z: np.array) -> np.array:
        """
        Function to calculate sigmoid value

        Args:
            z (np.array): Logit
        
        Returns:
            a: sigmoid
        """
        a = 1/(1 + np.exp(-z))

        return a

    def dense(self):
        
