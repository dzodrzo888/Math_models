import numpy as np

class NN:
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
    
    def softmax_calc(self, z: np.array) -> np.array:
        """
        Function to do softmax calculation

        Args:
            z (np.array): Logit

        Returns:
            a (np.array): 
        """
        ez = np.exp(z)

        a = ez/np.sum(ez)

        return a

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
    
    def relu_calc(self, z: np.array) -> float:
        """
        Relu activation calculation

        Args:
            z (np.array): Logit

        Returns:
            float: 
        """
        return np.maximum(0,z)
    
    def _activation(self, z: np.array, type: str) -> np.array:
        """
        Function to apply a activation method

        Args:
            z (np.array): Logit
            type (str): Type of activation function to be used.

        Raises:
            ValueError: Raised if invalid activation function type inputed.

        Returns:
            np.array: Activation result
        """

        if type == "softmax":
            return self.softmax_calc(z=z)
        elif type == "relu":
            return self.relu_calc(z=z)
        elif type == "sigmoid":
            return self.sigmoid_calc(z=z)
        else:
            raise ValueError("Invalid activation function inputed. Supported function: ['softmax', 'relu', 'sigmoid']")
    
    def dense(self, A_in: np.array, W: np.array, b: np.array, type: str) -> np.array:
        """
        NN layer used to 

        Args:
            A_in (np.array): Input vector
            W (np.array): Weights vector
            b (np.array): Bias vector

        Returns:
            A_out(np.array): _description_
        """
        Z = np.matmul(A_in, W) + b

        A_out = self._activation(Z)

        return A_out, Z

    def sequential(self, X: np.array, weights: list[np.array], biases: np.array, activations: list[str]) -> np.array:
        """
        Sequential func used to calculate the propability

        Args:
            X (np.array): Input vector
            weights (list[np.array]): List of weights
            biases (np.array): Vector of biases.
            activations (list[str]): List of activation types

        Returns:
            a_out(np.array): Prob value of a_out
        """
        A = X
        A_s = [A]
        Z_s = []

        for i in range(len(weights)):
            A, Z = self.dense(A, weights[i], biases[i], type=activations[i])
            A_s.append(A)
            Z_s.append(Z)

        return A_s, Z_s 
        
if __name__ == "__main__":
    ...