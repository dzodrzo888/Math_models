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
    
    def _MSE_calc(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculates mean squared error

        Args:
            y_true (np.array): Y true value
            y_pred (np.array): Y predicted value

        Returns:
            mse (float): Mean squared error
        """
        mse = np.mean(np.square(y_true - y_pred))
        return mse
    
    def _binary_crossentropy_calc(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculates binary crossentropy

        Args:
            y_true (np.array): Y true value.
            y_pred (np.array): Y predicted value.

        Returns:
            bin_loss_mean (float): Binary crossentropy loss value.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        bin_loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

        bin_loss_mean = -np.mean(bin_loss)

        return bin_loss_mean

    def _compute_loss(self, y_true, y_pred, type: str):
        if type == "mean_squared_error":
            return self._MSE_calc(y_true=y_true, y_pred=y_pred)
        elif type == "binary_crossentropy":
            return self._binary_crossentropy_calc(y_true=y_true, y_pred=y_pred)
        else:
            raise ValueError("Invalid loss function inputed. Supported function: ['mean_squared_error', 'binary_crossentropy']")
        
if __name__ == "__main__":
    ...