import numpy as np

class NN:
    """
    This class is used to create a simple binary classification neural network.
    """
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

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
    
    def relu_calc(self, z: np.array) -> np.array:
        """
        Relu activation calculation

        Args:
            z (np.array): Logit

        Returns:
            float: 
        """
        return np.maximum(0,z)
    
    def _activation(self, z: np.array, kind: str) -> np.array:
        """
        Function to apply a activation method

        Args:
            z (np.array): Logit
            kind (str): kind of activation function to be used.

        Raises:
            ValueError: Raised if invalid activation function kind inputed.

        Returns:
            np.array: Activation result
        """

        if kind == "softmax":
            return self.softmax_calc(z=z)
        elif kind == "relu":
            return self.relu_calc(z=z)
        elif kind == "sigmoid":
            return self.sigmoid_calc(z=z)
        else:
            raise ValueError("Invalid activation function inputed. Supported function: ['softmax', 'relu', 'sigmoid']")
        
    def _activation_derivative(self, a, kind: str):
        """
        Calculates the derivatives of the activation function

        Args:
            a (float): Activation function result
            kind (str): Type of activation function

        Raises:
            ValueError: Raised if invalid activation function kind inputed.

        Returns:
            float/int: Derivative of the activation
        """
        if kind == "sigmoid":
            return a * (1 - a)
        elif kind == "relu":
            return np.where(a > 0, 1, 0)
        else:
            raise ValueError("Invalid activation function inputed. Supported function: ['relu', 'sigmoid']")
        
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

    def _compute_loss(self, y_true, y_pred, kind: str):
        if kind == "mean_squared_error":
            return self._MSE_calc(y_true=y_true, y_pred=y_pred)
        elif kind == "binary_crossentropy":
            return self._binary_crossentropy_calc(y_true=y_true, y_pred=y_pred)
        else:
            raise ValueError("Invalid loss function inputed. Supported function: ['mean_squared_error', 'binary_crossentropy']")
        
    
    def dense(self, A_in: np.array, W: np.array, b: np.array, kind: str) -> np.array:
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

        A_out = self._activation(Z, kind=kind)

        return A_out, Z

    def sequential(self, X: np.array, weights: list[np.array], biases: np.array, activations: list[str]) -> np.array:
        """
        Sequential func used to calculate the propability

        Args:
            X (np.array): Input vector
            weights (list[np.array]): List of weights
            biases (np.array): Vector of biases.
            activations (list[str]): List of activation kinds

        Returns:
            a_out(np.array): Prob value of a_out
        """
        A = X
        A_s = [A]
        Z_s = []

        for i in range(len(weights)):
            A, Z = self.dense(A, weights[i], biases[i], kind=activations[i])
            A_s.append(A)
            Z_s.append(Z)

        return A_s, Z_s 
    
    def _backward(self, activations, zs, y_true, loss="binary_crossentropy"):
        ...

    def _update_parameters(self, grads_w, grads_b):
        ...
    
    def fit(X, y, epochs=100, loss="binary_crossentropy"):
        ...
    
    def predict(X, threshold=0.5):
        ...

    def predict_proba(X):
        ...

if __name__ == "__main__":
    ...