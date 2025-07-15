import numpy as np

class BinaryNN:
    """
    This class is used to create a simple binary classification neural network.
    """
    def __init__(self):
        ...

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

    def dense(self, A_in: np.array, W: np.array, b: np.array) -> np.array:
        """
        NN layer used to 

        Args:
            A_in (np.array): Input vector
            W (np.array): Weights vector
            b (np.array): Bias vector

        Returns:
            A_out(np.array): _description_
        """
        A_out_logit = np.matmul(A_in, W) + b

        A_out = self.sigmoid_calc(A_out_logit)

        return A_out

    def sequential(self, X: np.array, weights: list[np.array], biases: np.array) -> np.array:
        """
        Sequential func used to calculate the propability

        Args:
            X (np.array): Input vector
            weights (list[np.array]): List of weights
            biases (np.array): Vector of biases.

        Returns:
            a_out(np.array): Prob value of a_out
        """
        units = weights.shape[0]

        for j in range(units):

            if j == 0:
                a_j = self.dense(X, weights[0], biases[0])
            else:
                a_j = self.dense(a_j, weights[j], biases[j])
        
        a_out = a_j

        return a_out
        
if __name__ == "__main__":
    ...