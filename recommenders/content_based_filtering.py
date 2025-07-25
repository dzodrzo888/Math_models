"""This module implemnt the contentbasedfiltering cls"""
import numpy as np

class ContentBasedFiltering:
    """
    This class is used to implement the content based filtering alg.
    """

    def __init__(self, epochs=1000, learning_rate=0.1, lambda_=None, X=None, b=None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.regularization = False

        if self.lambda_:
            self.regularization = True

        self.X = X
        self.b = b

    def predict(self, X: np.ndarray, W: np.ndarray, b: np.ndarray):
        """
        Function used to predict the rating of a user for a item.

        Args:
            X (np.ndarray): Input features for user.
            W (np.ndarray): Input featurese for item.
            b (np.ndarray): Bias

        Returns:
            r_hat (np.ndarray): Prediction matrix
        """

        r_hat = np.matmul(X, W.T) + b

        return r_hat

    def _compute_cost(self, X: np.ndarray, W:np.ndarray, b: np.ndarray, Y: np.ndarray, R: np.ndarray):
        """
        Function used to compute costs of input features.

        Args:
            X (np.ndarray): Input features for user.
            W (np.ndarray): Input featurese for item.
            b (np.ndarray): Bias
            Y (np.ndarray): Target value
            R (np.ndarray): Rating matrix.

        Returns:
            cost(float): Cost value
        """

        n_items = np.sum(R)

        error = np.square(R * ((np.matmul(X, W.T) + b) - Y))

        if self.regularization:
            cost = np.sum(error) / (2 * n_items) + (self.lambda_ / (2 * n_items)) * np.sum(np.square(X))

        else:
            cost = np.sum(error) / (2 * n_items)

        return cost

    def _compute_gradients(self, X: np.ndarray, W:np.ndarray, b: np.ndarray, Y: np.ndarray, R: np.ndarray):
        """
        Computes gradients

        Args:
            X (np.ndarray): Input features for user.
            W (np.ndarray): Input featurese for item.
            b (np.ndarray): Bias
            Y (np.ndarray): Target value
            R (np.ndarray): Rating matrix.

        Returns:
            np.ndarray: Calculated gradients
        """
        n_items = np.sum(R)

        error = R * ((np.matmul(X, W.T) + b) - Y)
        dx = np.matmul(error, W) / n_items

        if self.regularization:
            dx += self.lambda_ * self.X / n_items

        db = np.sum(error, axis=0) / n_items

        return dx, db

    def fit(self, W: np.ndarray, Y: np.ndarray, R: np.ndarray, n_features=10):
        """
        Fits the model and updates the parametrs

        Args:
            Y (np.ndarray): Target value
            R (np.ndarray): Rating matrix.
        """

        n_users, n_items = Y.shape

        if self.X is None:
            self.X = np.random.rand(n_users, n_features)
        if self.b is None:
            self.b = np.zeros(n_items)

        for epoch in range(self.epochs):

            dx, db = self._compute_gradients(X=self.X, W=W, b=self.b, Y=Y, R=R)
            cost = self._compute_cost(X=self.X, W=W, b=self.b, Y=Y, R=R)

            self.X -= self.learning_rate * dx
            self.b -= self.learning_rate * db

            if epoch % 100 == 0:
                print(f"Curretn loss: {cost:.2f}. For epoch: {epoch}")


if __name__ == "__main__":
    X = np.random.rand(5, 3)  
    W = np.random.rand(4,3)
    b = np.zeros(4)

    Y = np.array(
        [[0, 0, 0, 0],
        [0, 0, 0, 0,],
        [0, 0, 0, 0],
        [0, 0, 0, 0,],
        [5, 0, 0, 0,]])
    R = np.array(
        [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0]])

    cont_bs_cls = ContentBasedFiltering(X=X, b=b, lambda_=0.1)

    cont_bs_cls.fit(Y=Y, R=R, W=W)
    print(cont_bs_cls.predict(X=X, W=W, b=b))