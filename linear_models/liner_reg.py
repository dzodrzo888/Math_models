"""This module is used to create LinearRegression class."""
import numpy as np
from helpers import early_stopping

class LinearRegression:
    """
    Class to calculate, fit and meassure accuracy of a lin_reg model.
    """

    def __init__(self, learning_rate=0.01, epochs=1000, ridge=None, lasso=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.ridge = ridge
        self.lasso = lasso

        if ridge and lasso:
            raise ValueError("Cannot initialize both ridge and lasso")

    def _initialize_parameters(self, n_features):
        """
        Initialize parametrs (weights, bias)

        Args:
            n_features (int): Number of features
        """
        # Weights initialization
        self.weights = np.zeros(n_features)
        # Bias initialization
        self.bias = 0.0

    def predict(self, X):
        """
        Predicts y_hat using the lin reg equation

        Args:
            X (np.array): X
            w (float): weights
            b (float): bias

        Returns:
            np.array: Predicted y value
        """
        # Compute predictions
        y_hat = np.dot(X, self.weights) + self.bias
        return y_hat

    def _compute_cost(self, y, y_hat):
        """
        Compute cost function

        Args:
            y (np.array): Y actual values.
            y_hat (np.array): Y predicted.

        Returns:
            float: Cost value
        """
        m = len(y)
        # MSE calculation
        mse = np.sum(np.square(y_hat - y)) / m

        if self.ridge:
            return mse + (self.ridge / (2 * m)) * np.sum(np.square(self.weights))
        elif self.lasso:
            return mse + (self.lasso / m) * (np.sum(np.abs(self.weights)))
        else:
            return mse

    def _compute_gradients(self, X, y, y_hat):
        """
        Computes gradients.

        Args:
            X (np.array): X feature.
            y (np.array): Y actual values.
            y_hat (np.array): Y predicted.

        Returns:
            dw(np.array): Derivative with respect to w.
            db(float): Derivative with respect to b.
        """
        # Calculate derivate values

        m = len(y)

        dw = (2/len(y))*np.dot(X.T, (y_hat - y))
        db = (2/len(y))*np.sum(y_hat - y)

        if self.ridge:
            dw += self.ridge / m * self.weights

        elif self.lasso:
            dw += self.lasso / m * np.sign(self.weights)

        return dw, db

    def fit(self, X, y):
        """
        Trains the linear regression model on inputed data.

        Args:
            X (np.array): X feature.
            y (np.array): Y actual value.
        """
        _, n_features = X.shape
        # Initialize parametrs
        self._initialize_parameters(n_features)

        early_stopping_cls = early_stopping.EarlyStopping(delta=0.001, patience=5, mode='min')

        for epoch in range(self.epochs):
            # linear model
            y_hat = self.predict(X)

            cost = self._compute_cost(y, y_hat)

            self.losses.append(cost)

            stop_training = early_stopping_cls.on_epoch_end(curr_epoch_val=cost, weights=self.weights)

            if stop_training:
                self.weights = early_stopping_cls.get_best_weights()
                best_cost = early_stopping_cls.get_best_cost_value()
                print(f"Training stopped, no more improvements {best_cost}")
                break

            dw, db = self._compute_gradients(X, y, y_hat)

            # Calcilate gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4}: Cost = {cost:.4f}")

    def rmse_calc(self, y, y_hat):
        """
        Computes squared mse.

        Args:
            y (np.array): Y actual values.
            y_hat (np.array): Y predicted.

        Returns:
            float: Root mean squared error.
        """
        m = len(y)
        mse = np.sum(np.square(y_hat - y)) / m

        return np.sqrt(mse)

    def r_sqrt_calc(self, y, y_hat):
        """
        Calcualtes R squared

        Args:
            y (np.array): Y actual values.
            y_hat (np.array): Y predicted.

        Returns:
            float: R squared.
        """

        # Calculate mean
        y_mean = np.mean(y)

        # Calcualte sum of squares
        ss_res = np.sum(np.square(y-y_hat))
        ss_total = np.sum(np.square(y-y_mean))

        # Calculate r squared
        r_sqrt = 1 - (ss_res/ss_total)

        return r_sqrt

if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(lasso=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = model.r_sqrt_calc(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")