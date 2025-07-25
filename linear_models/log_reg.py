"""This module is used to create LogisticRegression class."""
import numpy as np

class LogisticRegression:
    """
    Class to calculate, fit and meassure accuracy of a log_reg model.
    """
    def __init__(self, learning_rate=0.01, epochs=1000, ridge=None, lasso=None):
        # Initialize variables
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.ridge = ridge
        self.lasso = lasso

        if ridge and lasso:
            raise ValueError("Cannot initialize both ridge and lasso")

    def _sigmoid(self, z):
        """
        Sigmoid activation function.

        Args:
            z (np.array): Logit.

        Returns:
            sigmoid (np.array): Sigmoid.
        """
        # Sigmoid function calculation
        sigmoid = 1/(1+np.exp(-z))

        return sigmoid

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

    def _compute_predictions(self, X):
        """
        Computes logit predictions.

        Args:
            X (np.array): X feature.

        Returns:
            z(np.array): Logits.
        """
        # Compute predictions
        z=np.dot(X, self.weights) + self.bias

        return z

    def _compute_gradients(self, X, y, y_pred):
        """
        Computes gradients.

        Args:
            X (np.array): X feature.
            y (np.array): Y actual values.
            y_pred (np.array): Y predicted.

        Returns:
            dw(np.array): Derivative with respect to w.
            db(float): Derivative with respect to b.
        """
        # Initialize variables
        m = len(y)
        error = y_pred - y

        # Calculate derivatives values
        dw = (1/m)*np.dot(X.T, error)
        db = (1/m)*np.sum(error)

        if self.ridge:
            dw += self.ridge / m * self.weights

        elif self.lasso:
            dw += self.lasso / m * np.sign(self.weights)

        return dw, db

    def _loss(self, y, y_pred):
        """
        Compute loss.

        Args:
            y (np.array): Y actual values.
            y_pred (np.array): Y predicted.

        Returns:
            cost (float): Cost.
        """
        # Set vars - Add epsilon to prevent -inf
        m = len(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Calculate loss function
        loss = -y*np.log(y_pred)-(1-y)*np.log(1-y_pred)
        cost = np.mean(loss)

        if self.ridge:
            return cost + (self.ridge / (2 * m)) * np.sum(np.square(self.weights))
        elif self.lasso:
            return cost + (self.lasso / m) * (np.sum(np.abs(self.weights)))
        else:
            return cost

    def fit(self, X, y):
        """
        Trains the logistic regression model on inputed data.

        Args:
            X (np.array): X feature.
            y (np.array): Y actual value.
        """
        # Set vars
        _, n_features = X.shape
        self._initialize_parameters(n_features)

        for i in range(self.epochs):
            # Linear model
            lin_pred = self._compute_predictions(X)
            y_hat = self._sigmoid(lin_pred)

            dw, db = self._compute_gradients(X, y, y_hat)

            # Calculate gradient descent
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

            # Print losses
            if i % 100 == 0:
                loss = self._loss(y, y_hat)
                self.losses.append(loss)
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """
        Predict propabilities

        Args:
            X (np.array): X features

        Returns:
            y_hat (np.array): Predictions based on the model.
        """
        # Get predictions
        y_hat = self._sigmoid(self._compute_predictions(X))

        return y_hat

    def predict(self, X):
        """
        Predict cls labels based on the X input

        Args:
            X (np.array): X features.

        Returns:
            cls_labels (np.array): Class labels
        """
        # Predict cls labels.
        cls_labels = (self.predict_proba(X) >= 0.5).astype(int)
        
        return cls_labels

    def accuracy(self, y_true, y_pred):
        """
        Calculates accuracy of predictions.

        Args:
            y_true (np.array): True y values.
            y_pred (np.array): Predicted y values

        Returns:
            float: The accuracy of predictions
        """
        # Get accuracy of predictions
        return np.mean(y_true == y_pred)

# Example usage
if __name__ == "__main__":
    # Generate dummy dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(learning_rate=0.1, epochs=1000, ridge=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = model.accuracy(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")