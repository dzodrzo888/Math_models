"""This module is used to create LogisticRegression class."""
import numpy as np
from linear_models.linear_base import LinearBaseModel

class LogisticRegression(LinearBaseModel):
    """
    Class to calculate, fit and meassure accuracy of a log_reg model.
    """
    def _sigmoid(self, z: np.ndarray):
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

    def _compute_predictions(self, X: np.ndarray):
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

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
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

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray):
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
        if self.lasso:
            return cost + (self.lasso / m) * (np.sum(np.abs(self.weights)))
        return cost

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the logistic regression model on inputed data.

        Args:
            X (np.array): X feature.
            y (np.array): Y actual value.
        """
        # Set vars
        self.initializer.initialize_parameters(X=X)
        self.weights = self.initializer.weights
        self.bias = self.initializer.bias

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
                loss = self._compute_loss(y, y_hat)
                self.losses.append(loss)
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X: np.ndarray):
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

    def predict(self, X: np.ndarray) -> np.ndarray:
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

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray):
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