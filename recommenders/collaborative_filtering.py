import numpy as np

class ColaborativeFiltering:
    """
    This class is used to perform operations using colaborative filetring.
    """

    def __init__(self, epochs=1000, learning_rate=0.01, lambda_=None, X=None, W=None, b=None):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.regularization = False

        if lambda_:
            self.regularization = True
            
        self.X = X
        self.W = W
        self.b = b

    def _compute_cost(self, X: np.ndarray, W:np.ndarray, b: np.ndarray, Y: np.ndarray, R: np.ndarray) -> float:
        """
        Function used to compute costs of input features.

        Args:
            X (np.ndarray): Input features for user.
            W (np.ndarray): Input featurese for item.
            b (np.ndarray): Bias
            Y (np.ndarray): Target value
            R (np.ndarray): Rating matrix.

        Returns:
            cost_scalar(float): Cost value
        """
        
        error = R * np.square((np.matmul(X, W.T) + b) - Y) / 2

        if self.regularization:
            cost = np.sum(error) + self.lambda_ / 2 * np.sum(np.square(W)) + self.lambda_ / 2 * np.sum(np.square(X)) 
        else:
            cost = np.sum(error)

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

        error = R * ((np.matmul(X, W.T) + b) - Y)
        dx = np.matmul(error, W)
        dw = np.matmul(error.T, X)
        
        if self.regularization:
            dx += self.lambda_ * X

            dw += self.lambda_ * W

        db = np.sum(error, axis=0)

        return dx, dw, db

    def predict(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Function used to predict the rating of a item W by user X

        Args:
            X (np.ndarray): Input features for user.
            W (np.ndarray): Input featurese for item.
            b (np.ndarray): Bias

        Returns:
            r_hat (np.ndarray): Prediction matrix
        """
        r_hat = np.matmul(X, W.T) + b

        return r_hat
    
    def fit(self, Y: np.ndarray, R: np.ndarray, n_features=10):
        """
        Fits the model and updates the parametrs

        Args:
            X (np.ndarray): Input features for user.
            W (np.ndarray): Input featurese for item.
            b (np.ndarray): Bias
            Y (np.ndarray): Target value
            R (np.ndarray): Rating matrix.
        """
        n_users, n_items = Y.shape

        np.random.seed(42)
        if self.X is None:
            self.X = np.random.rand(n_users, n_features) * 0.01  # Small random values
        if self.W is None:
            self.W = np.random.rand(n_items, n_features) * 0.01
        if self.b is None:
            self.b = np.zeros(n_items)


        for epoch in range(self.epochs):
            
            dx, dw, db = self._compute_gradients(X=self.X, W=self.W, b=self.b, Y=Y, R=R)
            cost = self._compute_cost(X=self.X, W=self.W, b=self.b, Y=Y, R=R)

            self.X = self.X - self.learning_rate * dx
            self.W = self.W - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            if epoch % 100 == 0:
                print(f"Curretn loss: {cost:.2f}. For epoch: {epoch}")

if __name__ == "__main__":
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
    col_fil_cls = ColaborativeFiltering(lambda_=1)
    col_fil_cls.fit( Y, R)
    X = np.random.rand(5, 3)
    W = np.random.rand(4, 3)
    b = np.zeros(4)
    r_hat = col_fil_cls.predict(X, W, b)
    print(r_hat)
