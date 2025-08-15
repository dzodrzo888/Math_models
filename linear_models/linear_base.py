from abc import ABC, abstractmethod
import numpy as np

class LinearBaseModel(ABC):

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, ridge=None, lasso=None):
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

    @abstractmethod
    def _initialize_parameters(self, n_features: int):
        pass

    @abstractmethod
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        pass

    @abstractmethod
    def _compute_loss(self, y: np.ndarray, y_hat: np.ndarray):
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass
