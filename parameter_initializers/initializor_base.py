from abc import ABC, abstractmethod
import numpy as np

class InitializerBaseModel(ABC):
    """
    Base class to abstract initialization methods

    Attributes:
        weights (np.ndarray): Weights for models.
        bias (float): Bias for models.
    """

    def __init__(self):
        self.weights = np.array([0])
        self.bias = 0.0

    @abstractmethod
    def initialize_parameters(self, X: np.ndarray):
        pass
