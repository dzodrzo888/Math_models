from initializor_base import InitializerBaseModel
import numpy as np

class ZeroInitialization(InitializerBaseModel):
    
    def _initialize_parameters(self, n_features: int):
        """
        Initialize parametrs (weights, bias)

        Args:
            n_features (int): Number of features
        """
        # Weights initialization
        self.weights = np.zeros(n_features)
        # Bias initialization
        self.bias = 0.0