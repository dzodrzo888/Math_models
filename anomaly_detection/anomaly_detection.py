import numpy as np

class GaussianAnomalyDetection:

    def __init__(self):
        self.epsilon = None
    
    def _estimate_gaussian(self, X: np.array):

        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        return mu, var
    
    def _multivariate_gaussian(self, X: np.array, mu: np.array, var: np.array) -> np.array:
        
        exponent = -0.5 * np.square(X - mu) / var
        scaling = 1 / np.sqrt(2 * np.pi * var)
        probs = np.prod(scaling * np.exp(exponent), axis=1)

        return probs
    
    def select_threshold(self, y_val: np.array, prob_val: np.array) -> float:
        ...

    def fit(self, X: np.array):
        ...

    def predict(self, X: np.array):
        ...


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(300, 2)
    gaussian_cls = GaussianAnomalyDetection()
