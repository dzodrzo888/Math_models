import numpy as np

class GaussianAnomalyDetection:

    def __init__(self, epsilon=None):
        self.epsilon = epsilon
        self.mu = None
        self.var = None
    
    def _estimate_gaussian(self, X: np.array):
        """
        Estimates mean and variance

        Args:
            X (np.array): Input features

        Returns:
            mu (np.array): mean
            var (np.array): variance
        """

        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        return mu, var
    
    def _multivariate_gaussian(self, X: np.array, mu: np.array, var: np.array) -> np.array:
        """
        Computes propabilites using gaussian

        Args:
            X (np.array): Input features
            mu (np.array): mean
            var (np.array): Variance

        Returns:
            np.array: Propabilites
        """
        
        exponent = -0.5 * np.square(X - mu) / var
        scaling = 1 / np.sqrt(2 * np.pi * var)
        probs = np.prod(scaling * np.exp(exponent), axis=1)

        return probs
    
    def select_threshold(self, y_val: np.array, prob_val: np.array) -> float:
        """
        Selects the most optimal epsilon value.

        Args:
            y_val (np.array): True anomalies
            prob_val (np.array): Propabilities of a anomaly
        """
        
        best_epsilon = 0
        best_f1 = 0

        min_prob, max_prob = min(prob_val), max(prob_val)

        epsilons = np.logspace(np.log10(min_prob), np.log10(max_prob), num=1000)

        for epsilon in epsilons:
            
            predictions = prob_val < epsilon
            tp = np.sum(np.logical_and(predictions, y_val))
            fp = np.sum(np.logical_and(predictions, np.logical_not(y_val)))
            fn = np.sum(np.logical_and(np.logical_not(predictions), y_val))

            prec = tp / (tp + fp)

            rec = tp / (tp + fn)

            f1 = 2 * prec * rec / (prec + rec)

            if f1 > best_f1:
                best_f1 = f1
                best_epsilon = epsilon
        
        self.epsilon = best_epsilon

    def fit(self, X: np.array):
        """
        Fits the anomally model

        Args:
            X (np.array): Input features
        """
        
        self.mu, self.var = self._estimate_gaussian(X=X)

    def get_propabilities(self, X:np.array) -> np.array:
        """
        Returns calculated propabilites.

        Args:
            X (np.array): Input features.

        Raises:
            ValueError: Raised when model not fitted.

        Returns:
            np.array: Propabilities
        """

        if self.mu is None or self.var is None:
            raise ValueError("Fit the model before calculating propabilities!")
        
        return self._multivariate_gaussian(X=X, mu=self.mu, var=self.var)

    def predict(self, X: np.array):
        """
        Predicts if inputs are anomalies

        Args:
            X (np.array): Input features.

        Raises:
            ValueError: Raised when epsilon not set

        Returns:
            list: List of values.
        """
        
        if self.epsilon is None:
            raise ValueError("Epsilon is None! Run select_threshold or set epsilon!")
        
        probs = self._multivariate_gaussian(X=X, mu=self.mu, var=self.var)

        return (probs < self.epsilon).astype(int)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(300, 2)
