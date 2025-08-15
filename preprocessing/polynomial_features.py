"""This module is used to transform features into theri polynomial degree form."""
from itertools import product
import numpy as np

class PolynomialFeatures:
    """
    This class is used to transform a set of features into their polynomial degree form.
    """

    def __init__(self, degree: tuple | int, interaction_only=False, include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_input_features = None
        self.combinations = None

    def fit(self, X: np.ndarray):
        """
        Fits the model. Calculates combinations + gets n_features.

        Args:
            X (np.ndarray): Input features.

        Raises:
            TypeError: Raised when self.degree is inputed in wrong format.

        Returns:
            PolynomialFeatures: The fitted instance.
        """

        if not isinstance(self.degree, tuple) and not isinstance(self.degree, int):
            raise TypeError("Wrong degree type inputed!")

        self.n_input_features = X.shape[1]

        min_val = 0
        max_val = self.degree
        if isinstance(self.degree, tuple):
            min_val = self.degree[0]
            max_val = self.degree[1]

        unfiltered_combinations = product(range(min_val, max_val + 1), repeat=self.n_input_features)

        filterd_combinations  = list(filter(lambda d: sum(d) <= max_val, unfiltered_combinations))

        filterd_combinations.sort(key=lambda x: (sum(x), tuple(reversed(x))))

        self.combinations = filterd_combinations

        print(self.combinations)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the features into their polynomials.

        Args:
            X (np.ndarray): Input features.

        Raises:
            ValueError: Raised when model not fitted or when n_input_features != X.shape[1].

        Returns:
            np.ndarray: Transformed polynomial features.
        """

        if not hasattr(self, "combinations") or not hasattr(self, "n_input_features"):
            raise ValueError("Call fit() before transform().")

        if X.shape[1] != self.n_input_features:
            raise ValueError(f"Expected {self.n_input_features} features, got {X.shape[1]}.")

        return np.hstack([
            np.prod(X ** np.array(comb), axis=1).reshape(-1, 1)
            for comb in self.combinations
            ])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits and transforms the model

        Args:
            X (np.ndarray): Input features

        Returns:
            np.ndarray: Transformed polynomial features.
        """
        self.fit(X=X)
        return self.transform(X=X)

if __name__ == "__main__":
    X = np.arange(6).reshape(3, 2)

    poly = PolynomialFeatures(5, interaction_only=True)
    poly.fit(X=X)
