import numpy as np

class PolynomialFeatures:

    def __init__(self, degree: tuple | int, interaction_only=False, include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_input_features = None
    
    def fit(self, X: np.ndarray, Y = None):

        if not isinstance(self.degree, tuple) and not isinstance(self.degree, int):
            raise TypeError("Wrong degree type inputed!")

        self.n_input_features = X.shape[1]
        print(self.n_input_features)
        return self

    def transform(self, X: np.ndarray):
        
        if self.n_input_features is None:
            print("You have to fit the model first!")
            return

        min_val = 0
        max_val = self.degree
        if isinstance(self.degree, tuple):
            min_val = self.degree[0]
            max_val = self.degree[1]
        
        for i in range(min_val, max_val):
            ...

    def fit_transform(self, X: np.ndarray):
        self.fit(X=X)
        return self.transform(X=X)

if __name__ == "__main__":
    X = np.arange(6).reshape(3, 2)

    poly = PolynomialFeatures(3)
    poly.fit(X=X)