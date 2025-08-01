import numpy as np
import pandas as pd

class PCA:

    def __init__(self, k_vectors: int = 5):
        self.k_vectors = k_vectors
        self.eigenvalues = None
        self.eigenvectors = None
        self.X_centered = None
        self.W = None
        self.X_u = None

    def _fit(self, X: pd.DataFrame):
        """
        Fits the model on the data

        Args:
            X (pd.DataFrame): Input features.
        """
        self.X_u = np.mean(X.values, axis=0)

        self.X_centered = X - self.X_u

        cov_matrix = np.cov(self.X_centered, rowvar=False)

        self.eigenvalues, self.eigenvectors = np.linalg.eig(cov_matrix)

    def _transform(self) -> np.ndarray:
        """
        Transforms the matrix based on the learned features.

        Returns:
            np.ndarray: Transformed matrix
        """
        idx_sort = self.eigenvalues.argsort()

        eigenvectors_sort = self.eigenvectors[:, idx_sort[::-1]]

        self.W = eigenvectors_sort[:, :self.k_vectors]

        Z = np.dot(self.X_centered, self.W)

        return Z
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fits and transforms the data.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Transformed matrix
        """
        self._fit(X=X)
        Z = self._transform()

        return Z
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Transforms the matrix back to original values based on the learned eigenvalues.

        Args:
            Z (np.ndarray): Input matrix.

        Returns:
            np.ndarray: Predicted original matrix.
        """
        X_centered_hat = np.dot(Z, self.W.T)

        X_hat = X_centered_hat + self.X_u

        return X_hat
    
    def explained_varience(self, index:int) -> np.ndarray:
        """
        Returns the eigenvectors by index row.

        Args:
            index (int): Index

        Returns:
            np.ndarray: Vector row.
        """
        return self.eigenvectors[index]
    
    def explained_variance_ratio(self, index: int) -> np.ndarray:
        """
        Return ratio of the eigenvectors by index row.

        Args:
            index (int): Index.

        Returns:
            np.ndarray: Ratioed vector row.
        """
        return self.eigenvectors[index] / np.sum(self.eigenvectors)

if __name__ == "__main__":

    data = {
    'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
    'Weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
    'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
    'Gender': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  # 1 = Male, 0 = Female
    }
    df = pd.DataFrame(data)
    X = df.drop('Gender', axis=1)
    y = df["Gender"]

    pca_cls = PCA()

    z = pca_cls.fit_transform(X)

    x_hat = pca_cls.inverse_transform(z)
    print(type(x_hat))
    print(type(pca_cls.explained_variance_ratio(0)))