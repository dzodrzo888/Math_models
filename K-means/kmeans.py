import numpy as np

class Kmeans:
    """
    This class is used to perform K-means clustering
    """

    def __init__(self, k: int, max_iter: int):
        self.k = k
        self.centroids = None
        self.max_iter = max_iter

    def _assign_clusters(self, X: np.array, centroids: np.array) -> np.array:
        """
        Assigns clusters to points

        Args:
            X (np.array): Input features

        Returns:
            labels( np.array): Labels of points.
        """
        m = X.shape[0]
        labels = np.zeros(m)

        for i in range(m):
            best_l2 = np.inf
            best_centr = 0

            for j in range(self.k):
                curr_l2 = np.linalg.norm(X[i] - centroids[j])

                if curr_l2 < best_l2:
                    best_l2 = curr_l2
                    best_centr = j

            labels[i] = best_centr
        
        return labels
    
    def _update_centroids(self, X: np.array ,labels: np.array, centroids: np.array):
        """
        Updates centroid values.

        Args:
            X (np.array): Input features.
            labels (np.array): Labels of points.
        """

        for i in range(self.k):
            curr_points = X[labels == i]
            centroids[i] = np.mean(curr_points, axis=0)
    
    def get_centroids(self) -> np.array:
        """
        Returns centroids

        Returns:
            np.array: Centroids.
        """
        return self.centroids

    def fit(self, X: np.array):
        """
        Fits the model. Updates centroid values

        Args:
            X (np.array): Input features
        """
        
        self.centroids = np.random.rand(self.k, X.shape[1])

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X=X, centroids=self.centroids)
            self._update_centroids(X=X, labels=labels, centroids=self.centroids)
    
    def predict(self, X: np.array):
        """
        Predicts new input features.

        Args:
            X (np.array): Input features

        Raises:
            ValueError: Value error raised when centroids were not initialized

        Returns:
            labels (np.array): Cls labels.
        """

        if self.centroids is None:
            raise ValueError("Model needs to be fitted first!")
        
        labels = self._assign_clusters(X=X, centroids=self.centroids)

        return labels

        

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(10, 2)
    kmeans_cls = Kmeans(k=3, max_iter=10)
