import numpy as np

class Kmeans:
    """
    This class is used to perform K-means clustering
    """

    def __init__(self, k: int):
        self.k = k
        self.centroids = np.random.rand(k, 2)

    def _assign_clusters(self, X: np.array):
        m = X.shape[0]
        labels = np.zeros(m)

        for i in range(m):
            best_l2 = np.inf
            best_centr = 0

            for j in range(self.k):
                curr_l2 = np.linalg.norm(X[i] - self.centroids[j])

                if curr_l2 < best_l2:
                    best_l2 = curr_l2
                    best_centr = j

            labels[i] = best_centr
        
        return labels
    
    def _update_centroids(self, X: np.array ,labels: np.array):
        """
        Updates centroid values.

        Args:
            X (np.array): Input features.
            labels (np.array): _description_
        """

        for i in range(self.k):
            curr_points = X[labels == i]
            self.centroids[i] = np.mean(curr_points, axis=0)
    
    def get_centroids(self) -> np.array:
        """
        Returns centroids

        Returns:
            np.array: Centroids.
        """
        return self.centroids
        

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(10, 2)
    kmeans_cls = Kmeans(k=3)
    idx = kmeans_cls._assign_clusters(X)
    kmeans_cls._update_centroids(X, idx)
    print(kmeans_cls.centroids)