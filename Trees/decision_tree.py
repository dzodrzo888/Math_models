import numpy as np

class Tree:
    "This class is used to compute a recursive decisiont tree"
    
    def compute_entropy(self, y):
        """
        Computes entropy

        Args:
            y (np.array): Target values

        Returns:
            entropy (float): Entropy value
        """

        if len(y) == 0:
            return 0

        entropy = 0.

        p_1 = np.count_nonzero(y == 1) / len(y)

        if p_1 == 0 or p_1 == 1:
            return 0
        
        entropy = -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)

        return entropy

    def split_dataset(self, X: np.array, node_indices: list[int], feature=0):
        """
        Splits dataset into a right and left tree branch

        Args:
            X (np.array): Input features.
            node_indices (list[int]): List of current input values.
            feature (int, optional): Selected feature. Defaults to 0.

        Returns:
            left_branch(list): Left branch of the node.
            right_branch(list): Right branch of the node.
        """

        left_branch = []
        right_branch = []

        for index, _ in enumerate(node_indices):

            if X[node_indices[index]][feature] == 1:
                left_branch.append(node_indices[index])
            else:
                right_branch.append(node_indices[index])

        return left_branch, right_branch
    
    def compute_information_gain(self, X: np.array, y: np.array, node_indices: list[int], feature=0):
        """
        Computes information gain

        Args:
            X (np.array): Input features.
            y (np.array): Target values
            node_indices (list[int]): List of current input values.
            feature (int, optional): Current feature. Defaults to 0.

        Returns:
            information_gain(float): Information gain result.
        """

        information_gain = 0.
        left_indices, right_indices = self.split_dataset(X, node_indices, feature)

        y_root =  y[node_indices]
        y_left = y[left_indices]
        y_right = y[right_indices]

        left_weight = len(y_left) / len(y_root)
        right_weight = len(y_right) / len(y_root)

        information_gain = self.compute_entropy(y_root) - (left_weight * self.compute_entropy(y_left) + right_weight * self.compute_entropy(y_right))

        return information_gain

    def find_best_split(self, X: np.array, y: np.array, node_indices: list[int]):
        """
        Finds the best feature to split. If not found returns -1.

        Args:
            X (np.array): Input features.
            y (np.array): Target values
            node_indices (list[int]): List of current input values.

        Returns:
            int: Feature with largest info_gain. If not found returns -1.
        """

        if len(np.unique(y[node_indices])) == 1:
            return -1
        
        num_features = X.shape[1]

        best_feature = -1
        best_info_gain = -1

        for feature in num_features:
            information_gain = self.compute_information_gain(X, y, node_indices, feature)

            if information_gain > best_info_gain:
                best_info_gain = information_gain
                best_feature = feature
        
        return best_feature

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.randint(2, size=(10, 3))
    y_train = np.random.randint(2, size=10)
    root_indices = list(range(10))

    tree_cls = Tree()
