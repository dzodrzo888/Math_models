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

        p_1 = np.count_nonzero(y == 1)

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
            _type_: _description_
        """

        left_branch = []
        right_branch = []
        print(f"This is the X value {X}. This is the indices: {node_indices}, This is the feature {feature}")
        for index, _ in enumerate(node_indices):

            if X[node_indices[index]][feature] == 1:
                left_branch.append(node_indices[index])
            else:
                right_branch.append(node_indices[index])

        return left_branch, right_branch

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.randint(2, size=(10, 3))
    y_train = np.random.randint(2, size=10)
    root_indices = list(range(10))

    tree_cls = Tree()