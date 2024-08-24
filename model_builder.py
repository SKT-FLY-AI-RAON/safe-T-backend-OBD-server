import numpy as np
import pandas as pd
class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=100):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        self.c = self.tree_avg_length(self.sample_size)

    def fit(self, X: np.ndarray, improved=False):
        if isinstance(X, pd.DataFrame):
            X = X.values
        height_limit = int(np.ceil(np.log2(self.sample_size)))

        for n in range(self.n_trees):
            sample_rows = np.random.choice(X.shape[0], size=self.sample_size, replace=False)
            tree = IsolationTree(height_limit)
            tree.fit(X[sample_rows], improved)
            self.trees.append(tree)

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        paths = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            paths += tree.path_length(X)

        paths /= len(self.trees)
        return paths

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return 2 ** - (self.path_length(X) / self.c)

    def predict_from_anomaly_scores(self, scores, threshold):
        return (scores >= threshold).astype(int)

    def tree_avg_length(self, n):
        if n > 2:
            return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)
        elif n == 2:
            return 1
        return 0


class inNode:
    def __init__(self, left, right, q, p):
        self.left = left
        self.right = right
        self.q = q
        self.p = p


class exNode:
    def __init__(self, size):
        self.size = size


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None

    def fit(self, X: np.ndarray, improved=False):
        self.root = self._fit(X, 0, improved)
        return self

    def _fit(self, X, current_height, improved):
        if current_height >= self.height_limit or X.shape[0] <= 1:
            return exNode(X.shape[0])

        q = np.random.randint(0, X.shape[1])
        p = np.random.uniform(X[:, q].min(), X[:, q].max())

        left_mask = X[:, q] < p
        right_mask = ~left_mask

        return inNode(
            left=self._fit(X[left_mask], current_height + 1, improved),
            right=self._fit(X[right_mask], current_height + 1, improved),
            q=q, p=p
        )

    def path_length(self, X: np.ndarray) -> np.ndarray:
        path_lengths = np.zeros(X.shape[0], dtype=float)

        for i in range(X.shape[0]):
            node = self.root
            current_length = 0
            while isinstance(node, inNode):
                current_length += 1
                if X[i, node.q] < node.p:
                    node = node.left
                else:
                    node = node.right
            path_lengths[i] = current_length + self._c_factor(node.size)

        return path_lengths

    def _c_factor(self, size):
        if size > 2:
            return 2 * (np.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)
        elif size == 2:
            return 1
        return 0
