import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
numpy.random.seed(711)
class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        # self.height_limit = math.ceil(math.log(sample_size, 2))
        # self.current_height = 0
        self.trees = []
        self.c = compute_c(self.sample_size)

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        # hlim = math.ceil(math.log(self.sample_size, 2))
        if isinstance(X, pd.DataFrame):
            X = X.values
        # tree_hlim = [[X[np.random.choice(len(X), self.sample_size)], hlim] for i in range(self.n_trees)]
        # for i in range(self.n_trees):
        #     X_sample = np.random.choice(len(X), self.sample_size, replace=False)
        #     iTree = IsolationTree(X[X_sample], hlim)
        #     self.trees.append(iTree)  #.growTree(X[X_sample]))
        # return self

        if improved:
            hlim = math.ceil(math.log(self.sample_size, 2))
            tree_hlim = [[X[np.random.choice(len(X), self.sample_size)], hlim] for i in range(self.n_trees)]
            # hlim = int(np.ceil(np.log2(self.sample_size)))
            # pairwise =  [[X[np.random.choice(len(X), self.sample_size)], hlim] for i in range(self.n_trees)]
            p = Pool(5)
            self.trees = p.starmap(IsolationTree, tree_hlim)
            return self
        else:
            # hlim = int(np.ceil(np.log2(self.sample_size)))
            # pairwise = [[X[np.random.choice(len(X), self.sample_size)], hlim] for i in range(self.n_trees)]
            p = Pool(5)
            self.trees = p.starmap(IsolationTree, tree_hlim)
            return self





    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        paths = np.array([Path(x, y).path for x in X for y in self.trees]).reshape(len(X), -1)
        # result_np = np.array(result_list).reshape(len(X), -1)
        mean_result = paths.mean(axis=1)
        return mean_result


    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        h_x = self.path_length(X)
        anomaly_score = 2 ** ((-h_x) / self.c)
        # print(f"anomalu score:{anomaly_score}")
        return anomaly_score

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.array([1 if score >= threshold else 0 for score in scores])

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."""
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


class IsolationTree:
    def __init__(self, X, height_limit, current_height=0):
        self.X = X
        self.current_height = current_height
        self.size = len(X)
        self.height_limit = height_limit
        self.Q = np.arange(X.shape[1], dtype='int')
        self.SplitAtt = None  # np.random.choice(np.arange(X.shape[1], dtype='int'), 1)
        self.SplitVal = None  # np.random.uniform(np.min(X[:, self.SplitAtt]), np.max(X[:, self.SplitAtt]), 1)
        self.root = self.growTree(X, hlim=self.height_limit)
        self.n_nodes = self.nodes(self.root)


    def growTree(self, X, hlim=100, current_height=0, improved=False):
        # self.SplitAtt = np.random.choice(np.arange(X.shape[1], dtype='int'), 1)
        # self.SplitVal = np.random.uniform(np.min(X[:, self.SplitAtt]), np.max(X[:, self.SplitAtt]), 1)
        # n_node not adding up node counts
        self.current_height = current_height
        if current_height >= hlim or len(X) <= 1:
            # self.n_nodes += 1
            return Node(X, self.SplitAtt, self.SplitVal, current_height,
                        left=None, right=None, node_type='exNode')
        else:
            # q = np.random.choice(np.arange(X.shape[1], dtype='int'), 1)
            self.SplitAtt = np.random.randint(np.shape(X)[1],size=1)[0]
            if np.min(X[:, self.SplitAtt]) == np.max(X[:, self.SplitAtt]):
                # print(f"1: height_limit")
                # self.n_nodes += 1
                return Node(X, self.SplitAtt, self.SplitVal, current_height,
                        left=None, right=None, node_type='exNode')
            # else:
                # p = np.random.uniform(np.min(X[:, q]), np.max(X[:, q]), 1)
            # self.SplitVal = np.random.uniform(np.min(X[:, self.SplitAtt]), np.max(X[:, self.SplitAtt]), 1).item()
            self.SplitVal = np.random.uniform(min(X[:, self.SplitAtt]), max(X[:, self.SplitAtt]), 1).item()

            # np.median(X[:, self.splitAtt])
            # left_determine = X[:, self.SplitAtt] < self.SplitVal
            # X_left = X[X[:, self.SplitAtt] < self.SplitVal]  #[left_determine.squeeze()]
            # # right_determine = X[:, self.SplitAtt] >= self.SplitVal
            # X_right = X[X[:, self.SplitAtt] >= self.SplitVal]  #[right_determine.squeeze()]
            # w = np.where(X[:, self.SplitAtt] < self.SplitVal, True, False)
            # hlim -= 1
            left_index = X[:, self.SplitAtt] < self.SplitVal
            right_index = np.invert(left_index)
                # self.n_nodes += 1
            return Node(X, self.SplitAtt, self.SplitVal, current_height,
                            left=self.growTree(X[left_index], current_height + 1, hlim),
                            right=self.growTree(X[right_index], current_height + 1, hlim),
                            node_type='inNode')

    def nodes(self, t):
        if not t:
            return 0
        else:
            return 1 + self.nodes(t.left) + self.nodes(t.right)

class Node:
    def __init__(self, X, SplitAtt, SplitVal, e, left, right, node_type=''):
        self.e = e
        self.size = len(X)
        self.X = X  # to be removed
        self.SplitAtt = SplitAtt
        self.SplitVal = SplitVal
        self.left = left
        self.right = right
        self.ntype = node_type


def find_TPR_threshold(y, scores, desired_TPR):
    # TPR = TP / (TP + FN)
    # FPR = FP / (FP + TN)
    for t in range(100, 0, -1):
        y_pred = np.zeros_like(y)
        TP, FN, FP, TN = 0, 0, 0, 0
        for i in range(len(scores)):
            if scores[i] >= (t / 100):
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        TN, FP, FN, TP = confusion_matrix(y, y_pred).flat
        if (TP / (TP + FN)) >= desired_TPR:
            return t / 100, (FP / (FP + TN))
    return None, None


def compute_c(samplesize):
    if samplesize > 2:
        return 2 * (math.log(samplesize - 1, 2) + 0.5772156649) - (2 * (samplesize - 1) / samplesize)
        # None
    elif samplesize == 2:
        return 1
    else:
        return 0
    # return c

class Path(object):
    def __init__(self, x, t):
        self.x = x
        self.current_path = 0
        self.path = self.find_path(t.root)

    def find_path(self, t):
        while t.ntype != 'exNode':
            q = t.SplitAtt
            self.current_path += 1
            if self.x[q] < t.SplitVal:
                t = t.left
            else:
                t = t.right
        else:
            # if t.size == 1:
            #     return self.current_path
            self.current_path = self.current_path + compute_c(t.size)
            return self.current_path
