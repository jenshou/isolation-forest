import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        self.c = compute_c(self.sample_size)

    def fit(self, X: np.ndarray, improved=False):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if improved:
            hlim = math.ceil(math.log(self.sample_size, 2))
            tree_hlim = [[X[np.random.choice(len(X), self.sample_size)], hlim] for i in range(self.n_trees)]
            p = Pool(5)
            self.trees = p.starmap(IsolationTree, tree_hlim)
            return self
        else:
            p = Pool(5)
            self.trees = p.starmap(IsolationTree, tree_hlim)
            return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        paths = np.array([Path(x, y).path for x in X for y in self.trees]).reshape(len(X), -1)
        mean_result = paths.mean(axis=1)
        return mean_result


    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        h_x = self.path_length(X)
        anomaly_score = 2 ** ((-h_x) / self.c)
        return anomaly_score

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        return np.array([1 if score >= threshold else 0 for score in scores])

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


class IsolationTree:
    def __init__(self, X, height_limit, current_height=0):
        self.X = X
        self.current_height = current_height
        self.size = len(X)
        self.height_limit = height_limit
        self.Q = np.arange(X.shape[1], dtype='int')
        self.SplitAtt = None  
        self.SplitVal = None  
        self.root = self.growTree(X, hlim=self.height_limit)
        self.n_nodes = self.nodes(self.root)

    def growTree(self, X, hlim=100, current_height=0, improved=False):
        self.current_height = current_height
        if current_height >= hlim or len(X) <= 1:
            return Node(X, self.SplitAtt, self.SplitVal, current_height,
                        left=None, right=None, node_type='exNode')
        else:
            self.SplitAtt = np.random.randint(np.shape(X)[1],size=1)[0]
            if np.min(X[:, self.SplitAtt]) == np.max(X[:, self.SplitAtt]):
                return Node(X, self.SplitAtt, self.SplitVal, current_height,
                        left=None, right=None, node_type='exNode')

            self.SplitVal = np.random.uniform(min(X[:, self.SplitAtt]), max(X[:, self.SplitAtt]), 1).item()
            left_index = X[:, self.SplitAtt] < self.SplitVal
            right_index = np.invert(left_index)
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
    elif samplesize == 2:
        return 1
    else:
        return 0

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
            self.current_path = self.current_path + compute_c(t.size)
            return self.current_path
