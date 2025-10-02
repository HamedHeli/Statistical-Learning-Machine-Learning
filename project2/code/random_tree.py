from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            random_tree = RandomTree(self.max_depth)
            random_tree.fit(X,y)
            self.trees.append(random_tree)


    def predict(self, X_pred):
        t = X_pred.shape[0]
        y_hat = np.zeros((t, self.num_trees))
        y_hat_agg = np.zeros(t)
        for i in range(self.num_trees):
            y_hat[:,i] = self.trees[i].predict(X_pred)
        for i in range(t):
            y_hat_agg[i] = utils.mode(y_hat[i, :])
        return y_hat_agg
 