import numpy as np
import utils


class DecisionStumpEquality:
    """
    This is a decision stump that branches on whether the value of X is
    "almost equal to" some threshold.

    This probably isn't a thing you want to actually do, it's just an example.
    """

    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])

                # Find most likely class for each split
                is_almost_equal = np.round(X[:, j]) == t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


class DecisionStumpErrorRate:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None
    
    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = X[i, j]

                # Find most likely class for each split
                is_more = X[:, j] > t
                y_yes_mode = utils.mode(y[is_more])
                y_no_mode = utils.mode(y[~is_more])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] <= t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode
    def predict(self, X):
        n, d = X.shape

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] > self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


class DecisionStumpInfoGain(DecisionStumpErrorRate):
    # This is not required, but one way to simplify the code is
    # to have this class inherit from DecisionStumpErrorRate.
    # Which methods (init, fit, predict) do you need to overwrite?
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None


        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        maxGain = 0

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = X[i, j]

                # Find most likely class for each split
                is_more = X[:, j] > t
                y_yes_mode = utils.mode(y[is_more])
                y_no_mode = utils.mode(y[~is_more])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] <= t] = y_no_mode

                # Compute error
                count_yes = np.bincount(y[is_more], minlength = len(count))
                n_yes = len(y[is_more]) ## n_less is always positive  
                count_no = np.bincount(y[~is_more], minlength = len(count))
                n_no = len(y[~is_more]) 
                if n_yes == 0 or n_no == 0:
                    continue 
                # Find the gain for each possible split
                gain = entropy(count/n) - n_no/n*entropy(count_no/n_no) - n_yes/n*entropy(count_yes/n_yes)

                # Compare to minimum error so far
                if gain > maxGain:
                    # This is the lowest error, store this value
                    maxGain = gain
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode


## Hard-coded Predict Function
def predict_hard_coded(X):
    n, d = X.shape
    y_hat = np.zeros(n)
    for i in range(n):
        if X[:,0] > -80.30510 and X[:,1] > 36.453576:
            y_hat[i] = 0
        elif X[:,0] > -80.30510 and X[:,1] <= 36.453576:
            y_hat[i] = 1
        elif X[:,0] <= -80.30510 and X[:,1] > 37.669007:
            y_hat[i] = 0
        elif X[:,0] <= -80.30510 and X[:,1] <= 37.669007:
            y_hat[i] = 1
    return y_hat

