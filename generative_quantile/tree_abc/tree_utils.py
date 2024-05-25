from IPython.core.debugger import set_trace
from sklearn.tree import _tree as ctree
from collections import deque

import numpy as np

def getIndicesInHrect(points, hrect):
    """Returns the indices in points where the point lies inside hyperrectangle hrect"""
    return np.logical_and(hrect[:,0] < points, hrect[:,1] > points).all(axis=1)

class AABB:
    """Axis-aligned bounding box"""
    def __init__(self, dim, support=None):
        self.support = support
        if support is None:
            self.limits = np.array([[-np.inf, np.inf]] * dim)
        else:
            self.limits = np.array(support, dtype=float)

    def __repr__(self):
        return f"AABB: {self.limits}"

    def split(self, f, v):
        left = AABB(self.limits.shape[0], self.support)
        right = AABB(self.limits.shape[0], self.support)
        left.limits = self.limits.copy()
        right.limits = self.limits.copy()

        left.limits[f, 1] = v
        right.limits[f, 0] = v

        return left, right

class Tree:

    def __init__(self, X, y, support):
        self.support = support
        self.leaf_boundaries, self.leaf_intensities = self._get_leaf_info()

        self.leaf_acceptances = np.zeros(len(self.leaf_boundaries))
        self.leaf_rejections = np.zeros(len(self.leaf_boundaries))

        for i,hrect in enumerate(self.leaf_boundaries):
            relevant_idxs = getIndicesInHrect(X, hrect)
            self.leaf_acceptances[i] = np.sum(y[relevant_idxs] == 1)
            self.leaf_rejections[i] = np.sum(y[relevant_idxs] == 0)



    def _is_leaf(self, _):
        raise NotImplementedError("Implement in a child class")

    def _left_child(self, _):
        raise NotImplementedError("Implement in a child class")

    def _right_child(self, _):
        raise NotImplementedError("Implement in a child class")

    def _split_var(self, _):
        raise NotImplementedError("Implement in a child class")

    def _value(self, _):
        raise NotImplementedError("Implement in a child class")

    def _intensity(self, _):
        raise NotImplementedError("Implement in a child class")

    def _get_leaf_info(self):
        """Compute final decision rule for each node in tree"""

        aabbs = {k:AABB(self.dim, self.support) for k in self.node_indices}

        queue = deque([0])
        while queue:
            node_idx = queue.pop()

            if not self._is_leaf(node_idx):
                l_child_idx = self._left_child(node_idx)
                r_child_idx = self._right_child(node_idx)

                aabbs[l_child_idx], aabbs[r_child_idx] = aabbs[node_idx].split(self._split_var(node_idx), self._value(node_idx))
                queue.extend([l_child_idx, r_child_idx])

        leaf_info = [
                        (aabbs[node_idx].limits, self._intensity(node_idx))
                        for node_idx in self.node_indices if self._is_leaf(node_idx)
                    ]
        return zip(*leaf_info)

class CARTTree(Tree):
    """Wrapper for logistic CART implementation in sklearn.tree.DecisionTreeClassifier"""

    def __init__(self, X, y, tree, support):
        dim = support.shape[0]
        self._tree = tree
        self.length = tree.node_count
        self.node_indices = np.arange(self.length)
        if dim is not None:
            self.dim = np.max(tree.feature) + 1
        else:
            self.dim = dim
        super().__init__(X, y, support)

    def _is_leaf(self, node_idx):
        return self._tree.children_left[node_idx] == ctree.TREE_LEAF

    def _left_child(self, node_idx):
        return self._tree.children_left[node_idx]

    def _right_child(self, node_idx):
        return self._tree.children_right[node_idx]

    def _split_var(self, node_idx):
        return self._tree.feature[node_idx]

    def _value(self, node_idx):
        return self._tree.threshold[node_idx]

    def _intensity(self, node_idx):
        return self._tree.value[node_idx, 0, 1] / np.sum(self._tree.value[node_idx, 0])

class TreeProposal:
    """Histogram proposal distribution for importance sampling from a decision tree

    Number of bins is inherited from the number of leaves in the tree

    Attributes:
        boundaries: Endpoints of axis-aligned hyperrectangles for each bin
        probs: Probability mass belonging to each bin
    """

    support: np.ndarray

    boundaries: np.ndarray
    probs: np.ndarray

    def __init__(self, tree, support, prior, rule):
        """Initializes the instance based on dimension, likelihood model, and proposal rule

        Args:
            dim: dimensionality of the parameter space
            dtree: a decision tree (sklearn.tree.DecisionTreeClassifier) object modeling the likelihood
            support: axis-aligned hyperrectangle that is a superset of the prior support
            prior: a function that returns the prior density within the given support
            rule: a function that maps prior density and likelihood model to a proposal density
        """
        self.boundaries = np.array(tree.leaf_boundaries)
        self.support = support
        self.rule = rule

        # currently only supports uniform prior, need to modify this line to support other priors
        # would need to calculate prior mass within each leaf bin by numerical integration
        prior_probs = np.ones(len(self.boundaries))
        #
        # unnormalized_probs = rule(prior_probs, intensities)
        #
        # self.probs = unnormalized_probs / np.sum(
        #     unnormalized_probs * np.prod(self.boundaries[:, :, 1] - self.boundaries[:, :, 0], axis=1)
        # )

        self.accs = tree.leaf_acceptances
        self.rejs = tree.leaf_rejections

        # lh_model = (1+self.accs) / (2+self.accs + self.rejs)
        lh_model = (self.accs) / (self.accs + self.rejs)
        # lh_model = np.random.beta(1+self.accs, 1+self.rejs, size=len(self.accs))
        lh_model = lh_model / np.sum(lh_model)
        # print(f"lh_model: {lh_model}")
        unn_probs = rule(prior_probs, lh_model)
        # unn_probs = np.sqrt(lh_model) #DEbugging
        unn_probs = unn_probs / np.sum(unn_probs)
        # print(f"UNN_PROBS: {np.round(unn_probs, 3)}")
        denominator = np.sum(
            unn_probs * np.prod(self.boundaries[:, :, 1] - self.boundaries[:, :, 0], axis=1)
        )
        if denominator < 1e-6: #this logic helps if a near-singular measure is sampled
            self.probs = unn_probs / np.sum(unn_probs)
        else:
            self.probs = unn_probs / np.sum(
                unn_probs * np.prod(self.boundaries[:, :, 1] - self.boundaries[:, :, 0], axis=1)
            )

    def sample(self, size, returnbins=False):
        """Draws size samples from this instance's distribution"""


        n_bins = self.boundaries.shape[0]
        dim = self.support.shape[0]

        # print(f"PROBS: {self.probs}")
        # print(f"ActualThing: {self.probs * np.prod(self.boundaries[:, :, 1] - self.boundaries[:, :, 0], axis=1)}")

        bin_indices = np.random.choice(
            n_bins, size, p=self.probs * np.prod(self.boundaries[:, :, 1] - self.boundaries[:, :, 0], axis=1)
        )

        samples = np.zeros((size, dim))
        for i in range(dim):
            samples[:, i] = np.random.uniform(self.boundaries[bin_indices, i, 0], self.boundaries[bin_indices, i, 1])

        if returnbins:
            return samples, bin_indices
        else:
            return samples


    def _density(self, theta):
        bin_index = np.where(
            np.all(np.logical_and(theta > self.boundaries[:, :, 0], theta < self.boundaries[:, :, 1]), axis=1)
        )[0]

        if bin_index.shape[0] > 0:
            return self.probs[bin_index[0]]
        else:
            return 0

    def density(self, thetas):
        """Returns the density according to this instance's distribution at theta"""
        dm = np.zeros(thetas.shape[0])

        for i in range(thetas.shape[0]):
            dm[i] = self._density(thetas[i])

        return dm

    def update(self, bin, accepted, n_add=1):
        """Updates the tree probabilities"""
        if accepted:
            self.accs[bin] += n_add
        else:
            self.rejs[bin] += n_add
        #
        prior_probs = np.ones(len(self.boundaries))
        # unn_probs = (1+self.accs) / (2+self.accs + self.rejs)
        # self.probs = unn_probs / np.sum(
        #     unn_probs * np.prod(self.boundaries[:, :, 1] - self.boundaries[:, :, 0], axis=1)
        # )
        #
        # self.accs = tree.leaf_acceptances
        # self.rejs = tree.leaf_rejections
        #
        # lh_model = (1+self.accs) / (2+self.accs + self.rejs)
        lh_model = (self.accs) / (self.accs + self.rejs)
        # lh_model = np.random.beta(1+self.accs, 1+self.rejs, size=len(self.accs))
        lh_model = lh_model / np.sum(lh_model)
        # print(f"lh_model: {lh_model}")
        unn_probs = self.rule(prior_probs, lh_model)
        # unn_probs = np.sqrt(lh_model) #DEbugging
        unn_probs = unn_probs / np.sum(unn_probs)
        # print(f"UNN_PROBS: {np.round(unn_probs,3)}")
        denominator = np.sum(
            unn_probs * np.prod(self.boundaries[:, :, 1] - self.boundaries[:, :, 0], axis=1)
        )
        if denominator < 1e-6: #this logic helps if a near-singular measure is sampled
            self.probs = unn_probs / np.sum(unn_probs)
        else:
            self.probs = unn_probs / np.sum(
                unn_probs * np.prod(self.boundaries[:, :, 1] - self.boundaries[:, :, 0], axis=1)
            )

