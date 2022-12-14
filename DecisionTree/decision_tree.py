from functools import reduce
from math import log2
import pandas as pd
import numpy as np
import copy


class Node:
    def __init__(self, is_leaf=False, label=None):
        # Data matching this branch
        self.data = None
        # The value used to split parent to this node
        self.value = None
        # A dict of available attributes and their possible values
        self.attrs = None
        # Best attribute to split on
        self.best_attr = None

        self.children = {}
        self.depth = -1
        self.is_leaf = is_leaf

        # Label for if this is a leaf
        self.label = label


class DecisionTree:

    '''Selection method is expected to be an integer corresponding as follows:
        0: Entropy
        1: Majority Error
        2: Gini Index
    '''
    def __init__(self, label: str, max_depth: int, selection_method=0):
        self.label = label
        self.method = [self._calc_H, self._calc_ME,
                       self._calc_GI][selection_method]
        self.max_depth = max_depth
        self.root = None

    '''Generates the decision tree using the current ID3 parameters.
        NOTE: This function does not return anything, it saves the root of the
        tree into self.root!

        Required Inputs
        ------
        data: pandas dataframe
            A dataframe consisting of the training data to generate the tree.
            The columns are expected to be the attributes
        attrs: dict
            A dictionary of attributes and a list of their possible values
    '''
    def generate_tree(self, data, attrs):
        self.root = Node()
        self.root.depth = 0
        self.root.data = data
        self.root.attrs = attrs

        stack = [self.root]

        while len(stack) > 0:
            cur = stack.pop(0)
            cur.children = self.ID3(cur)
            stack.extend(cur.children.values())

    '''Function to process nodes of the tree. Returns the children if existant
        Note: This function is not recursive and relies on a stack
        implemented in the helper function generate_tree.
    '''
    def ID3(self, root):
        # Base case already leaf, therefore it'll error when calculating purity
        if (root.is_leaf):
            return {}

        full_probs = self._calc_probs(root.data, self.label)
        full_purity = self.method(full_probs.values())

        # More base cases; pure, max depth, no attrs left
        if (full_purity == 0
                or len(root.attrs) == 0
                or root.depth == self.max_depth):
            root.is_leaf = True
            # Set label to most common value
            root.label = self._find_mode(root.data)
            return {}

        root.best_attr = self._find_best_attr(root, full_purity)
        child_attrs = copy.deepcopy(root.attrs)
        child_attrs.pop(root.best_attr, None)

        # For each attr value create child
        for value in root.attrs[root.best_attr]:
            subset = root.data[root.data[root.best_attr] == value]

            child = Node()
            child.depth = root.depth + 1
            child.attrs = child_attrs
            child.value = value
            child.data = subset

            if (len(subset) == 0):
                child.is_leaf = True
                child.label = self._find_mode(root.data)

            root.children[value] = child

        return root.children

    '''This is so I can overload it in children for weighted
        mode() returns a list so 0 index to get first.
    '''
    def _find_mode(self, data):
        return data[self.label].mode()[0]

    def _find_best_attr(self, root, full_purity):
        # Identify individual attribute purities
        gains = {}
        for attr in root.attrs:
            attr_dict = {value: subset
                         for value, subset
                         in root.data.groupby(attr)}
            attr_probs = {value: self._calc_probs(subset, self.label)
                          for value, subset
                          in attr_dict.items()}
            attr_purity = {value: self.method(probs.values())
                           for value, probs
                           in attr_probs.items()}
            attr_weights = {value: len(subset) / len(root.data)
                            for value, subset
                            in attr_dict.items()}
            weighted_purities = [weight * purity
                                 for weight, purity
                                 in zip(attr_weights.values(),
                                        attr_purity.values())]
            gains[attr] = full_purity - sum(weighted_purities)

        return max(gains, key=gains.get)

    def classify(self, entry):
        if (type(entry) is not pd.DataFrame):
            return self._classify_single(entry)
        return entry.apply(lambda row: self._classify_single(row), axis=1)

    def _classify_single(self, entry):
        node = self.root
        while not node.is_leaf:
            node = node.children[entry[node.best_attr]]
        return node.label

    '''Loops over all attributes and if continious converts them to binary.
        The method of conversion is by splitting at the median.
        It should be noted that this will modify the passed in data.
        Inputs
        ------
        data: Pandas DataFrame
            A dataframe containing all data
        cont_attrs: list
            A subset of the columns of data to be disretized
    '''
    def discretize_data(data, cont_attrs):
        for c in cont_attrs:
            median = data[c].median()
            data[c] = data[c].apply(lambda x: int(x < median))

    '''Replaces missing value, val, with majority from train_data.
        val defaults to 'unkown' but can any value can be passed to replace.
    '''
    def fill_missing(data, train_data, val='unknown'):
        for attr in data.columns:
            counts = list(train_data[attr].value_counts().index)
            # This if is needed in case val is the most common
            if counts[0] != val:
                majority = counts[0]
            else:
                majority = counts[1]
            data[attr] = data[attr].apply(lambda x: majority if x == val
                                          else x)
        print()

    '''Returns a dict of the probabilities of label values for the data
        Inputs
        -------
        data: Pandas dataframe
            A dataframe consisting of one entry per row
        label: String
            A string corresponding to one of the columns of the dataframe
    '''
    def _calc_probs(self, data, label):
        return dict(data[label].value_counts().map(lambda x: x / len(data)))

    '''Calculates the Entropy, H, of a set of probabilites, S.
    '''
    def _calc_H(self, S):
        if round(sum(S), 3) != 1:
            raise Exception('Something went wrong,' +
                            ' probabilities do not sum to 1')
        if 0 in S:
            return 0
        return - reduce(lambda cur, next: cur + next * log2(next), S, 0.0)

    '''Calculates the Majority Error, ME, of a set of probabilites, S.
    '''
    def _calc_ME(self, S):
        if round(sum(S), 3) != 1:
            raise Exception('Something went wrong,' +
                            ' probabilities do not sum to 1')
        return 1 - max(S)

    '''Calculates the Gini Index, GI, of a set of probabilites, S.
    '''
    def _calc_GI(self, S):
        if round(sum(S), 3) != 1:
            raise Exception('Something went wrong,' +
                            ' probabilities do not sum to 1')
        return 1 - reduce(lambda cur, next: cur + next**2, S, 0)


if __name__ == '__main__':
    data = pd.DataFrame(data={
        'x1': [0, 0, 0, 1, 0, 1, 0],
        'x2': [0, 1, 0, 0, 1, 1, 1],
        'x3': [1, 0, 1, 0, 1, 0, 0],
        'x4': [0, 0, 1, 1, 0, 0, 1],
        'y': [0, 0, 1, 1, 0, 0, 0]
    })
    attrs = {
        'x1': [0, 1],
        'x2': [0, 1],
        'x3': [0, 1],
        'x4': [0, 1],
    }
    tree = DecisionTree('y', 2)
    tree.generate_tree(data, attrs)
    print(tree._classify_single({'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'y': 0}))
    print(tree.classify(data))
