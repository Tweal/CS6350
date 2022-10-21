
from functools import reduce
from math import log2, log
from tqdm import tqdm as tqdm
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from DecisionTree.decision_tree import DecisionTree


class WeightedDecisionTree(DecisionTree):

    def generate_tree(self, data, attrs, weights):
        data['weights'] = weights
        super().generate_tree(data, attrs)

    def _calc_probs(self, data, label):
        # Get total sum of weights
        total = np.sum(data['weights'])

        grouped = data[[label, 'weights']].groupby(label)
        probs = grouped.agg(np.sum)
        # Normalize by total weight
        probs = probs.apply(lambda x: x / total)

        # Awkwardness is just to get from pd data frame to dict
        return probs['weights'].to_dict()

    def _find_mode(self, data):
        label_vals = data[self.label].unique()
        # Need to map labels to -1 and 1
        mapped = [1 if label == label_vals[0] else -1
                  for label in data[self.label]]
        # take sign of sum weights multiplied by mapped value
        sign = np.sign(sum([v1 * v2 for v1, v2
                            in zip(mapped, data['weights'])]))
        # map back to values
        weighted_mode = label_vals[0] if sign == 1 else label_vals[1]
        return weighted_mode

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
            attr_weights = {value: np.sum(subset['weights']) / len(root.data)
                            for value, subset
                            in attr_dict.items()}
            weighted_purities = [weight * purity
                                 for weight, purity
                                 in zip(attr_weights.values(),
                                        attr_purity.values())]
            gains[attr] = full_purity - sum(weighted_purities)

        return max(gains, key=gains.get)

    def _calc_weighted_error(self, data):
        # Classify each example
        data['h'] = self.classify(data)
        # 1 if correct prediction, -1 if incorrect
        data['yh'] = data.apply(lambda row: 1 if row[self.label] == row['h']
                                else -1, axis=1)
        w = data[data.yh == -1]['weights']
        # Weighted error is sum of wrong predictions
        return np.sum(w)

    def calc_alpha(self, data):
        err = self._calc_weighted_error(data)
        self.alpha = 0.5 * log((1 - err) / err)

    def calc_new_weights(self, data):
        weights = data['weights'] * np.exp(-self.alpha * data['yh'])
        # Normalize weights
        weights = weights / np.sum(weights)
        return weights


class ADABoost:
    def __init__(self, og_data, attrs, label):
        self.trees = []
        self.label_values = og_data[label].unique()

        # Initial weights to 1/m
        self.weights = np.ones(len(og_data)) / len(og_data)
        self.data = og_data
        self.attrs = attrs
        self.label = label

    def run(self, T=1000):
        for _ in tqdm(range(1, T + 1)):
            self.run_single()

    def run_single(self):
        data = self.data.copy()
        # Generate stump
        self.trees.append(WeightedDecisionTree(self.label, max_depth=1))
        self.trees[-1].generate_tree(data, self.attrs, self.weights)
        # Calculate alpha
        self.trees[-1].calc_alpha(data)
        # Update weights
        self.weights = self.trees[-1].calc_new_weights(data)

    def classify(self, entry):
        if (type(entry) is not pd.DataFrame):
            return self._classify_single(entry)
        return entry.apply(lambda row: self._classify_single(row), axis=1)

    def _classify_single(self, entry):
        # We need to map the classification to -1 or 1
        preds = [-1 if tree.classify(entry) == self.label_values[0] else 1
                 for tree in self.trees]
        # Sum tree votes
        results = [tree.alpha * pred for tree, pred in zip(self.trees, preds)]
        # get sign of the sum
        sign = np.sign(sum(results))
        # map that back to values
        return self.label_values[0] if sign == -1 else self.label_values[1]

    def calc_err(self, data):
        acc = data.apply(lambda row: row[self.label] == row['pred'], axis=1)
        acc = acc.sum() / len(data)
        return round((1 - acc), 3)

    '''Returns a list of errors for each tree
    '''
    def get_stump_errs(self, data):
        stump_preds = data.apply(lambda row: [tree.classify(row)
                                              for tree in self.trees],
                                 axis=1)
        stump_preds = pd.DataFrame(stump_preds.to_list())
        stump_crct = stump_preds.apply(lambda row: data[self.label] == row)
        stump_acc = stump_crct.agg(sum) / len(data)
        stump_err = 1 - stump_acc
        return stump_err.to_list()


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
    test = {'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'y': 0}

    ada = ADABoost(data, attrs, 'y')
    ada.run(T=10)
    print(ada._classify_single(test))
    print(ada.classify(data))
