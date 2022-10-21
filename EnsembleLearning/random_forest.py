from tqdm import tqdm as tqdm
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from DecisionTree.decision_tree import DecisionTree


class RandomTree(DecisionTree):
    def generate_tree(self, data, attrs, subset_size):
        self.subset_size = subset_size
        return super().generate_tree(data, attrs)

    def _find_best_attr(self, root, full_purity):
        # Identify individual attribute purities
        attrs = list(root.attrs.keys())
        # Only subset attrs if we have enough attrs to do so
        if (len(attrs) > self.subset_size):
            attrs = np.random.choice(attrs, self.subset_size,
                                     replace=False)
        gains = {}
        for attr in attrs:
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


class RandomForest:
    def __init__(self, data, attrs, label, subset_size):
        self.data = data
        self.attrs = attrs
        self.label = label
        self.label_values = data[label].unique()
        self.subset_size = subset_size

        self.trees = []

    def run(self, T=500, keep_output=False):
        for _ in tqdm(range(1, T + 1), position=1, leave=keep_output):
            self.run_single()

    def run_single(self):
        self.trees.append(RandomTree(self.label, max_depth=np.inf))
        self.trees[-1].generate_tree(self.data, self.attrs, self.subset_size)

    def classify(self, entry):
        if (type(entry) is not pd.DataFrame):
            return self._classify_single(entry)
        return entry.apply(lambda row: self._classify_single(row), axis=1)

    def _classify_single(self, entry):
        # We need to map the classification to 0 or 1
        preds = [1 if tree.classify(entry) == self.label_values[0] else 0
                 for tree in self.trees]
        # average tree predictions and round
        result = round(np.mean(preds))
        # map back to label values
        return self.label_values[0] if result else self.label_values[1]

    def calc_err(self, data):
        acc = data.apply(lambda row: row[self.label] == row['pred'], axis=1)
        acc = acc.sum() / len(data)
        return round((1 - acc), 3)


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

    rt = RandomTree('y', np.inf)
    rt.generate_tree(data, attrs, 2)
    print(rt.classify(test))
