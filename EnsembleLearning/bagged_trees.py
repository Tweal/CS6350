from tqdm import tqdm as tqdm
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from DecisionTree.decision_tree import DecisionTree


class BaggedTrees:
    def __init__(self, data, attrs, label):
        self.data = data
        self.attrs = attrs
        self.label = label
        self.label_values = data[label].unique()

        self.trees = []

    def run(self, T=500, keep_output=False):
        for _ in tqdm(range(1, T + 1), position=1, leave=keep_output):
            self.run_single()

    def run_single(self):
        '''sample data'''
        sample = self.data.sample(frac=1, replace=True).reset_index()
        self.trees.append(DecisionTree(self.label, max_depth=np.inf))
        self.trees[-1].generate_tree(sample, self.attrs)

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

    bt = BaggedTrees(data, attrs, 'y')
    bt.run(10)

    print(bt._classify_single(test))
    print(bt.classify(data))
