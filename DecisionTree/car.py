import pandas as pd
import sys
sys.path.append('..')
import DecisionTree.decision_tree as dt

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

columns = ['buying', 'maint', 'doors', 'persons',
           'lug_boot', 'safety', 'label']

train_data = pd.read_csv('../data/car/train.csv', names=columns)

test_data = pd.read_csv('car/test.csv', names=columns)

attrs = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

label = 'label'

train_acc = [[0 for x in range(6)] for y in range(3)]
test_acc = [[0 for x in range(6)] for y in range(3)]


def calc_acc(data):
    acc = data.apply(lambda row: row[label] == row['pred'], axis=1)
    acc = acc.sum() / len(data)
    return round(acc, 3)


for select_method in range(3):
    for max_depth in range(6):
        tree = dt.DecisionTree(label, select_method, max_depth + 1)
        tree.generate_tree(train_data, attrs)

        train_data['pred'] = tree.classify(train_data)
        train_acc[select_method][max_depth] = calc_acc(train_data)

        test_data['pred'] = tree.classify(test_data)
        test_acc[select_method][max_depth] = calc_acc(test_data)

methods = ['Entropy', 'Max Error', 'Gini Index']
print('Car.py')

print('Training accuracy:')
print(pd.DataFrame(train_acc, columns=range(1, 7), index=methods))
print()
print('Test accuracy:')
print(pd.DataFrame(test_acc, columns=range(1, 7), index=methods))
print('\n\n\n')
