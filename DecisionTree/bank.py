import decision_tree as dt
import pandas as pd
from tqdm import tqdm as tqdm

columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
           'housing', 'loan', 'contact', 'day', 'month', 'duration',
           'campaign', 'pdays', 'previous', 'poutcome', 'y']


train_data = pd.read_csv('bank/train.csv', names=columns)

test_data = pd.read_csv('bank/test.csv', names=columns)

cont_attrs = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays',
              'previous']
dt.DecisionTree.discretize_data(train_data, cont_attrs)
dt.DecisionTree.discretize_data(test_data, cont_attrs)

attrs = {
         'age': [0, 1],
         'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid',
                 'entrepreneur', 'student', 'blue-collar', 'self-employed',
                 'retired', 'technician', 'services'],
         'marital': ['married', 'divorced', 'single'],
         'education': ['unknown', 'secondary', 'primary', 'tertiary'],
         'default': ['yes', 'no'],
         'balance': [0, 1],
         'housing': ['yes', 'no'],
         'loan': ['yes', 'no'],
         'contact': ['unknown', 'telephone', 'cellular'],
         'day': [0, 1],
         'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                   'sep', 'oct', 'nov', 'dec'],
         'duration': [0, 1],
         'campaign': [0, 1],
         'pdays': [0, 1],
         'previous': [0, 1],
         'poutcome': ['unknown', 'other', 'failure', 'success']
        }
label = 'y'

train_acc = [[0 for x in range(16)] for y in range(3)]
test_acc = [[0 for x in range(16)] for y in range(3)]


def calc_acc(data):
    acc = data.apply(lambda row: row[label] == row['pred'], axis=1)
    acc = acc.sum() / len(data)
    return round(acc, 3)


def run():
    for select_method in range(3):
        print(["Entropy", "Majority Error", "Gini Index"][select_method])
        for max_depth in tqdm(range(16)):
            tree = dt.DecisionTree(label, select_method, max_depth + 1)
            tree.generate_tree(train_data, attrs)

            train_data['pred'] = tree.classify(train_data)
            train_acc[select_method][max_depth] = calc_acc(train_data)

            test_data['pred'] = tree.classify(test_data)
            test_acc[select_method][max_depth] = calc_acc(test_data)


def print_res():
    print('Training accuracy:')
    print(pd.DataFrame(train_acc, columns=range(1, 17), index=methods))
    print()
    print('Test accuracy:')
    print(pd.DataFrame(test_acc, columns=range(1, 17), index=methods))


methods = ['Entropy', 'Max Error', 'Gini Index']

print('With UNKNOWN')
run()
print_res()

dt.DecisionTree.fill_missing(train_data, train_data)
dt.DecisionTree.fill_missing(test_data, train_data)

print('\nReplaced UNKNOWN')
run()
print_res()
