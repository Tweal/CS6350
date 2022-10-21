from adaboost import ADABoost
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import sys
sys.path.append('..')
import DecisionTree.decision_tree as dt

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
           'housing', 'loan', 'contact', 'day', 'month', 'duration',
           'campaign', 'pdays', 'previous', 'poutcome', 'y']


train_data = pd.read_csv('../data/bank/train.csv', names=columns)

test_data = pd.read_csv('../data/bank/test.csv', names=columns)

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
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
              'oct', 'nov', 'dec'],
    'duration': [0, 1],
    'campaign': [0, 1],
    'pdays': [0, 1],
    'previous': [0, 1],
    'poutcome': ['unknown', 'other', 'failure', 'success']
}
label = 'y'


print('bank.py')


def ADA(print_results=False):
    T = 500
    train_err = []
    test_err = []

    ada = ADABoost(train_data, attrs, label)
    for t in range(1, T) if print_results else tqdm(range(1, T)):
        if print_results:
            print(f'Iteration {t}')

        # Run single iteration
        ada.run_single()

        # Save training results
        train_data['pred'] = ada.classify(train_data)
        train_err.append(ada.calc_err(train_data))
        if print_results:
            print(f'  Train Error : {train_err[-1]}')

        # Save test results
        test_data['pred'] = ada.classify(test_data)
        test_err.append(ada.calc_err(test_data))
        if print_results:
            print(f'  Test Error : {test_err[-1]}')

        breakpoint

    print(f'  Final Train Error : {train_err[-1]}')
    print(f'  Final Test Error : {test_err[-1]}')

    _, (ax1, ax2) = plt.subplots(2, 1)
    x = range(1, T)

    ax1.plot(x, train_err)
    ax1.plot(x, test_err)
    ax1.legend(['Train', 'Test'])
    ax1.set_title('ADABoost Error')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error Rate')

    # Instructions unclear so I'm plotting the tree generated at that iteration
    train_stump_errs = ada.get_stump_errs(train_data)
    test_stump_errs = ada.get_stump_errs(test_data)
    ax2.plot(x, train_stump_errs)
    ax2.plot(test_stump_errs)
    ax2.legend(['Train', 'Test'])
    ax2.set_title('Each Stump Error')
    ax2.set_ylabel('Error Rate')
    ax2.set_xlabel('Iteration')

    plt.show()

    breakpoint


ADA()
print()
