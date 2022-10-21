from random_forest import RandomForest, RandomTree
from bagged_trees import BaggedTrees
from adaboost import ADABoost
import pandas as pd
import numpy as np
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
T = 501
t_range = range(1, T, 10)


def ADA(print_results=False):
    print('Running AdaBoost. Expected run time: 5 minutes')
    train_err = []
    test_err = []

    ada = ADABoost(train_data, attrs, label)
    for t in t_range if print_results else tqdm(t_range):
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

    ax1.plot(t_range, train_err)
    ax1.plot(t_range, test_err)
    ax1.legend(['Train', 'Test'])
    ax1.set_title('ADABoost Error')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error Rate')

    # Instructions unclear so I'm plotting the tree generated at that iteration
    train_stump_errs = ada.get_stump_errs(train_data)
    test_stump_errs = ada.get_stump_errs(test_data)
    ax2.plot(t_range, train_stump_errs)
    ax2.plot(t_range, test_stump_errs)
    ax2.legend(['Train', 'Test'])
    ax2.set_title('Each Stump Error')
    ax2.set_ylabel('Error Rate')
    ax2.set_xlabel('Iteration')

    plt.show()

    breakpoint


def bagged_trees(print_results=False):
    print('Running Bagged Trees. Expected Run Time: 15 minutes')
    train_err = []
    test_err = []

    bt = BaggedTrees(train_data, attrs, label)
    for t in t_range if print_results else tqdm(t_range):
        if print_results:
            print(f'Iteration {t}')

        # Run single iteration
        bt.run_single()

        # Save training results
        train_data['pred'] = bt.classify(train_data)
        train_err.append(bt.calc_err(train_data))
        if print_results:
            print(f'  Train Error : {train_err[-1]}')

        # Save test results
        test_data['pred'] = bt.classify(test_data)
        test_err.append(bt.calc_err(test_data))
        if print_results:
            print(f'  Test Error : {test_err[-1]}')

        breakpoint

    print(f'  Final Train Error : {train_err[-1]}')
    print(f'  Final Test Error : {test_err[-1]}')

    plt.plot(t_range, train_err)
    plt.plot(t_range, test_err)
    plt.legend(['Train', 'Test'])
    plt.title('Bagged Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error Rate')

    plt.show()

    breakpoint


def bvd_bagged():
    print('Running Bias Variance Decomposition for Bagged.')
    print('Expected runtime: Goodluck...')

    bags = []
    for _ in tqdm(range(100)):
        sample = train_data.sample(n=1000).reset_index()
        bags.append(BaggedTrees(sample, attrs, label))
        bags[-1].run(T=500, keep_output=False)
        breakpoint

    '''First tree only'''
    trees = [bag.trees[0] for bag in bags]
    test_temp = test_data.copy()
    # need something to map label to 0 / 1 later
    val = test_temp[label].unique()[0]

    # Creates a series (ie column) of lists for tree votes
    temp = test_temp.apply(lambda row: [tree.classify(row)
                                        for tree in trees], axis=1)

    # Collapse all trees into the average of their labels
    test_temp['h'] = temp.apply(lambda row: np.mean([int(x == val) for
                                                     x in row]))

    # The lambda is calculating the bias for each sample before taking the mean
    bias = np.mean(test_temp.apply(lambda row: (int(row[label] == val) - 
                                                row['h']) ** 2, axis=1))
    var = np.var(test_temp["h"])
    se = bias + var
    print(f'  Single Tree:')
    print(f'\tBias: {bias}')
    print(f'\tVariance: {var}')
    print(f'\tSquared Error: {se}')

    '''All Trees'''
    # Creates a series (ie column) of lists for tree votes
    temp = test_temp.apply(lambda row: [bag.classify(row)
                                        for bag in bags], axis=1)

    # Collapse all trees into the average of their labels
    test_temp['h'] = temp.apply(lambda row: np.mean([int(x == val) for
                                                     x in row]))

    # The lambda is calculating the bias for each sample before taking the mean
    bias = np.mean(test_temp.apply(lambda row: (int(row[label] == val) - 
                                                row['h']) ** 2, axis=1))
    var = np.var(test_temp["h"])
    se = bias + var
    print(f'  Bagged Trees:')
    print(f'\tBias: {bias}')
    print(f'\tVariance: {var}')
    print(f'\tSquared Error: {se}')

    breakpoint


def random_forest(print_results=False):
    print('Running random forest: Expected run time: 36 minutes')

    for size in [2, 4, 6]:
        rf = RandomForest(train_data, attrs, label, size)

        train_err = []
        test_err = []

        for t in t_range if print_results else tqdm(t_range):
            if print_results:
                print(f'Iteration {t}')

            # Run single iteration
            rf.run_single()

            # Save training results
            train_data['pred'] = rf.classify(train_data)
            train_err.append(rf.calc_err(train_data))
            if print_results:
                print(f'  Train Error : {train_err[-1]}')

            # Save test results
            test_data['pred'] = rf.classify(test_data)
            test_err.append(rf.calc_err(test_data))
            if print_results:
                print(f'  Test Error : {test_err[-1]}')

        print(f'  Final Train Error : {train_err[-1]}')
        print(f'  Final Test Error : {test_err[-1]}')

        plt.plot(t_range, train_err)
        plt.plot(t_range, test_err)

    plt.legend([f'Train (2)', f'Test (2)',
                f'Train (4)', f'Test (4)',
                f'Train (6)', f'Test (6)'])
    plt.title('Random Forest Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error Rate')

    plt.show()
    breakpoint


def bvd_rf():
    print('Running Bias Variance Decomposition for Random Forest.')
    print('Expected runtime: Goodluck...')

    forests = []
    for _ in tqdm(range(100)):
        sample = train_data.sample(n=1000).reset_index()
        forests.append(RandomForest(sample, attrs, label, 6))
        forests[-1].run(T=550, keep_output=False)
        breakpoint

    '''First tree only'''
    trees = [forest.trees[0] for forest in forests]
    test_temp = test_data.copy()
    # need something to map label to 0 / 1 later
    val = test_temp[label].unique()[0]

    # Creates a series (ie column) of lists for tree votes
    temp = test_temp.apply(lambda row: [tree.classify(row)
                                        for tree in trees], axis=1)

    # Collapse all trees into the average of their labels
    test_temp['h'] = temp.apply(lambda row: np.mean([int(x == val) for
                                                     x in row]))

    # The lambda is calculating the bias for each sample before taking the mean
    bias = np.mean(test_temp.apply(lambda row: (int(row[label] == val) - 
                                                row['h']) ** 2, axis=1))
    var = np.var(test_temp["h"])
    se = bias + var
    print(f'  Single Tree:')
    print(f'\tBias: {bias}')
    print(f'\tVariance: {var}')
    print(f'\tSquared Error: {se}')

    '''All Trees'''
    # Creates a series (ie column) of lists for tree votes
    temp = test_temp.apply(lambda row: [forest.classify(row)
                                        for forest in forests], axis=1)

    # Collapse all trees into the average of their labels
    test_temp['h'] = temp.apply(lambda row: np.mean([int(x == val) for
                                                     x in row]))

    # The lambda is calculating the bias for each sample before taking the mean
    bias = np.mean(test_temp.apply(lambda row: (int(row[label] == val) - 
                                                row['h']) ** 2, axis=1))
    var = np.var(test_temp["h"])
    se = bias + var
    print(f'  Random Forest:')
    print(f'\tBias: {bias}')
    print(f'\tVariance: {var}')
    print(f'\tSquared Error: {se}')

    breakpoint


which = sys.argv[1]
if(which.lower() == 'ada'):
    ADA()

if(which.lower() == 'bagged'):
    bagged_trees()

if(which.lower() == 'bvdbagged'):
    bvd_bagged()

if(which.lower() == 'rf'):
    random_forest()

if(which.lower() == 'bvdrf'):
    bvd_rf()

breakpoint
