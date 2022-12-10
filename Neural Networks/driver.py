from neural_network import NeuralNetwork
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
import sys


train_data = pd.read_csv('../data/bank-note/train.csv', header=None)
num_cols = train_data.shape[1]
train_x = np.copy(train_data.values)
# Set the last column to 1
train_x[:, num_cols - 1] = 1
# Grab all of the rows and just the last column
train_y = train_data.values[:, num_cols - 1]
# Map from 0, 1 to -1, 1
train_y = 2 * train_y - 1

test_data = pd.read_csv('../data/bank-note/test.csv', header=None)
num_cols = train_data.shape[1]
test_x = np.copy(test_data.values)
test_x[:, num_cols - 1] = 1
test_y = test_data.values[:, num_cols - 1]
test_y = 2 * test_y - 1


def run(random_weights):
    widths = [5, 10, 25, 50, 100]
    for width in widths:
        nn = NeuralNetwork([train_x.shape[1], width, width, 1],
                           random_weights=random_weights)
        nn.train(train_x, train_y)

        # Train err
        preds = nn.predict(train_x)
        preds[preds >= 0] = 1
        preds[preds < 0] = -1

        train_err = 1 - np.mean(preds == train_y)

        # Test err
        preds = nn.predict(test_x)
        preds[preds >= 0] = 1
        preds[preds < 0] = -1

        test_err = 1 - np.mean(preds == test_y)

        print(f'  width: {width}, train error: {train_err.round(3)}, '
              f'test error: {test_err.round(3)}')


which = sys.argv[1] if len(sys.argv) != 1 else False
if(not which or which.lower() == 'random'):
    print('Neural Network')
    print(' With Randomized Initial Weights')
    run(True)
if(not which or which.lower() == 'zero'):
    print('Nerual Network')
    print(' With Zero Initial Weights')
    run(False)
