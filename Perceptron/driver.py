import perceptron
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

# Standard Perceptron
print('Standard Perceptron')
weights = perceptron.standard(train_x, train_y)
preds = np.sign(np.matmul(test_x, weights))
# Divide by 2 to offset the conversion to -1, 1. That makes the diffs 2 instead
# of 1. Without it our err ends up double what it should be.
err = np.sum(np.abs(preds - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
print(f'  Error: {err}')
print(f'  Weights: {np.reshape(weights, (1, -1))}')

# Voted Perceptron
print('\nVoted Perceptron')
counts, weights = perceptron.voted(train_x, train_y)
if len(sys.argv) > 1 and sys.argv[1] == 'output':
    for c, w in zip(counts, weights):
        print(f'  Count: {c[0]}, Weights: {w}')
    print()
# Weights is currently c x 5, need it to be 5 x c
weights = np.transpose(weights)
# This should give a 500 x c matrix
preds = np.sign(np.matmul(test_x, weights))
# Finally this gives a 500 x 1 matrix of predictions
voted_preds = np.sign(np.matmul(preds, counts))
pred_sum = np.sum(np.abs(voted_preds - np.reshape(test_y, (-1, 1))))
err = pred_sum / 2 / test_y.shape[0]
print(f'  Error: {err}')

# Average Perceptron, which should really not be called average.
print('\nAverage Perceptron')
weights = perceptron.average(train_x, train_y)
preds = np.sign(np.matmul(test_x, weights))
err = np.sum(np.abs(preds - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
print(f'  Error: {err}')
print(f'  Weights: {np.reshape(weights, (1, -1))}')
