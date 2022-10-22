from lms import LMS
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

train_data = pd.read_csv('../data/concrete/train.csv', header=None)
num_cols = train_data.shape[1]
train_x = np.copy(train_data.values)
train_x[:, num_cols - 1] = 1
train_y = train_data.values[:, num_cols - 1]

test_data = pd.read_csv('../data/concrete/test.csv', header=None)
num_cols = train_data.shape[1]
test_x = np.copy(test_data.values)
test_x[:, num_cols - 1] = 1
test_y = test_data.values[:, num_cols - 1]

# BGD
lms = LMS()
print('Batch Gradient Descent')
weights = lms.optimize(train_x, train_y)
print(f'  Weights: {weights}')
err = np.reshape(np.squeeze(np.matmul(test_x, weights)) - test_y, (-1, 1))
cost = 0.5 * np.sum(np.square(err))
print(f'  Cost of Test: {cost}')

# SGD
lms.set_method(1)
print('Stochastic Gradient Descent')
weights = lms.optimize(train_x, train_y)
print(f'  Weights: {weights}')
err = np.reshape(np.squeeze(np.matmul(test_x, weights)) - test_y, (-1, 1))
cost = 0.5 * np.sum(np.square(err))
print(f'  Cost of Test: {cost}')

# Analytical
lms.set_method(2)
print('Analytical Method')
weights = lms.optimize(train_x, train_y)
print(f'  Weights: {weights}')
err = np.reshape(np.squeeze(np.matmul(test_x, weights)) - test_y, (-1, 1))
cost = 0.5 * np.sum(np.square(err))
print(f'  Cost of Test: {cost}')

breakpoint
