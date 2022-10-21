from lms import LMS
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

train_data = pd.read_csv('../data/concrete/train.csv', header=None)
num_rows = train_data.shape[0]
num_cols = train_data.shape[1]
train_x = np.copy(train_data.values)
train_x[:, num_cols - 1] = 1
train_y = train_data.values[:, num_cols - 1]

test_data = pd.read_csv('../data/concrete/test.csv', header=None)
num_rows = train_data.shape[0]
num_cols = train_data.shape[1]
test_x = test_data.values[:, :num_cols - 1]
test_y = test_data.values[:, num_cols - 1]

lms = LMS()
# BGD
print('Batch Gradient Descent')
weights = lms.optimize(train_x, train_y)
print(f'  Weights: {weights}')

breakpoint
