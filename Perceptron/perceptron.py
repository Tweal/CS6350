import pandas as pd
import numpy as np


def standard(x, y, lr=0.1, T=10):
    rows, cols = x.shape
    w = np.zeros(cols)
    idx = np.arange(rows)
    for _ in range(T):
        np.random.shuffle(idx)
        x = x[idx, :]
        y = y[idx]
        for i in range(rows):
            tmp = np.sum(x[i] * w)
            if tmp * y[i] <= 0:
                w = w + lr * y[i] * x[i]
    # transpose w since np.transpose is stupid
    w = np.reshape(w, (-1, 1))
    return w


def voted(x, y, lr=0.1, T=10):
    rows, cols = x.shape
    idx = np.arange(rows)
    counts = np.array([])
    c = 0
    weights = np.array([])
    w = np.zeros(cols)
    for _ in range(T):
        np.random.shuffle(idx)
        x = x[idx, :]
        y = y[idx]
        for i in range(rows):
            tmp = np.sum(x[i] * w)
            if tmp * y[i] <= 0:
                weights = np.append(weights, w)
                counts = np.append(counts, c)
                w = w + lr * y[i] * x[i]
                c = 1
            else:
                c = c + 1
    # Reshape weights from 1d to 2d
    weights = np.reshape(weights, (counts.shape[0], -1))
    # Reshape counts to column vector
    counts = np.reshape(counts, (-1, 1))
    return counts, weights


def average(x, y, lr=0.1, T=10):
    rows, cols = x.shape
    w = np.zeros(cols)
    a = np.zeros(cols)
    idx = np.arange(rows)
    for _ in range(T):
        np.random.shuffle(idx)
        x = x[idx, :]
        y = y[idx]
        for i in range(rows):
            tmp = np.sum(x[i] * w)
            if tmp * y[i] <= 0:
                w = w + lr * y[i] * x[i]
            a = a + w
    a = np.reshape(a, (-1, 1))
    return a
