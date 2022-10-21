import pandas as pd
import numpy as np


class LMS:
    '''Selection method is expected to be an integer corresponding as follows:
        0: Bath Gradient Descent
        1: Stochastic Gradient Descent
    '''
    def __init__(self, method=0, r=0.01, threshold=10e-6):
        self.method = [self._opt_GD, self._opt_SGD][method]
        self.r = r
        self.threshold = threshold

    def optimize(self, x, y):
        return self.method(x, y)

    def _opt_GD(self, x, y):
        diff = 1
        weights = np.zeros([x.shape[1], 1])
        f_val = []
        while (diff > self.threshold):
            # Note to self, its - y because of the negative outside the sum
            err = np.reshape(np.squeeze(np.matmul(x, weights)) - y, (-1, 1))
            # gradient = - sum(y - wx)x
            g = np.reshape(np.sum(np.transpose(err * x), axis=1), (-1, 1))

            delta = self.r * g
            weights -= delta
            diff = np.linalg.norm(delta)

            err = np.reshape(np.squeeze(np.matmul(x, weights)) - y, (-1, 1))
            cost = 0.5 * np.sum(np.square(err))
            f_val.append(cost)

        fig = plt.figure()
        fig.suptitle('Gradient Descent')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function Value')
        plt.plot(f_val, 'b')
        plt.legend(['train'])

        breakpoint

    def _opt_SGD(self):
        breakpoint
