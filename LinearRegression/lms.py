import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LMS:

    def __init__(self, method=0, r=0.01, threshold=10e-6):
        self.set_method(method)
        self.r = r
        self.threshold = threshold
        self.max_iter = 10000

    '''Selection method is expected to be an integer corresponding as follows:
        0: Bath Gradient Descent
        1: Stochastic Gradient Descent
        2: Analytical
    '''
    def set_method(self, method):
        self.method = [self._opt_GD, self._opt_SGD, self._analytical][method]

    def optimize(self, x, y):
        return self.method(x, y)

    def _opt_GD(self, x, y):
        diff = 1
        weights = np.zeros([x.shape[1], 1])
        f_val = []
        iter = 0
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
            iter += 1
            if (iter > self.max_iter):
                raise Exception(f'Failed to Converge within {iter} iterations')

        plt.title('Gradient Descent')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function Value')
        plt.plot(f_val, 'b')
        plt.show()

        return weights

    def _opt_SGD(self, x, y):
        diff = 1
        weights = np.zeros([x.shape[1], 1])
        f_val = []
        iter = 0
        while (diff > self.threshold):
            ind = np.random.randint(x.shape[0])
            # sample x and y
            sx = x[ind]
            sy = y[ind]

            # Note to self, its - y because of the negative outside the sum
            err = np.reshape(np.squeeze(np.matmul(sx, weights)) - sy, (-1, 1))
            # gradient = -sum(y - wx)x
            g = np.reshape(np.sum(np.transpose(err * sx), axis=1), (-1, 1))

            delta = self.r * g
            weights -= delta
            diff = np.linalg.norm(delta)

            err = np.reshape(np.squeeze(np.matmul(x, weights)) - y, (-1, 1))
            cost = 0.5 * np.sum(np.square(err))
            f_val.append(cost)
            iter += 1
            if (iter > self.max_iter):
                raise Exception(f'Failed to Converge within {iter} iterations')

        plt.title('Stochastic Gradient Descent')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function Value')
        plt.plot(f_val, 'b')
        plt.show()

        return weights

    def _analytical(self, x, y):
        xt = np.transpose(x)
        left = np.linalg.inv(np.matmul(xt, x))
        right = np.matmul(xt, y)
        return np.matmul(left, right)
