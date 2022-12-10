import numpy as np
import warnings

warnings.filterwarnings('ignore')


class NeuralNetwork:
    def __init__(self, dim, T=100, lr=0.01, d=.1, gamma=.1,
                 random_weights=True):
        self.input_size = dim[0]
        self.output_size = dim[-1]
        self.dim = dim
        self.layers = len(dim)

        self.lr = lr
        self.gamma = gamma
        self.d = d

        self.lr_sched = lambda t: gamma / (1 + ((gamma / self.d) * t))

        self.T = T

        self.weights = [None]
        self.grads = [None]
        self.output = []

        for i in range(0, self.layers):
            if i > 0:
                input_dim = dim[i] - 1
                output_dim = dim[i - 1]
                if i == self.layers - 1:
                    input_dim += 1

                if random_weights:
                    weight = np.random.normal(size=(input_dim, output_dim))
                else:
                    weight = np.zeros((input_dim, output_dim))

                self.weights.append(weight)
                self.grads.append(np.zeros([input_dim, output_dim]))

            self.output.append(np.ones((dim[i], 1)))

    def train(self, train_x, train_y):
        for epoch in range(self.T):
            idx = np.random.permutation(train_x.shape[0])
            for x, y in zip(train_x[idx], train_y[idx]):
                self._predict_single(x.reshape((self.input_size, 1)))
                self.back_prop(y.reshape((self.output_size, 1)))
                self.update_weights(self.lr_sched(epoch))

    def update_weights(self, lr):
        self.weights = [None] + [weight - (grad * lr) for grad, weight
                                 in zip(self.grads[1:], self.weights[1:])]

    def back_prop(self, y):
        dL_dZ = self.output[-1] - y
        dZ_dW = np.tile(self.output[-2], (1, self.dim[-1])).T

        self.grads[-1] = dL_dZ * dZ_dW
        dZ_dZ = self.weights[-1][:, :-1]

        for i in range(1, self.layers - 1)[::-1]:
            dL_dZ, dZ_dZ = self.layer_backwards(dL_dZ, dZ_dZ, i)

    def layer_backwards(self, dL_dZ, dZ_dZ, layer_index):
        no_bias_dim = self.dim[layer_index] - 1

        layer_input = self.output[layer_index - 1]
        layer_output = self.output[layer_index][:-1]

        dL_dZ = dZ_dZ.T @ dL_dZ
        dZ_dZ = (layer_output * (1 - layer_output) *
                 self.weights[layer_index])[:, :-1]

        self.grads[layer_index] = dL_dZ * layer_output * (1 - layer_output) *\
            np.tile(layer_input, (1, no_bias_dim)).T
        return dL_dZ, dZ_dZ

    def predict(self, X):
        # There is probably a way to get this stupid thing to output the array
        # as the right shape but I couldn't figure it out.
        return np.array([self._predict_single(x.reshape(self.input_size)).T
                         for x in X]).reshape((-1,))

    def _predict_single(self, x):
        self.output[0] = x
        for i in range(1, self.layers):
            layer_output = self.weights[i] @ self.output[i - 1]
            if i < self.layers - 1:
                self.output[i][:-1, :] = sigmoid(layer_output).reshape((-1, 1))
            else:
                self.output[i] = layer_output

        return self.output[-1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
