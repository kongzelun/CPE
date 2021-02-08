import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# dataset path
DATA_PATH = "fashion-mnist_test.csv"
# The size of input data took for one iteration
BATCH_SIZE = 100
# The speed of convergence
LEARNING_RATE = 0.001
# Convergence error
CONVERGENCE_ERROR = 1e-2

LAYER_SHAPE = (784, 392, 28, 10)


def setup_logger():
    """
    setup logger.

    `return`: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger


def data_init(sample_count):
    """
    Initialize and sample dataset.
    ------------------------------

    `sample_count`: sample count.

    `returns`: features and labels of the dataset.
    """
    features = []
    labels = []

    with open(DATA_PATH) as dataset:
        from itertools import islice
        for line in islice(dataset, sample_count):
            row_data = [float(i) for i in line.split(',')]
            features.append(row_data[:-1])
            labels.append(row_data[-1])

    return Variable(torch.FloatTensor(features)), Variable(torch.FloatTensor(labels))


class Net(nn.Module):
    """
    """

    def __init__(self, layer_shape, beta=1, tao=2, lambda_=1, learning_rate=0.001):
        """
        """
        super(Net, self).__init__()

        self.layer_count = len(layer_shape)
        self.layers = []

        for _ in range(self.layer_count - 1):
            layer = nn.Linear(layer_shape[_], layer_shape[_ + 1])
            self.layers.append(layer)
            self.add_module("layer{}".format(_), layer)

        self.beta = beta
        self.tao = tao
        self.lambda_ = lambda_
        self.loss = 0.0
        self.learning_rate = learning_rate
        self.mahalanobis_matrix = None
        self.logger = setup_logger()

    def forward(self, x):
        """
        `x`: Variable, features
        `y`: Variable, labels
        """
        # self.logger.debug("Forward(). Input data shape: %s.", x.size())

        # project features through net
        for layer in self.layers:
            x = layer(x)
            x = F.tanh(x)

        return x

    def backward(self, output, x, y):
        """
        `x`: Variable, features
        `y`: Variable, labels
        """

        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())
        params = list(zip(params[::2], params[1::2]))

        # calculate zi(m) and h(m)
        # zi(m) is the output of m-th layer without function tanh(x)
        z_m_i = []
        h_m_i = []

        for xi in x:
            z_i = []
            h_i = [xi]
            for m in range(self.layer_count - 1):
                layer = self.layers[m]
                xi = layer(xi)
                z_i.append(xi)
                xi = F.tanh(xi)
                h_i.append(xi)

            z_m_i.append(z_i)
            h_m_i.append(h_i)

        self.logger.debug("z(m) and h(m) complete.")

        # calculate delta_ij_M
        delta_ij_M = []

        for i, xi in enumerate(x):
            delta_i_M = []
            for j, xj in enumerate(x):
                # calculate c
                l = 1
                if int(y[i].data) != int(y[j].data):
                    l = -1

                x_i = np.mat(self.forward(xi).data.numpy()).T
                x_j = np.mat(self.forward(xj).data.numpy()).T
                c = 1 - l * (self.tao - float((x_i - x_j).T * self.mahalanobis_matrix * (x_i - x_j)))

                # M = layer_count - 1, then we also need to project 1,2,3 to 0,1,2
                M = self.layer_count - 1 - 1
                # h(m) have M + 1 values
                temp = self._g_derivative(c) * l * (h_m_i[i][M + 1] - h_m_i[j][M + 1]) * self._s_derivative(z_m_i[i][M])

                delta_i_M.append(temp)

            delta_ij_M.append(delta_i_M)

        self.logger.debug("delta_ij(M) complete.")

        # calculate delta_ij_m
        delta_ij_m = []

        # combine delta_ij_M with delta_ij_m
        for m in range(self.layer_count - 1 - 1):
            delta_ij_m.append([])
        delta_ij_m.append(delta_ij_M)

        # M = layer_count - 1, then we also need to project 1,2,3 to 0,1,2
        # here m is in [0,M-1]
        for m in reversed(range(self.layer_count - 1 - 1)):
            delta_m = delta_ij_m[m]
            for i in range(len(x)):
                delat_i_m = []
                for j in range(len(x)):
                    temp = torch.mv(params[m + 1][0].t(), delta_ij_m[m + 1][i][j]) * self._s_derivative(z_m_i[i][m])
                    delat_i_m.append(temp)
                delta_m.append(delat_i_m)

        self.logger.debug("delta_ij(m) complete.")

        # calculate partial derivative of W
        partial_derivative_W_m = []

        for m in range(self.layer_count - 1):
            # temp = 0
            temp = self.lambda_ * params[m][0]
            for i in range(len(x)):
                for j in range(len(x)):
                    # h(m) have M + 1 values
                    # temp += Variable(delta_ij_m[m][i][j].unsqueeze(1)) * Variable(h_m_i[i][m].unsqueeze(0)) + Variable(delta_ij_m[m][j][i].unsqueeze(1)) * Variable(h_m_i[j][m].unsqueeze(0))
                    # self.logger.debug("%d, %d", i, j)
                    temp += (delta_ij_m[m][i][j].unsqueeze(1)) * (h_m_i[i][m].unsqueeze(0)) + (delta_ij_m[m][j][i].unsqueeze(1)) * (h_m_i[j][m].unsqueeze(0))
            # self.logger.debug("%d / %d", m + 1, self.layer_count - 1)
            partial_derivative_W_m.append(temp)

        self.logger.debug("partial_derivative_W(m) complete.")

        # calculate partial derivative of b
        partial_derivative_b_m = []

        for m in range(self.layer_count - 1):
            temp = self.lambda_ * params[m][1]
            for i in range(len(x)):
                for j in range(len(x)):
                    temp += delta_ij_m[m][i][j] + delta_ij_m[m][j][i]
            partial_derivative_b_m.append(temp)

        self.logger.debug("partial_derivative_b(m) complete.")

        # combine two partial derivatve vectors
        partial_derivative = []

        for m in range(self.layer_count - 1):
            partial_derivative.append(partial_derivative_W_m[m])
            partial_derivative.append(partial_derivative_b_m[m])

        # updata parameters
        for i, param in enumerate(self.parameters()):
            param.data -= self.learning_rate * partial_derivative[i].data

    def calculate_loss(self, x, y):
        """
        `x`:
        `y`:
        """

        loss1 = 0.0
        loss2 = 0.0

        x = x.data.numpy()
        y = y.data.numpy()

        # J1
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                l = 1
                if y[i] != y[j]:
                    l = -1

                x_i = np.mat(xi).T
                x_j = np.mat(xj).T
                c = 1 - l * (self.tao - float((x_i - x_j).T * self.mahalanobis_matrix * (x_i - x_j)))
                loss1 += self._g(c)

        loss1 = loss1 / 2

        self.logger.debug("J1 = %f", loss1)

        # J2
        for _ in list(self.parameters()):
            loss2 += float(_.norm(2))

        loss2 = self.lambda_ * loss2 / 2

        self.logger.debug("J2 = %f", loss2)

        self.loss = loss1 + loss2

        return self.loss

    def learn_mahalanobis_matrix(self, features, labels):
        """
        Calculate Mahalanobis Matrix.

        `features`: dataset features.
        `lables`: dataset labels.

        `return`: the projetcion matrix(W) and the Mahalanobis distance matrix(A)
        """

        features = features.data.numpy()
        labels = labels.data.numpy()

        self.logger.debug("Learn mahalanobis matrix start!")

        # random sampling
        data_count = len(features)
        feature_count = features.shape[1]
        sample_count = int(data_count * 0.35)
        sample_index = np.array(random.sample(range(data_count), sample_count))

        features = np.array(features)
        labels = np.array(labels)

        must_link_matrix = np.zeros(shape=(sample_count, sample_count))
        cannot_link_matrix = np.zeros(shape=(sample_count, sample_count))
        sample_labels = labels[sample_index]
        sample_features = features[sample_index]

        for i in range(0, sample_count):
            for j in range(0, sample_count):
                if sample_labels[i] == sample_labels[j]:
                    must_link_matrix[i][j] = 1
                    cannot_link_matrix[i][j] = 0
                else:
                    must_link_matrix[i][j] = 0
                    cannot_link_matrix[i][j] = 1

        A, W = learn_mahalanobis_metric(sample_features, must_link_matrix, cannot_link_matrix, feature_count)

        self.mahalanobis_matrix = A.real

        self.logger.debug("Learn mahalanobis matrix complete!")

        return self.mahalanobis_matrix

    def _g(self, z):
        """
        Generalized logistic loss function.
        -----------------------------------
        """
        return float((np.log(1 + np.exp(self.beta * z))) / self.beta)

    def _g_derivative(self, z):
        """
        The derivative of g(z).
        """
        return float(1 / (np.exp(-1 * self.beta * z) + 1))

    def _s_derivative(self, z):
        """
        The derivative of tanh(z).
        """
        return 1 - F.tanh(z) ** 2


if __name__ == "__main__":
    features, labels = data_init(10)

    net = Net(LAYER_SHAPE)
    output = net(features)
    net.learn_mahalanobis_matrix(output, labels)
    net.calculate_loss(output, labels)
    net.backward(output, features, labels)
