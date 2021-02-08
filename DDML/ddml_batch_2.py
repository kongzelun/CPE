import os
import random
import logging
import math
import numpy as np
import torch
import torch.cuda as cuda
from torch import FloatTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


def setup_logger(level=logging.DEBUG):
    """
    Setup logger.

    `return`: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class DDMLDataset(Dataset):
    """
    Implement a Dataset.
    """

    file_path = 'fashion-mnist_test.csv'

    def __init__(self, size=10):
        self._size = size
        self.data = []

        with open(self.file_path) as f:
            for line in random.sample(list(f), self._size):
                row = [float(_) for _ in line.split(',')]
                self.data.append((row[:-1], row[-1:]))

    def __getitem__(self, index):
        s = self.data[index]

        if False:  # cuda.is_available():
            return cuda.FloatTensor(s[0]), cuda.FloatTensor(s[1])
        else:
            return FloatTensor(s[0]) / 255, FloatTensor(s[1])

    def __len__(self):
        # return len(self.features)
        return self._size


class Net(nn.Module):

    def __init__(self, layer_shape, beta=0.5, tao=5, lambda_=0.01, learning_rate=0.001):
        """

        :param layer_shape:
        :param beta:
        :param tao:
        :param lambda_:
        :param learning_rate:
        """
        super(Net, self).__init__()

        self.layer_count = len(layer_shape)
        self.layers = []

        for _ in range(self.layer_count - 1):
            stdv = math.sqrt(6.0 / (layer_shape[_] + layer_shape[_ + 1]))
            layer = nn.Linear(layer_shape[_], layer_shape[_ + 1])
            layer.weight.data.uniform_(-stdv, stdv)
            layer.bias.data.uniform_(0, 0)
            self.layers.append(layer)
            self.add_module("layer{}".format(_), layer)

        self._s = F.tanh

        self.beta = beta
        self.tao = tao
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.gradient = []
        self.logger = logging.getLogger(__name__)

    def forward(self, feature):
        """
        Do forward.
        -----------
        :param features: Variable, feature
        :return:
        """

        # project features through net
        x = feature
        for layer in self.layers:
            x = layer(x)
            x = self._s(x)

        return x

    def compute_gradient(self, dataloader):
        """
        Compute gradient.
        -----------------
        :param dataloader: DataLoader of train data batch.
        :return:
        """

        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())
        params_M = params[::2]
        params_b = params[1::2]

        # calculate zi(m) and hi(m)
        # zi(m) is the output of m-th layer without function tanh(x)
        z_i_m = [[0 for m in range(self.layer_count - 1)] for i in range(len(dataloader))]
        h_i_m = [[0 for m in range(self.layer_count)] for i in range(len(dataloader))]

        for i, (xi, yi) in enumerate(dataloader):
            xi = Variable(xi)
            h_i_m[i][0] = xi
            for m in range(self.layer_count - 1):
                layer = self.layers[m]
                xi = layer(xi)
                z_i_m[i][m] = xi
                xi = self._s(xi)
                h_i_m[i][m + 1] = xi

        # z_i_m = []
        # h_i_m = []
        #
        # for xi, yi in dataloader:
        #     xi = Variable(xi)
        #     z_i = []
        #     h_i = [xi]
        #     for m in range(self.layer_count - 1):
        #         layer = self.layers[m]
        #         xi = layer(xi)
        #         z_i.append(xi)
        #         xi = F.tanh(xi)
        #         h_i.append(xi)
        #
        #     z_i_m.append(z_i)
        #     h_i_m.append(h_i)

        #
        # calculate delta_ij_m
        #

        delta_ij_m = [[[0 for m in range(self.layer_count - 1)] for j in range(len(dataloader))] for i in range(len(dataloader))]
        # M = layer_count - 1, then we also need to project 1,2,3 to 0,1,2
        M = self.layer_count - 1 - 1

        for i, (xi, yi) in enumerate(dataloader):
            xi = Variable(xi)

            for j, (xj, yj) in enumerate(dataloader):
                xj = Variable(xj)

                # calculate c
                if int(yi) == int(yj):
                    l = 1
                else:
                    l = -1

                c = 1 - l * (self.tao - self._compute_distance(xi, xj))

                # h(m) have M + 1 values and m start from 0, in fact, delta_ij_m have M value and m start from 1
                delta_ij_m[i][j][M] = (self._g_derivative(c) * l * (h_i_m[i][M + 1] - h_i_m[j][M + 1])) * self._s_derivative(z_i_m[i][M])

        for i, (xi, yi) in enumerate(dataloader):
            for j, (xj, yj) in enumerate(dataloader):
                for m in reversed(range(M)):
                    delta_ij_m[i][j][m] = torch.mm(delta_ij_m[i][j][m + 1], params_M[m + 1]) * self._s_derivative(z_i_m[i][m])

        # calculate partial derivative of W
        partial_derivative_W_m = [self.lambda_ * params_M[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for i in range(len(dataloader)):
                for j in range(len(dataloader)):
                    partial_derivative_W_m[m] += (delta_ij_m[i][j][m] * h_i_m[i][m].t()).t() + (delta_ij_m[j][i][m] * h_i_m[j][m].t()).t()

        # calculate partial derivative of b
        partial_derivative_b_m = [self.lambda_ * params_b[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for i in range(len(dataloader)):
                for j in range(len(dataloader)):
                    partial_derivative_b_m[m] += (delta_ij_m[i][j][m] + delta_ij_m[j][i][m]).squeeze()

        # combine two partial derivative vectors
        gradient = []

        for m in range(self.layer_count - 1):
            gradient.append(partial_derivative_W_m[m])
            gradient.append(partial_derivative_b_m[m])

        self.gradient = gradient

    def backward(self):
        """
        Doing backward propagation.
        ---------------------------
        """

        if self.gradient:
            # update parameters
            for i, param in enumerate(self.parameters()):
                param.data = param.data.sub(self.learning_rate * self.gradient[i].data)

            # clear gradient
            del self.gradient[:]
        else:
            self.logger.warning("Gradient is not computed.")

    def compute_loss(self, dataloader):
        """
        Compute loss.
        -------------
        :param dataloader: DataLoader of train data batch.
        :return:
        """

        loss1 = 0.0
        loss2 = 0.0

        # J1
        for (xi, yi) in dataloader:
            xi = Variable(xi)
            for (xj, yj) in dataloader:
                xj = Variable(xj)

                if int(yi) == int(yj):
                    l = 1
                else:
                    l = -1

                dist = self._compute_distance(xi, xj)
                c = 1 - l * (self.tao - dist)
                loss1 += self._g(c)

        loss1 = loss1 / 2

        self.logger.debug("J1 = %f", loss1)

        # J2
        for p in list(self.parameters()):
            loss2 += p.data.norm()

        loss2 = self.lambda_ * loss2 / 2

        self.logger.debug("J2 = %f", loss2)

        return loss1, loss2

    def is_similar(self, feature1, feature2):
        distance = self._compute_distance(feature1, feature2)
        result = distance <= self.tao

        return result, distance

    def _compute_distance(self, feature1, feature2):
        """
        Compute the distance of two samples.
        ------------------------------------
        :param feature1: Variable
        :param feature2: Variable
        :return: The distance of the two sample.
        """
        return (self(feature1) - self(feature2)).data.norm() ** 2

    def _g(self, z):
        """
        Generalized logistic loss function.
        -----------------------------------
        :param z:
        """
        return float((np.log(1 + np.exp(self.beta * z))) / self.beta)

    def _g_derivative(self, z):
        """
        The derivative of g(z).
        -----------------------
        :param z:
        """
        return float(1 / (np.exp(-1 * self.beta * z) + 1))

    def _s_derivative(self, z):
        """
        The derivative of tanh(z).
        --------------------------
        :param z:
        """
        return 1 - self._s(z) ** 2


if __name__ == '__main__':

    train_epoch_number = 1000
    train_batch_size = 10
    test_data_size = 100

    layer_shape = (784, 392, 196, 98)

    # logger = setup_logger()
    logger = setup_logger(level=logging.INFO)

    test_data = DDMLDataset(size=test_data_size)
    test_data_loader = DataLoader(dataset=test_data)

    net = Net(layer_shape, beta=0.5, tao=5, lambda_=0.1, learning_rate=0.001)

    pkl = "pkl/ddml({} {} {} {}).pkl".format(net.beta, net.tao, net.lambda_, net.learning_rate)

    if False:  # cuda.is_available():
        net.cuda()
        logger.info("Using cuda!")

    if os.path.exists(pkl):
        state_dict = torch.load(pkl)
        net.load_state_dict(state_dict)
    else:
        for epoch in range(train_epoch_number):
            train_data = DDMLDataset(size=train_batch_size)
            train_data_loader = DataLoader(dataset=train_data)
            net.compute_gradient(train_data_loader)
            net.backward()
            loss1, loss2 = net.compute_loss(train_data_loader)
            logger.info("Iteration: %6d, Loss1: %9.3f, Loss2: %9.3f", epoch + 1, loss1, loss2)

            torch.save(net.state_dict(), "pkl/ddml({} {} {} {}).pkl".format(net.beta, net.tao, net.lambda_, net.learning_rate))

    similar_dist_sum = 0.0
    dissimilar_dist_sum = 0.0
    similar_incorrect = 0
    dissimilar_incorrect = 0
    similar_correct = 0
    dissimilar_correct = 0
    num = 0

    for xi, yi in test_data_loader:
        xi = Variable(xi)
        for xj, yj in test_data_loader:
            xj = Variable(xj)

            actual = (int(yi) == int(yj))
            result, dist = net.is_similar(xi, xj)

            if actual:
                similar_dist_sum += dist
                if result:
                    similar_correct += 1
                else:
                    similar_incorrect += 1
            else:
                dissimilar_dist_sum += dist
                if not result:
                    dissimilar_correct += 1
                else:
                    dissimilar_incorrect += 1

            num += 1

            logger.info("%6d, %5s, %5s, %9.3f", num, actual, result, dist)

    logger.info("Similar Average Distance: %.6f", similar_dist_sum / (similar_correct + similar_incorrect))
    logger.info("Dissimilar: Average Distance: %.6f", dissimilar_dist_sum / (dissimilar_correct + dissimilar_incorrect))
    logger.info("\nConfusion Matrix:\n\t%6d\t%6d\n\t%6d\t%6d", similar_correct, similar_incorrect, dissimilar_incorrect, dissimilar_correct)
