import os
import math
import random
import logging
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
    dataset = []
    labels = set()
    with open(file_path) as f:
        for line in f:
            row = [float(_) for _ in line.split(',')]
            dataset.append((row[:-1], row[-1:]))
            labels.add(int(row[-1]))

    def __init__(self, label, size=10):
        """

        :param label: the label must in the pairs.
        :param size: size of the dataloader.
        """
        self.data = []

        if label not in DDMLDataset.labels:
            raise ValueError("Label not in the dataset.")

        self.label = label

        # if cuda.is_available():
        #     tensor = cuda.FloatTensor
        # else:
        #     tensor = FloatTensor

        tensor = FloatTensor

        while len(self.data) < size:
            while True:
                s1 = random.choice(DDMLDataset.dataset)
                s2 = random.choice(DDMLDataset.dataset)
                if int(s1[1][0]) == self.label:
                    break

            self.data.append(((tensor(s1[0]) / 255, tensor(s1[1])), (tensor(s2[0]) / 255, tensor(s2[1]))))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DDMLNet(nn.Module):

    def __init__(self, layer_shape, beta=0.5, tao=5.0, learning_rate=0.01):
        """

        :param layer_shape:
        :param beta:
        :param tao:
        :param learning_rate:
        """
        super(DDMLNet, self).__init__()

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
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)

    def g(self, z):
        """
        Generalized logistic loss function.
        -----------------------------------
        :param z:
        """
        return torch.log(1 + torch.exp(self.beta * z)) / self.beta

    def _g_derivative(self, z):
        """
        The derivative of g(z).
        -----------------------
        :param z:
        """
        return float(1 / (torch.exp(-1 * self.beta * z) + 1))

    def _s_derivative(self, z):
        """
        The derivative of tanh(z).
        --------------------------
        :param z:
        """
        return 1 - self._s(z) ** 2

    def forward(self, feature1, feature2):
        """
        Do forward.
        -----------
        :param feature1: Variable
        :param feature2: Variable
        :return:
        """

        # project features through net
        x1 = feature1
        x2 = feature2
        for layer in self.layers:
            x1 = layer(x1)
            x1 = self._s(x1)
            x2 = layer(x2)
            x2 = self._s(x2)

        return (x2 - x1).norm() ** 2

    def backward(self, dataloader):
        """

        :param dataloader:
        :return:
        """

        gradient = self._compute_gradient(dataloader)

        # update parameters
        for i, param in enumerate(self.parameters()):
            param.data.sub_(self.learning_rate * gradient[i].data)

    def _compute_gradient(self, dataloader):
        """
        Compute gradient.
        -----------------
        :param dataloader: DataLoader of train data pairs batch.
        :return:
        """

        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())
        params_W = params[::2]
        params_b = params[1::2]

        # calculate z(m) and h(m)
        # z(m) is the output of m-th layer without function tanh(x)
        z_i_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(dataloader))]
        h_i_m = [[0 for m in range(self.layer_count)] for index in range(len(dataloader))]
        z_j_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(dataloader))]
        h_j_m = [[0 for m in range(self.layer_count)] for index in range(len(dataloader))]

        for index, (si, sj) in enumerate(dataloader):
            xi = Variable(si[0])
            xj = Variable(sj[0])
            h_i_m[index][0] = xi
            h_j_m[index][0] = xj
            for m in range(self.layer_count - 1):
                layer = self.layers[m]
                xi = layer(xi)
                xj = layer(xj)
                z_i_m[index][m] = xi
                z_j_m[index][m] = xj
                xi = self._s(xi)
                xj = self._s(xj)
                h_i_m[index][m + 1] = xi
                h_j_m[index][m + 1] = xj

        #
        # calculate delta_ij(m)
        # calculate delta_ji(m)
        #

        delta_ij_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(dataloader))]
        delta_ji_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(dataloader))]

        # M = layer_count - 1, then we also need to project 1,2,3 to 0,1,2
        M = self.layer_count - 1 - 1

        # calculate delta(M)
        for index, (si, sj) in enumerate(dataloader):
            xi = Variable(si[0])
            xj = Variable(sj[0])
            yi = si[1]
            yj = sj[1]

            # calculate c
            if int(yi) == int(yj):
                l = 1
            else:
                l = -1

            c = 1 - l * (self.tao - self.compute_distance(xi, xj))

            # h(m) have M + 1 values and m start from 0, in fact, delta_ij_m have M value and m start from 1
            delta_ij_m[index][M] = (self._g_derivative(c) * l * (h_i_m[index][M + 1] - h_j_m[index][M + 1])) * self._s_derivative(z_i_m[index][M])
            delta_ji_m[index][M] = (self._g_derivative(c) * l * (h_j_m[index][M + 1] - h_i_m[index][M + 1])) * self._s_derivative(z_j_m[index][M])

        # calculate delta(m)

        for index in range(len(dataloader)):
            for m in reversed(range(M)):
                delta_ij_m[index][m] = torch.mm(delta_ij_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_i_m[index][m])
                delta_ji_m[index][m] = torch.mm(delta_ji_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_j_m[index][m])

        # calculate partial derivative of W
        partial_derivative_W_m = [self.lambda_ * params_W[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(dataloader)):
                partial_derivative_W_m[m] += (delta_ij_m[index][m] * h_i_m[index][m].t()).t() + (delta_ji_m[index][m] * h_i_m[index][m].t()).t()

        # calculate partial derivative of b
        partial_derivative_b_m = [self.lambda_ * params_b[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(dataloader)):
                partial_derivative_b_m[m] += (delta_ij_m[index][m] + delta_ji_m[index][m]).squeeze()

        # combine two partial derivative vectors
        gradient = []

        for m in range(self.layer_count - 1):
            gradient.append(partial_derivative_W_m[m])
            gradient.append(partial_derivative_b_m[m])

        return gradient


class DDMLLoss(nn.Module):

    @staticmethod
    def forward(distance, l, net):
        """
        Compute loss.
        -------------
        :param distance: Variable
        :param l:
        :param net: ddml network
        :return:
        """
        c = 1 - l * (net.tao - distance)
        loss1 = net.g(c) / 2

        return loss1


def main():
    test_label = 0

    train_epoch_number = 10000
    train_batch_size = 1
    test_data_size = 10000

    layer_shape = (784, 1568, 392)

    logger = setup_logger(level=logging.INFO)

    test_data = DDMLDataset(label=test_label, size=test_data_size)
    test_data_loader = DataLoader(dataset=test_data)

    net = DDMLNet(layer_shape, beta=0.5, tao=5.0, learning_rate=0.01)

    # if cuda.is_available():
    #     net.cuda()
    #     logger.info("Using cuda!")

    pkl = "pkl/ddml({}: {}-{}-{}).pkl".format(test_label, net.beta, net.tao, net.lambda_)

    if os.path.exists(pkl):
        state_dict = torch.load(pkl)
        net.load_state_dict(state_dict)
        logger.info("Load state from file.")

    for epoch in range(train_epoch_number):
        train_data = DDMLDataset(label=test_label, size=train_batch_size)
        train_data_loader = DataLoader(dataset=train_data)
        net.backward(train_data_loader)

        # loss1, loss2 = net.compute_loss(train_data_loader)
        # logger.info("Iteration: %6d, Loss1: %6.3f, Loss2: %6.3f", epoch + 1, loss1, loss2)
        for i, (xi, xj) in enumerate(train_data_loader):
            feature1 = Variable(xi[0], requires_grad=True)
            label1 = xi[1]
            feature2 = Variable(xj[0], requires_grad=True)
            label2 = xj[1]

            if int(label1) == int(label2):
                l = 1
            else:
                l = -1