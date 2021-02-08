import os
import random
import logging
import math
import numpy as np
import torch
# import torch.cuda as cuda
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

    def __init__(self, size=0):
        self.size = size
        self.data = []

        # self.features = []
        # self.labels = []

        # with open(self.file_path) as f:
        #     from itertools import islice
        #     if size > 0:
        #         for line in islice(f, size):
        #             row = [float(_) for _ in line.split(',')]
        #             self.features.append(row[:-1])
        #             self.labels.append([row[-1]])
        #     else:
        #         for line in f:
        #             row = [float(_) for _ in line.split(',')]
        #             self.features.append(row[:-1])
        #             self.labels.append([row[-1]])

        with open(self.file_path) as f:
            for line in f:
                row = [float(_) for _ in line.split(',')]
                self.data.append({'feature': row[:-1], 'label': row[-1:]})

    def __getitem__(self, index):
        # return cuda.FloatTensor(self.features[index]), cuda.FloatTensor(self.labels[index])
        # return (FloatTensor(self.features[index]), FloatTensor(self.labels[index])), (FloatTensor(self.features[index]), FloatTensor(self.labels[index]))

        s1 = random.choice(self.data)
        s2 = random.choice(self.data)

        # return (FloatTensor(s1['feature']) / 255, FloatTensor(s1['label'])), (FloatTensor(s2['feature']) / 255, FloatTensor(s2['label']))
        return (FloatTensor(s1['feature']), FloatTensor(s1['label'])), (FloatTensor(s2['feature']), FloatTensor(s2['label']))

    def __len__(self):
        # return len(self.features)
        return self.size


class Net(nn.Module):

    def __init__(self, layer_shape, beta=0.5, tao=5, lambda_=0.001, learning_rate=0.001):
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

        self.s = F.tanh

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
        # self.logger.debug("Forward(). Input data shape: %s.", x.size())

        # project features through net
        x = feature
        for layer in self.layers:
            x = layer(x)
            x = self.s(x)

        return x

    def compute_gradient(self, sample1, sample2):
        """
        Compute gradient.
        -----------------
        :param sample1: Dict, with 'feature' and 'label' index.
        :param sample2: Dict, with 'feature' and 'label' index.
        :return:
        """

        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())
        params = list(zip(params[::2], params[1::2]))

        feature1 = sample1['feature']
        feature2 = sample2['feature']

        l = 0
        if int(sample1['label'].data) == int(sample2['label'].data):
            l = 1
        else:
            l = -1

        # calculate zi(m) and h(m)
        # zi(m) is the output of m-th layer without function tanh(x)

        z1 = []
        h1 = [feature1]
        z2 = []
        h2 = [feature2]

        x1 = feature1
        x2 = feature2
        for layer in self.layers:
            x1 = layer(x1)
            x2 = layer(x2)
            z1.append(x1)
            z2.append(x2)
            x1 = self.s(x1)
            x2 = self.s(x2)
            h1.append(x1)
            h2.append(x2)

        # for m in range(self.layer_count - 1):
        #     layer = self.layers[m]
        #     x1 = layer(x1)
        #     x2 = layer(x2)
        #     z1.append(x1)
        #     z2.append(x2)
        #     x1 = self.s(x1)
        #     x2 = self.s(x2)
        #     h1.append(x1)
        #     h2.append(x2)

        self.logger.debug("z(m) and h(m) complete.")

        # calculate delta_ij(m)
        delta_12_m = [0 for m in range(self.layer_count - 1)]
        delta_21_m = [0 for m in range(self.layer_count - 1)]

        # calculate delta_ij(M)
        M = self.layer_count - 1 - 1

        # calculate c
        c = 1 - l * (self.tao - self._compute_distance(feature1, feature2))

        delta_12_m[M] = self._g_derivative(c) * l * self._s_derivative(z1[M])
        delta_21_m[M] = self._g_derivative(c) * l * self._s_derivative(z2[M])

        for m in reversed(range(M)):
            delta_12_m[m] = torch.mm(delta_12_m[m + 1], params[m + 1][0]) * self._s_derivative(z1[m])
            delta_21_m[m] = torch.mm(delta_21_m[m + 1], params[m + 1][0]) * self._s_derivative(z2[m])

        self.logger.debug("delta_ij(m) complete.")

        # calculate partial derivative of W
        partial_derivative_W_m = []

        for m in range(self.layer_count - 1):
            temp = (self.lambda_ * params[m][0]) + (delta_12_m[m] * h1[m].t()).t() + (delta_21_m[m] * h2[m].t()).t()
            partial_derivative_W_m.append(temp)

        self.logger.debug("partial_derivative_W(m) complete.")

        # calculate partial derivative of b
        partial_derivative_b_m = []

        for m in range(self.layer_count - 1):
            temp = self.lambda_ * params[m][1] + delta_12_m[m] + delta_21_m[m]
            partial_derivative_b_m.append(temp)

        self.logger.debug("partial_derivative_b(m) complete.")

        # combine two partial derivatve vectors
        gradient = []

        for m in range(self.layer_count - 1):
            gradient.append(partial_derivative_W_m[m])
            gradient.append(partial_derivative_b_m[m])

        self.gradient = gradient

    def backward(self):
        """
        """

        if self.gradient:
            # update parameters
            for i, param in enumerate(self.parameters()):
                param.data = param.data.sub(0.01 * self.learning_rate * self.gradient[i].data)

            # clear gradient
            del self.gradient[:]
        else:
            self.logger.warning("Gradient is not computed.")

    def compute_loss(self, sample1, sample2):
        """

        :param sample1:
        :param sample2:
        :return:
        """

        loss2 = 0.0

        feature1 = sample1['feature']
        feature2 = sample2['feature']

        l = 0
        if int(sample1['label'].data) == int(sample2['label'].data):
            l = 1
        else:
            l = -1

        # J1
        c = 1 - l * (self.tao - self._compute_distance(feature1, feature2))
        loss1 = self._g(c) / 2

        self.logger.debug("J1 = %f", loss1)

        # J2
        for _ in list(self.parameters()):
            loss2 += _.data.norm()

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
        return 1 - self.s(z) ** 2


if __name__ == '__main__':
    pkl = 'ddml.pkl'

    train_sample_size = 1000
    test_sample_size = 100

    layer_shape = (784, 392, 196)

    # logger = setup_logger()
    logger = setup_logger(level=logging.INFO)

    test_data = DDMLDataset(size=test_sample_size)
    test_data_loader = DataLoader(dataset=test_data)

    net = Net(layer_shape, beta=1, tao=10, lambda_=0.01, learning_rate=0.01)
    # net.cuda()

    if False:  # os.path.exists(pkl):
        state_dict = torch.load(pkl)
        net.load_state_dict(state_dict)
    else:
        train_data = DDMLDataset(size=train_sample_size)
        train_data_loader = DataLoader(dataset=train_data)

        for i, (s1, s2) in enumerate(train_data_loader):
            sample1 = {'feature': Variable(s1[0]), 'label': Variable(s1[1])}
            sample2 = {'feature': Variable(s2[0]), 'label': Variable(s2[1])}
            actual = int(sample1['label'].data) == int(sample2['label'].data)
            net.compute_gradient(sample1, sample2)
            net.backward()
            loss1, loss2 = net.compute_loss(sample1, sample2)
            logger.info("Iteration: %6d, %5s, Loss1: %9.6f, Loss2: %9.6f", i + 1, actual, loss1, loss2)

        torch.save(net.state_dict(), pkl)

    similar_correct = 0
    dissimilar_correct = 0
    similar_total = 0
    dissimilar_total = 0
    similar_dist = 0.0
    dissimilar_dist = 0.0

    for i, (s1, s2) in enumerate(test_data_loader):
        sample1 = {'feature': Variable(s1[0]), 'label': Variable(s1[1])}
        sample2 = {'feature': Variable(s2[0]), 'label': Variable(s2[1])}

        result, dist = net.is_similar(sample1['feature'], sample2['feature'])
        actual = int(sample1['label'].data) == int(sample2['label'].data)

        if actual:
            similar_total += 1
            if result:
                similar_correct += 1
            similar_dist += dist
        else:
            dissimilar_total += 1
            if not result:
                dissimilar_correct += 1
            dissimilar_dist += dist

        logger.info("%d: %5s, %f", i + 1, actual, dist)

    logger.info("Similar: Total: %d, Purity: %.3f, Average Distance: %.6f", similar_total, similar_correct / similar_total, similar_dist / similar_total)
    logger.info("Dissimilar: Total: %d, Purity: %.3f, Average Distance: %.6f", dissimilar_total, dissimilar_correct / dissimilar_total, dissimilar_dist / dissimilar_total)