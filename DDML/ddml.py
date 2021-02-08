import os
import random
import logging
import math
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
        # return len(self.features)
        return len(self.data)


class Net(nn.Module):

    def __init__(self, layer_shape, beta=0.5, tao=5.0, b=1.0, learning_rate=0.0001):
        """

        :param layer_shape:
        :param beta:
        :param tao:
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
        self.b = b
        self.learning_rate = learning_rate
        self.gradient = []
        self.logger = logging.getLogger(__name__)

    def _g(self, z):
        """
        Generalized logistic loss function.
        -----------------------------------
        :param z:
        """
        return float(math.log(1 + math.exp(self.beta * z)) / self.beta)

    def _g_derivative(self, z):
        """
        The derivative of g(z).
        -----------------------
        :param z:
        """
        return float(1 / (math.exp(-1 * self.beta * z) + 1))

    def _s_derivative(self, z):
        """
        The derivative of tanh(z).
        --------------------------
        :param z:
        """
        return 1 - self._s(z) ** 2

    def forward(self, feature):
        """
        Do forward.
        -----------
        :param feature: Variable, feature
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

            c = self.b - l * (self.tao - self.compute_distance(xi, xj))

            # h(m) have M + 1 values and m start from 0, in fact, delta_ij_m have M value and m start from 1
            delta_ij_m[index][M] = (self._g_derivative(c) * l * (h_i_m[index][M + 1] - h_j_m[index][M + 1])) * self._s_derivative(z_i_m[index][M])
            delta_ji_m[index][M] = (self._g_derivative(c) * l * (h_j_m[index][M + 1] - h_i_m[index][M + 1])) * self._s_derivative(z_j_m[index][M])

        # calculate delta(m)

        for index in range(len(dataloader)):
            for m in reversed(range(M)):
                delta_ij_m[index][m] = torch.mm(delta_ij_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_i_m[index][m])
                delta_ji_m[index][m] = torch.mm(delta_ji_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_j_m[index][m])

        # calculate partial derivative of W
        partial_derivative_W_m = [0 * params_W[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(dataloader)):
                partial_derivative_W_m[m] += (delta_ij_m[index][m] * h_i_m[index][m].t()).t() + (delta_ji_m[index][m] * h_i_m[index][m].t()).t()

        # calculate partial derivative of b
        partial_derivative_b_m = [0 * params_b[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(dataloader)):
                partial_derivative_b_m[m] += (delta_ij_m[index][m] + delta_ji_m[index][m]).squeeze()

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
        # loss2 = 0.0

        # J1
        for si, sj in dataloader:
            xi = Variable(si[0])
            xj = Variable(sj[0])
            yi = si[1]
            yj = sj[1]

            if int(yi) == int(yj):
                l = 1
            else:
                l = -1

            dist = self.compute_distance(xi, xj)
            # dist = (self(xi) - self(xj)).data.norm()
            c = self.b - l * (self.tao - dist)
            loss1 += self._g(c)

        loss1 = loss1 / 2

        self.logger.debug("J1 = %f", loss1)

        # J2
        # for p in list(self.parameters()):
        #     loss2 += p.data.norm()
        #
        # loss2 = self.lambda_ * loss2 / 2
        #
        # self.logger.debug("J2 = %f", loss2)

        return loss1

    def compute_distance(self, feature1, feature2):
        """
        Compute the distance of two samples.
        ------------------------------------
        :param feature1: Variable
        :param feature2: Variable
        :return: The distance of the two sample.
        """
        return (self(feature1) - self(feature2)).data.norm() ** 2


def main():
    test_label = 0

    train_epoch_number = 100
    train_batch_size = 100
    test_data_size = 10000

    layer_shape = (784, 1568, 196)

    # logger = setup_logger()
    logger = setup_logger(level=logging.INFO)

    test_data = DDMLDataset(label=test_label, size=test_data_size)
    test_data_loader = DataLoader(dataset=test_data)

    net = Net(layer_shape, beta=2.5, tao=20.0, b=5.0, learning_rate=0.0001)

    pkl = "pkl/ddml({}: {}-{}-{}).pkl".format(test_label, layer_shape, net.beta, net.tao)
    txt = "pkl/ddml({}: {}-{}-{}).txt".format(test_label, layer_shape, net.beta, net.tao)

    # if cuda.is_available():
    #     net.cuda()
    #     logger.info("Using cuda!")

    if os.path.exists(pkl):
        state_dict = torch.load(pkl)
        net.load_state_dict(state_dict)
        logger.info("Load state from file.")

    loss_sum = 0.0

    for epoch in range(train_epoch_number):
        train_data = DDMLDataset(label=test_label, size=train_batch_size)
        train_data_loader = DataLoader(dataset=train_data)
        net.compute_gradient(train_data_loader)
        net.backward()
        loss = net.compute_loss(train_data_loader)
        loss_sum += loss
        logger.info("Iteration: %6d, Loss: %6.3f, Average Loss: %6.3f", epoch + 1, loss, loss_sum / (epoch + 1))

    torch.save(net.state_dict(), pkl)

    #
    # test
    #

    similar_dist_sum = 0.0
    dissimilar_dist_sum = 0.0
    similar_incorrect = 0
    dissimilar_incorrect = 0
    similar_correct = 0
    dissimilar_correct = 0
    num = 0

    distance_list = [0 for l in DDMLDataset.labels]
    pairs_count = [0 for l in DDMLDataset.labels]

    for si, sj in test_data_loader:
        xi = Variable(si[0])
        yi = int(si[1])
        xj = Variable(sj[0])
        yj = int(sj[1])

        actual = (yi == yj)
        dist = net.compute_distance(xi, xj)
        result = (dist <= net.tao - net.b)

        distance_list[yj] += dist
        pairs_count[yj] += 1

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

        logger.info("%6d, %2d, %2d, %9.3f", num, int(yi), int(yj), dist)

    logger.info("Similar: Average Distance: %.6f", similar_dist_sum / (similar_correct + similar_incorrect))
    logger.info("Dissimilar: Average Distance: %.6f", dissimilar_dist_sum / (dissimilar_correct + dissimilar_incorrect))
    logger.info("\nConfusion Matrix:\n\t%6d\t%6d\n\t%6d\t%6d", similar_correct, similar_incorrect, dissimilar_incorrect, dissimilar_correct)

    with open(txt, mode='a') as t:

        print('Average Loss: {:6.3f}'.format(loss_sum / train_epoch_number), file=t)
        print("Confusion Matrix:\n\t{:6d}\t{:6d}\n\t{:6d}\t{:6d}".format(similar_correct, similar_incorrect, dissimilar_incorrect, dissimilar_correct), file=t)

        print('   ', end='', file=t)
        for label in DDMLDataset.labels:
            print('{:^7}'.format(label), end='\t', file=t)
        print('\n{}: '.format(test_label), end='', file=t)

        for l in DDMLDataset.labels:
            try:
                v = '{:.3f}'.format(distance_list[l] / pairs_count[l])
            except ZeroDivisionError:
                v = ' None '

            print(v, end='\t', file=t)

        print('\n', file=t)


if __name__ == '__main__':
    main()

# def main_old():
# distance_matrix = [[0 for l2 in DDMLDataset.labels] for l1 in DDMLDataset.labels]
# pairs_count = [[0 for l2 in DDMLDataset.labels] for l1 in DDMLDataset.labels]
#
# for si, sj in test_data_loader:
#     distance_matrix[l1][l2] += dist
#     pairs_count[l1][l2] += 1

# print('   ', end='')
# for label in DDMLDataset.labels:
#     print('{:^7}'.format(label), end='\t')
#
# print()
#
# for l1 in DDMLDataset.labels:
#     print(l1, end=': ')
#     for l2 in DDMLDataset.labels:
#         try:
#             v = '{:.3f}'.format(distance_matrix[l1][l2] / pairs_count[l1][l2])
#         except ZeroDivisionError:
#             v = ' None '
#         print(v, end='\t')
#     print()
