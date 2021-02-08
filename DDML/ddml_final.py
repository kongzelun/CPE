import os
import math
import random
import logging
from itertools import combinations_with_replacement
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm


def setup_logger(level=logging.DEBUG):
    """
    Setup logger.
    -------------
    :param level:
    :return: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def read_dataset(file_path):
    """
    Read and preprocess dataset.
    :param file_path: dataset file path
    :return: dataset and classes set.
    """
    dataset = []
    labels = set()

    with open(file_path) as f:
        for line in f:
            row = [float(_) for _ in line.split(',')]
            dataset.append((row[:-1], row[-1:]))
            labels.add(int(row[-1]))

    return dataset, labels


class DDMLDataset(Dataset):
    """
    Implement a Dataset.
    """

    def __init__(self, dataset, device):
        """

        :param dataset: numpy.ndarray
        :param device: torch.device
        """

        self.data = []

        for s in dataset:
            self.data.append((tensor(s[:-1], dtype=torch.float, device=device) / 255,
                              tensor(s[-1], dtype=torch.long, device=device)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DDMLNet(nn.Module):

    def __init__(self, layer_shape, class_count, beta=1.0, tao=5.0, b=1.0, learning_rate=0.001):
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

        # Softmax
        self.softmax_layer = nn.Linear(layer_shape[-1], class_count)
        self.softmax = nn.Softmax(dim=-1)

        self._s = F.tanh

        self.beta = beta
        self.tao = tao
        self.b = b
        self.learning_rate = learning_rate

        self.logger = logging.getLogger(__name__)

    def _g(self, z):
        """
        Generalized logistic loss function.
        -----------------------------------
        :param z:
        """
        return math.log(1 + math.exp(self.beta * z)) / self.beta

    def _g_derivative(self, z):
        """
        The derivative of g(z).
        -----------------------
        :param z:
        """
        try:
            value = 1 / (math.exp(-1 * self.beta * z) + 1)
        except OverflowError:
            value = 0

        return value

    def _s_derivative(self, z):
        """
        The derivative of tanh(z).
        --------------------------
        :param z:
        """
        return 1 - self._s(z) ** 2

    def forward(self, inputs):
        """
        Do forward.
        -----------
        :param inputs: Variable, feature
        :return:
        """

        # project features through net
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = self._s(x)

        return x

    def softmax_forward(self, inputs):
        """
        Do softmax forward.
        -------------------
        :param inputs: Variable, feature
        :return:
        """

        x = self(inputs)
        x = self.softmax_layer(x)
        x = self._s(x)
        x = self.softmax(x)

        return x

    def compute_distance(self, input1, input2):
        """
        Compute the distance of two samples.
        ------------------------------------
        :param input1: Variable
        :param input2: Variable
        :return: The distance of the two sample.
        """
        return (self(input1) - self(input2)).data.norm() ** 2

    def ddml_optimize(self, pairs):
        """

        :param pairs:
        :return: loss.
        """
        loss = 0.0

        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())[0:2 * (self.layer_count - 1)]
        params_W = params[0::2]
        params_b = params[1::2]

        # calculate z(m) and h(m)
        # z(m) is the output of m-th layer without function tanh(x)
        # h(m) start from 0, which is m-1
        z_i_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(pairs))]
        h_i_m = [[0 for m in range(self.layer_count)] for index in range(len(pairs))]
        z_j_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(pairs))]
        h_j_m = [[0 for m in range(self.layer_count)] for index in range(len(pairs))]

        for index, (si, sj) in enumerate(pairs):
            xi = si[0]
            xj = sj[0]
            h_i_m[index][-1] = xi
            h_j_m[index][-1] = xj
            for m in range(self.layer_count - 1):
                layer = self.layers[m]
                xi = layer(xi)
                xj = layer(xj)
                z_i_m[index][m] = xi
                z_j_m[index][m] = xj
                xi = self._s(xi)
                xj = self._s(xj)
                h_i_m[index][m] = xi
                h_j_m[index][m] = xj

        # calculate delta_ij(m)
        # calculate delta_ji(m)
        delta_ij_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(pairs))]
        delta_ji_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(pairs))]

        # M = layer_count - 1, then we also need to project 1,2,3 to 0,1,2
        M = self.layer_count - 1 - 1

        # calculate delta(M)
        for index, (si, sj) in enumerate(pairs):
            xi = si[0]
            xj = sj[0]
            yi = si[1]
            yj = sj[1]

            # calculate c and loss
            if int(yi) == int(yj):
                l = 1
            else:
                l = -1

            dist = self.compute_distance(xi, xj)
            c = self.b - l * (self.tao - dist)
            loss += self._g(c)

            # h(m) have M + 1 values and m start from 0, in fact, delta_ij_m have M value and m start from 1
            delta_ij_m[index][M] = (self._g_derivative(c) * l * (h_i_m[index][M] - h_j_m[index][M])) * self._s_derivative(z_i_m[index][M])
            delta_ji_m[index][M] = (self._g_derivative(c) * l * (h_j_m[index][M] - h_i_m[index][M])) * self._s_derivative(z_j_m[index][M])

        loss /= len(pairs)

        # calculate delta(m)
        for index in range(len(pairs)):
            for m in reversed(range(M)):
                delta_ij_m[index][m] = torch.mm(delta_ij_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_i_m[index][m])
                delta_ji_m[index][m] = torch.mm(delta_ji_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_j_m[index][m])

        # calculate partial derivative of W
        partial_derivative_W_m = [0 for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(pairs)):
                partial_derivative_W_m[m] += torch.mm(delta_ij_m[index][m].t(), h_i_m[index][m - 1])
                partial_derivative_W_m[m] += torch.mm(delta_ji_m[index][m].t(), h_j_m[index][m - 1])

        # calculate partial derivative of b
        partial_derivative_b_m = [0 for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(pairs)):
                partial_derivative_b_m[m] += (delta_ij_m[index][m] + delta_ji_m[index][m]).squeeze()

        gradient = []

        # combine two partial derivative vectors
        for m in range(self.layer_count - 1):
            gradient.append(partial_derivative_W_m[m].data)
            gradient.append(partial_derivative_b_m[m].data)

        for i, param in enumerate(self.parameters()):
            if i < len(gradient):
                param.data.sub_(self.learning_rate * gradient[i])
            else:
                break

        return loss


def train(ddml_network, train_dataloader, class_count, epoch_number):
    logger = logging.getLogger(__name__)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss(size_average=True)
    optimizer = torch.optim.SGD(ddml_network.parameters(), lr=ddml_network.learning_rate)

    average_loss = 0.0

    for epoch in range(epoch_number):

        loss_sum = 0.0

        logger.info("Epoch number: %d", epoch + 1)

        for index, (x, y) in enumerate(train_dataloader):
            # x: [batch_size, feature_number]
            # y: [batch_size, 1]

            pairs = list(combinations_with_replacement(zip(x, y), 2))

            ####################
            # softmax backward #
            ####################
            softmax_loss = 0.0

            for sx, sy in zip(x, y):
                # sx: [feature_number]
                # sy: [1]
                sy = sy.long()

                # if using_cuda:
                #     sy = sy.unsqueeze(0)
                #     onehot = Variable(torch.zeros(1, class_count).scatter_(1, sy.data.cpu(), torch.ones(sy.size())).cuda())
                #     # onehot = Variable(torch.zeros(train_dataloader.batch_size, class_count).scatter_(1, y.cpu(), torch.ones(y.size())).cuda())
                # else:
                #     sy = sy.unsqueeze(0)
                #     onehot = Variable(torch.zeros(1, class_count).scatter_(1, sy.data, torch.ones(sy.size())))
                #     # onehot = Variable(torch.zeros(train_dataloader.batch_size, class_count).scatter_(1, y, torch.ones(y.size())))

                optimizer.zero_grad()
                sx = ddml_network.softmax_forward(sx)
                loss = criterion(sx, sy)
                # loss = criterion(sx, onehot)
                loss.backward()
                optimizer.step()
                softmax_loss += float(loss)

            softmax_loss /= train_dataloader.batch_size

            #################
            # ddml backward #
            #################
            ddml_loss = ddml_network.ddml_optimize(pairs)

            loss_sum += (softmax_loss + ddml_loss)
            average_loss = loss_sum / (index + 1)
            logger.info("Iteration: %6d, Softmax Loss: %6.3f, DDML Loss: %6.3f, Average Loss: %10.6f", index + 1, softmax_loss, ddml_loss, average_loss)

    return average_loss


def svm_test(x, y, classes, split=5000):
    """
    SVM test.
    ---------
    :param x: features, numpy.ndarray.
    :param y: classes, numpy.ndarray.
    :param classes: class set.
    :param split: train set count.
    :return: cm and accuracy.
    """
    svc = svm.SVC(kernel='linear', C=32, gamma=0.1)

    train_x = x[0:split]
    train_y = y[0:split]

    test_x = x[split:]
    test_y = y[split:]

    svc.fit(train_x, train_y)

    predictions = svc.predict(test_x)

    accuracy = accuracy_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions, sorted(classes))

    return accuracy, cm


def main(train_epoch_number=1):
    train_dataset_path = 'data/fashion-mnist-stream-train-s.csv'
    test_dataset_path = 'data/fashion-mnist-stream-test.csv'
    # dataset_path = 'data/cifar10-sample.csv'
    # dataset_path = 'data/cifar-10-generate.csv'

    train_batch_size = 10
    train_test_split_index = 5000

    logger = setup_logger(level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, train_classes = read_dataset(train_dataset_path)
    test_dataset, test_classes = read_dataset(test_dataset_path)
    # random.shuffle(dataset)
    class_count = 10  # len(classes)
    ddml_dataset = DDMLDataset(train_dataset, device)
    svm_dataset = DDMLDataset(test_dataset, device)

    layer_shape = (784, 3136, 3136, 784)
    # layer_shape = (1024, 2048, 4096, 4096, 2048, 1024)
    # layer_shape = (3072, 6144, 6144, 1536)

    net = DDMLNet(layer_shape, class_count=class_count, beta=1.0, tao=20.0, b=2.0, learning_rate=0.002)
    net.to(device)

    pkl_path = "pkl/ddml({}-{}-{}-{}).pkl".format(layer_shape, net.beta, net.tao, net.b)
    txt = "pkl/ddml({}-{}-{}-{}).txt".format(layer_shape, net.beta, net.tao, net.b)

    if os.path.exists(pkl_path):
        state_dict = torch.load(pkl_path)
        net.load_state_dict(state_dict)
        logger.info("Load state from file.")

    ############
    # training #
    ############
    train_dataloader = DataLoader(dataset=ddml_dataset, shuffle=True, batch_size=train_batch_size)
    average_loss = train(net, train_dataloader, class_count, train_epoch_number)
    torch.save(net.state_dict(), pkl_path)

    ############
    # svm test #
    ############
    svm_dataloader = DataLoader(dataset=svm_dataset, shuffle=False)

    svm_x = []
    svm_y = []

    for s in svm_dataloader:
        x = Variable(s[0])
        x = net(x)
        svm_x.append(x.data.cpu().squeeze().numpy())
        y = int(s[1])
        svm_y.append(y)

    svm_x = np.array(svm_x)
    svm_y = np.array(svm_y)
    svm_accuracy, cm = svm_test(svm_x, svm_y, [_ for _ in range(10)], split=train_test_split_index)

    logger.info("SVM Accuracy: %f", svm_accuracy)
    logger.info("SVM Confusion Matrix: \n%s", cm)

    ##################
    # writing result #
    ##################
    with open(txt, mode='a') as t:
        print("Average Loss: {}".format(average_loss), file=t)
        print("SVM Accuracy: {}".format(svm_accuracy), file=t)
        print("SVM Confusion Matrix: \n{}\n".format(cm), file=t)


if __name__ == '__main__':
    # train_epoch_number = 0 means no train.
    main(train_epoch_number=1)
