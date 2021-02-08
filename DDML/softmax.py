import sys
import os
import math
import logging
from itertools import combinations
import torch
import torch.cuda as cuda
from torch import FloatTensor, LongTensor
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
    # formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
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

    def __init__(self, dataset, using_cuda=False):

        self.data = []

        if using_cuda:
            tensor = cuda.FloatTensor
        else:
            tensor = FloatTensor

        for s in dataset:
            self.data.append((tensor(s[0]) / 255, tensor(s[1])))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SVMDataset(Dataset):
    """
    Dataset for svm test.
    """

    def __init__(self, dataset, using_cuda=False):  # , size=None):

        self.features = []
        self.labels = []

        if using_cuda:
            tensor = cuda.FloatTensor
        else:
            tensor = FloatTensor

        # fixed
        for s in dataset:
            self.features.append(tensor(s[0]) / 255)
            self.labels.append(int(s[1][0]))

        # random
        # while len(self.features) < size:
        #     s = random.choice(TestDataset.dataset)
        #     self.features.append(tensor(s[0]) / 255)
        #     self.labels.append(int(s[1][0]))

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)


class DDMLNet(nn.Module):

    def __init__(self, layer_shape, classes_count, beta=1.0, tao=5.0, b=1.0, learning_rate=0.001, using_cuda=False):
        """

        :param layer_shape:
        :param beta:
        :param tao:
        :param learning_rate:
        """
        super(DDMLNet, self).__init__()

        self.layer_count = len(layer_shape)
        self.layers = []

        self.using_cuda = using_cuda

        for _ in range(self.layer_count - 1):
            stdv = math.sqrt(6.0 / (layer_shape[_] + layer_shape[_ + 1]))
            layer = nn.Linear(layer_shape[_], layer_shape[_ + 1])
            layer.weight.data.uniform_(-stdv, stdv)
            layer.bias.data.uniform_(0, 0)
            self.layers.append(layer)
            self.add_module("layer{}".format(_), layer)

        # Softmax
        self.softmax_layer = nn.Linear(layer_shape[-1], classes_count)
        self.softmax = nn.Softmax(dim=-1)

        self._s = F.tanh

        self.beta = beta
        self.tao = tao
        self.b = b
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)

        if self.using_cuda:
            self.cuda()

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

    def pairwise_loss(self, pairs):
        """
        Compute loss.
        -------------
        :param pairs: DataLoader of train data batch.
        :return:
        """

        loss = 0.0

        for si, sj in pairs:
            xi = Variable(si[0])
            xj = Variable(sj[0])
            yi = si[1]
            yj = sj[1]

            if int(yi) == int(yj):
                l = 1
            else:
                l = -1

            dist = self.compute_distance(xi, xj)
            c = self.b - l * (self.tao - dist)
            loss += self._g(c)

        loss = loss / 2

        self.logger.debug("Loss = %f", loss)

        return loss

    def compute_gradient(self, pairs):
        """

        :param pairs:
        :return: gradient.
        """
        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())[0:2 * (self.layer_count - 1)]
        params_W = params[0::2]
        params_b = params[1::2]

        # calculate z(m) and h(m)
        # z(m) is the output of m-th layer without function tanh(x)
        z_i_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(pairs))]
        h_i_m = [[0 for m in range(self.layer_count)] for index in range(len(pairs))]
        z_j_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(pairs))]
        h_j_m = [[0 for m in range(self.layer_count)] for index in range(len(pairs))]

        for index, (si, sj) in enumerate(pairs):
            xi = Variable(si[0].unsqueeze(0))
            xj = Variable(sj[0].unsqueeze(0))
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

        delta_ij_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(pairs))]
        delta_ji_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(pairs))]

        # M = layer_count - 1, then we also need to project 1,2,3 to 0,1,2
        M = self.layer_count - 1 - 1

        # calculate delta(M)
        for index, (si, sj) in enumerate(pairs):
            xi = Variable(si[0].unsqueeze(0))
            xj = Variable(sj[0].unsqueeze(0))
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
        for index in range(len(pairs)):
            for m in reversed(range(M)):
                delta_ij_m[index][m] = torch.mm(delta_ij_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_i_m[index][m])
                delta_ji_m[index][m] = torch.mm(delta_ji_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_j_m[index][m])

        # calculate partial derivative of W
        partial_derivative_W_m = [0 * params_W[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(pairs)):
                partial_derivative_W_m[m] += (delta_ij_m[index][m] * h_i_m[index][m].t()).t() + (delta_ji_m[index][m] * h_i_m[index][m].t()).t()

        # calculate partial derivative of b
        partial_derivative_b_m = [0 * params_b[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(pairs)):
                partial_derivative_b_m[m] += (delta_ij_m[index][m] + delta_ji_m[index][m]).squeeze()

        # combine two partial derivative vectors
        gradient = []

        for m in range(self.layer_count - 1):
            gradient.append(partial_derivative_W_m[m])
            gradient.append(partial_derivative_b_m[m])

        return gradient

    def pairwise_optimize(self, gradient):
        for i, param in enumerate(self.parameters()):
            if i < len(gradient):
                param.data.sub_(self.learning_rate * gradient[i].data)


def train(ddml_network, train_dataloader, class_count, epoch_number, using_cuda):
    logger = logging.getLogger(__name__)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss(size_average=True)
    optimizer = torch.optim.SGD(ddml_network.parameters(), lr=ddml_network.learning_rate)

    for epoch in range(epoch_number):

        loss_sum = 0.0

        logger.info("Epoch number: %d", epoch + 1)

        for index, (x, y) in enumerate(train_dataloader):

            pairs = list(combinations(zip(x, y), 2))

            # softmax backwards
            # x = Variable(x, requires_grad=True)
            # y = y.long()
            #
            # if using_cuda:
            #     one_hot = Variable(torch.zeros(train_dataloader.batch_size, class_count).scatter_(1, y.cpu(), torch.ones(y.size())).cuda())
            #
            # else:
            #     one_hot = Variable(torch.zeros(train_dataloader.batch_size, class_count).scatter_(1, y, torch.ones(y.size())))
            #
            # optimizer.zero_grad()
            # x = ddml_network.softmax_forward(x)
            # loss = criterion(x, Variable(y.squeeze()))
            # # loss = criterion(x, one_hot)
            # loss.backward()
            # optimizer.step()

            loss = 0.0

            for sx, sy in zip(x, y):
                sx = Variable(sx.unsqueeze(0), requires_grad=True)
                sy = Variable(sy.long())

                optimizer.zero_grad()
                sx = ddml_network.softmax_forward(sx)
                l = criterion(sx, sy)
                l.backward()
                optimizer.step()
                loss += l

            loss /= train_dataloader.batch_size

            # pairwise
            gradient = ddml_network.compute_gradient(pairs)
            ddml_network.pairwise_optimize(gradient)

            loss += ddml_network.pairwise_loss(pairs) / len(pairs)

            loss_sum += loss
            logger.info("Iteration: %6d, Loss: %7.3f, Average Loss: %10.6f", index + 1, loss, loss_sum / (index + 1))


def ddml_test(ddml_network, dataloader, classes):
    logger = logging.getLogger(__name__)

    similar_dist_sum = 0.0
    dissimilar_dist_sum = 0.0
    similar_incorrect = 0
    dissimilar_incorrect = 0
    similar_correct = 0
    dissimilar_correct = 0
    predictions = []
    trues = []
    num = 0

    distance_list = [[0 for l in classes] for l in classes]
    pairs_count = [[0 for l in classes] for l in classes]

    for x, y in dataloader:
        xi = Variable(x[0])
        yi = int(y[0])
        xj = Variable(x[1])
        yj = int(y[1])

        actual = (yi == yj)
        dist = ddml_network.compute_distance(xi, xj)
        result = (dist <= ddml_network.tao - ddml_network.b)

        distance_list[min(yi, yj)][max(yi, yj)] += dist
        pairs_count[min(yi, yj)][max(yi, yj)] += 1

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

        prediction_i = int(torch.max(ddml_network.softmax_forward(xi).data, 0)[1])
        prediction_j = int(torch.max(ddml_network.softmax_forward(xj).data, 0)[1])
        trues.append(yi)
        trues.append(yj)
        predictions.append(prediction_i)
        predictions.append(prediction_j)

        logger.info("%6d, %2d(%2d), %2d(%2d), %9.3f", num, int(yi), prediction_i, int(yj), prediction_j, dist)

    logger.info("Similar: Average Distance: %.6f", similar_dist_sum / (similar_correct + similar_incorrect))
    logger.info("Dissimilar: Average Distance: %.6f", dissimilar_dist_sum / (dissimilar_correct + dissimilar_incorrect))

    softmax_accuracy = accuracy_score(trues, predictions)

    cm = confusion_matrix(trues, predictions, labels=sorted(classes))

    return softmax_accuracy, similar_correct, similar_incorrect, dissimilar_correct, dissimilar_incorrect, distance_list, pairs_count, cm


def svm_test(X, Y, classes, split=5000):
    svc = svm.SVC(kernel='linear', C=10, gamma=0.1)

    train_x = X[0:split]
    train_y = Y[0:split]

    test_x = X[split:]
    test_y = Y[split:]

    svc.fit(train_x, train_y)

    predictions = svc.predict(test_x)

    accuracy = accuracy_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions, sorted(classes))

    return accuracy, cm


def main(train_epoch_number=2):
    # dataset_path = 'fashion-mnist-generation.csv'
    dataset_path = 'cifar10-sample.csv'
    # dataset_path = 'cifar-10-generate.csv'

    train_batch_size = 5
    train_test_split_index = 5000

    using_cuda = False

    logger = setup_logger(level=logging.INFO)

    if cuda.is_available():
        using_cuda = True
        logger.info("Using cuda!")

    dataset, classes = read_dataset(dataset_path)
    class_count = len(classes)
    ddml_dataset = DDMLDataset(dataset[:train_test_split_index], using_cuda=using_cuda)
    svm_dataset = DDMLDataset(dataset, using_cuda=using_cuda)

    # layer_shape = (784, 1568, 784, 392)
    layer_shape = (1024, 2048, 4096, 512)
    # layer_shape = (3072, 6144, 6144, 1536)

    # net = DDMLNet(layer_shape, classes_count=len(classes), beta=1.0, tao=15.0, b=2.0, learning_rate=0.001, using_cuda=using_cuda)
    net = DDMLNet(layer_shape, classes_count=len(classes), beta=1.0, tao=10.0, b=1.0, learning_rate=0.0005, using_cuda=using_cuda)

    pkl_path = "pkl/ddml({}-{}-{}-{}).pkl".format(layer_shape, net.beta, net.tao, net.b)
    txt = "pkl/ddml({}-{}-{}-{}).txt".format(layer_shape, net.beta, net.tao, net.b)

    if os.path.exists(pkl_path):
        state_dict = torch.load(pkl_path)
        net.load_state_dict(state_dict)
        logger.info("Load state from file.")

    #
    # training
    #
    train_dataloader = DataLoader(dataset=ddml_dataset, shuffle=True, batch_size=train_batch_size)
    train(net, train_dataloader, class_count, train_epoch_number, using_cuda)
    torch.save(net.state_dict(), pkl_path)

    #
    # testing
    #
    test_dataloader = DataLoader(dataset=ddml_dataset, shuffle=False, batch_size=2)
    test_result = ddml_test(net, test_dataloader, classes)
    softmax_accuracy, similar_correct, similar_incorrect, dissimilar_correct, dissimilar_incorrect, distance_list, pairs_count, cm = test_result

    logger.info("Softmax Classification: %.6f", softmax_accuracy)
    logger.info("\nConfusion Matrix:\n\t%6d\t%6d\n\t%6d\t%6d", similar_correct, similar_incorrect, dissimilar_incorrect, dissimilar_correct)

    #
    # svm testing
    #
    svm_dataloader = DataLoader(dataset=svm_dataset, shuffle=False)

    X = []
    Y = []
    for s in svm_dataloader:
        x = Variable(s[0])
        x = net(x)
        X.append(x.data.cpu().squeeze().numpy())
        y = int(s[1])
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    svm_accuracy, cm = svm_test(X, Y, list(classes), split=train_test_split_index)

    logger.info("SVM Accuracy: %f", svm_accuracy)
    logger.info("SVM Confusion Matrix: \n%s", cm)

    # writing result
    with open(txt, mode='a') as t:
        print("\n\nSoftmax Accuracy: {}".format(softmax_accuracy), file=t)
        print("Confusion Matrix:\n\t{:6d}\t{:6d}\n\t{:6d}\t{:6d}".format(similar_correct, similar_incorrect, dissimilar_incorrect, dissimilar_correct), file=t)

        print('   ', end='', file=t)
        for label in classes:
            print('{:^7}'.format(label), end='\t', file=t)

        for label1 in sorted(classes):
            print('\n{}: '.format(label1), end='', file=t)

            for label2 in sorted(classes):
                try:
                    v = '{:6.3f}'.format(distance_list[label1][label2] / pairs_count[label1][label2])
                except ZeroDivisionError:
                    try:
                        v = '{:6.3f}'.format(distance_list[label2][label1] / pairs_count[label2][label1])
                    except ZeroDivisionError:
                        v = '{:^7}'.format('NaN')

                print(v, end='\t', file=t)

        print('\n', file=t)
        print("SVM Accuracy: {}".format(svm_accuracy), file=t)
        print("SVM Confusion Matrix: \n{}".format(cm), file=t)


if __name__ == '__main__':
    # train_epoch_number = 0 means no train.
    main(train_epoch_number=1)
