import os
import logging
from itertools import combinations_with_replacement
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import torchvision
# from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
# import ddml_fashion_mnist as ddml
# import ddml_cifar10 as ddml
import ddml_guardian_short as ddml


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


def train(net, dataloader, criterion, optimizer):
    logger = logging.getLogger(__name__)

    statistics_batch = len(dataloader) / 5

    cnn_loss = 0.0
    ddml_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        # onehot
        # target = torch.zeros(labels.shape[0], 10).scatter_(1, labels.long().view(-1, 1), 1).to(net.device)
        inputs, labels = inputs.to(net.device), labels.to(net.device)
        pairs = list(combinations_with_replacement(zip(inputs, labels), 2))

        ################
        # cnn backward #
        ################
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # bce
        # loss = criterion(outputs, target)
        # cross entropy
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        cnn_loss += loss.item()

        #################
        # ddml backward #
        #################
        ddml_loss += net.ddml_optimize(pairs)

        # print statistics
        if (i + 1) % statistics_batch == 0:
            logger.debug('%5d: nn loss: %.4f, ddml loss: %.4f', i + 1, cnn_loss / statistics_batch, ddml_loss / statistics_batch)
            cnn_loss = 0.0
            ddml_loss = 0.0


def test(net, dataloader):
    correct = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(net.device), labels.to(net.device)
        outputs = net(inputs)
        value, result = torch.max(outputs, dim=1)

        if result == labels:
            correct += 1

    return correct / len(dataloader)


def svm_test(net, trainloader, testloader):
    svm_test.svc = svm.SVC(kernel='linear', C=10, gamma=0.1)

    train_x = []
    train_y = []
    for x, y in trainloader:
        x, y = x.to(net.device), y.to(net.device)
        x = net.ddml_forward(x)
        x = x.to(torch.device('cpu'))
        train_x.append(x.squeeze().detach().numpy())
        train_y.append(int(y))

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    svm_test.svc.fit(train_x, train_y)

    test_x = []
    test_y = []
    for x, y in testloader:
        x, y = x.to(net.device), y.to(net.device)
        x = net.ddml_forward(x)
        x = x.to(torch.device('cpu'))
        test_x.append(x.squeeze().detach().numpy())
        test_y.append(int(y))

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    predictions = svm_test.svc.predict(test_x)

    accuracy = accuracy_score(test_y, predictions)
    cm = confusion_matrix(test_y, predictions, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

    return accuracy, cm


if __name__ == '__main__':
    LOGGER = setup_logger(level=logging.DEBUG)

    TRAIN_BATCH_SIZE = 4
    TRAIN_EPOCH_NUMBER = 100
    TRAIN_TEST_SPLIT_INDEX = 5000

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")

    #######################
    # torchvision dataset #
    #######################
    # transform = transforms.Compose([transforms.ToTensor()])
    #
    # trainset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=True, transform=transform, download=False)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=4)
    #
    # testset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=False, transform=transform, download=False)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=4)

    ###############
    # csv dataset #
    ###############
    DATASET = np.loadtxt(ddml.DATASET_PATH, delimiter=',')
    np.random.shuffle(DATASET)
    LOGGER.debug("Dataset shape: %s", DATASET.shape)

    TRAINSET = ddml.DDMLDataset(DATASET[:TRAIN_TEST_SPLIT_INDEX])
    TRAINLOADER = DataLoader(dataset=TRAINSET, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    SVM_TRAINLOADER = DataLoader(dataset=TRAINSET, batch_size=1, shuffle=True, num_workers=4)

    TESTSET = ddml.DDMLDataset(DATASET[TRAIN_TEST_SPLIT_INDEX:])
    TESTLOADER = DataLoader(dataset=TESTSET, batch_size=1, shuffle=False, num_workers=4)

    cnnnet = ddml.DDMLNet(device=DEVICE, beta=0.5, tao=10.0, b=2.0, learning_rate=0.001)

    cross_entropy = nn.CrossEntropyLoss()
    # bce = nn.BCELoss()
    sgd = optim.SGD(cnnnet.parameters(), lr=0.001, momentum=0.9)

    if os.path.exists(ddml.PKL_PATH):
        state_dict = torch.load(ddml.PKL_PATH)
        try:
            cnnnet.load_state_dict(state_dict)
            LOGGER.info("Load state from file %s.", ddml.PKL_PATH)
        except RuntimeError:
            LOGGER.error("Loading state from file %s failed.", ddml.PKL_PATH)

    for epoch in range(TRAIN_EPOCH_NUMBER):
        LOGGER.info("Trainset size: %d, Epoch number: %d", len(TRAINSET), epoch + 1)
        train(cnnnet, TRAINLOADER, criterion=cross_entropy, optimizer=sgd)
        torch.save(cnnnet.state_dict(), ddml.PKL_PATH)

        if (epoch + 1) % 5 == 0:
            LOGGER.info("Testset size: %d", len(TESTSET))
            nn_accuracy = test(cnnnet, TESTLOADER)
            LOGGER.info("Accuracy: %6f", nn_accuracy)
            svm_accuracy, svm_cm = svm_test(cnnnet, SVM_TRAINLOADER, TESTLOADER)
            LOGGER.info("SVM Accuracy: %6f", svm_accuracy)
            LOGGER.info("Confusion Matrix: \n%s", svm_cm)
            # torch.save(cnnnet.state_dict(), 'pkl/ddml-({:.4f}).pkl'.format(svm_accuracy))
