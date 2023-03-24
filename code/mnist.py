import os, sys, pickle, time, torch, random, utils
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import struct
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, KFold
import threading
from queue import Queue

from models import *


########## load mnist data set ##########
def load_mnist(path, kind):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), -1)
    return images, labels


# Restart training the model each iteration on MNIST
class mnist_utils:
    def __init__(self, args):
        # parameter
        self.args = args
        self.name = 'MNIST'
        self.train_times = 5
        self.batch_num = 256
        self.worker_num = 0
        self.learning_rate = 0.01
        self.channel_size = 1
        self.picture_size = 28
        # output status: 0-save data, 1-save picture
        self.o_state = self.args.output_model
        # data path
        self.data_path = '../data/{}'.format(self.name)
        # temp path
        self.tmp_path = self.args.tmp_dir + '/{}'.format(self.name)
        # result path
        self.result_path = self.args.result_dir
        # configure
        self.divide_type = self.args.divide_type
        self.epoch_num = self.args.epochs
        # size of training subset and test subset
        self.train_num = int(60000 / self.epoch_num)
        self.test_num = int(10000 / self.epoch_num)

        self.trainsets_x, self.trainsets_y = load_mnist(self.data_path, 'train')
        self.testsets_x, self.testsets_y = load_mnist(self.data_path, 't10k')

        # get index of sampling
        index_state = utils.getIndex(self.divide_type)

        train_path = self.tmp_path + '/{}_state{}_train({}).pkl'.format(self.name, index_state, self.epoch_num)
        test_path = self.tmp_path + '/{}_state{}_test({}).pkl'.format(self.name, index_state, self.epoch_num)

        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            self.generate_sample_index(index_state)

        self.train_slice = utils.load_data(train_path)
        self.test_slice = utils.load_data(test_path)

    # get the hyperparameters of the base model
    def get_paras(self):
        return self.epoch_num, self.train_times, self.learning_rate

    # get base model
    def get_model(self):
        # choose LeNet-5
        model = LeNet5()
        return model

    # split--Multithreading
    def mul_scatter(self, o_models, scatter_num, dataloader):
        # number of threads
        thread_num = 4

        # copy
        tmp = []
        if not o_models:
            clf = self.get_model()
            for j in range(scatter_num):
                tmp.append(deepcopy(clf))
        else:
            for o_model in o_models:
                for j in range(scatter_num):
                    tmp.append(deepcopy(o_model))

        # train
        model_num = len(tmp)
        loop_num = int(model_num / thread_num)
        n_models = []
        if loop_num > 0:
            for i in range(loop_num):
                # define multithreading
                q = Queue()
                threads = []
                for j in range(thread_num):
                    t = threading.Thread(target=self.batch_train, args=(tmp[int(i * thread_num + j)], dataloader, q))
                    t.start()
                    threads.append(t)
                for thread in threads:
                    thread.join()

                for _ in range(len(threads)):
                    n_models.append(deepcopy(q.get()))

        if model_num != loop_num * thread_num:
            # defining multithreading
            q = Queue()
            threads = []
            for k in range(loop_num * thread_num, model_num):
                t = threading.Thread(target=self.batch_train, args=(tmp[k], dataloader, q))
                t.start()
                threads.append(t)
            for thread in threads:
                thread.join()

            for _ in range(len(threads)):
                n_models.append(deepcopy(q.get()))

        return n_models

    # train model
    def train_model(self, model, dataloader):
        # if torch.cuda.is_available():
        #     model.cuda()
        if self.args.gpu_model:
            model = model.to(self.args.gpu_option)

        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)

        for epoch in range(self.train_times):
            # running_loss = 0.0
            for i, data in enumerate(dataloader):
                # input data
                inputs, labels = data
                # if torch.cuda.is_available():
                #     inputs = inputs.cuda()
                #     labels = labels.cuda()
                if self.args.gpu_model:
                    inputs = inputs.to(self.args.gpu_option)
                    labels = labels.to(self.args.gpu_option)
                inputs, labels = Variable(inputs), Variable(labels)

                # clear gradient
                optimizer.zero_grad()

                # forward + backward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # update parameters
                optimizer.step()

                # # log information
                # running_loss += loss.item()

        model.cpu()

        return model

    # test model
    def test_model(self, model, dataloader):
        if self.args.gpu_model:
            model = model.to(self.args.gpu_option)

        # count the number of samples that are correctly predicted
        correct_num = 0
        sample_num = 0

        # define contingency table
        contingency = []
        for i, data in enumerate(dataloader):
            # input data
            inputs, labels = data
            if self.args.gpu_model:
                inputs = inputs.to(self.args.gpu_option)
                labels = labels.to(self.args.gpu_option)
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            for j in range(len(predicted)):
                sample_num += 1
                predicted_num = predicted[j].item()
                label_num = labels[j].item()
                # compare y and ^y
                if predicted_num == label_num:
                    correct_num += 1
                    contingency.append(True)
                else:
                    contingency.append(False)

        # calculation top-1 accuracy
        correct_rate = correct_num / sample_num

        model.cpu()

        return correct_rate, contingency

    # batch training
    def batch_train(self, model, dataloader, q):
        if self.args.gpu_model:
            model = model.to(self.args.gpu_option)

        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)

        for epoch in range(self.train_times):
            # running_loss = 0.0
            for i, data in enumerate(dataloader):
                # input data
                inputs, labels = data
                # if torch.cuda.is_available():
                #     inputs = inputs.cuda()
                #     labels = labels.cuda()
                if self.args.gpu_model:
                    inputs = inputs.to(self.args.gpu_option)
                    labels = labels.to(self.args.gpu_option)
                inputs, labels = Variable(inputs), Variable(labels)

                # clear gradient
                optimizer.zero_grad()

                # forward + backward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # update parameters
                optimizer.step()

        model.cpu()

        # return result
        q.put(model)

    # generate sample index
    def generate_sample_index(self, state=0):
        # the sample index of training set and test set
        train_slice = []
        test_slice = []

        # sequential segmentation
        if state == 1:
            for i in range(self.epoch_num):
                train_slice.append(list(range(i * self.train_num, (i + 1) * self.train_num)))
                test_slice.append(list(range(i * self.test_num, (i + 1) * self.test_num)))
        # random sampling (put back)
        elif state == 2:
            for i in range(self.epoch_num):
                train_slice.append(random.sample(range(self.train_num * self.epoch_num), self.train_num))
                test_slice.append(random.sample(range(self.test_num * self.epoch_num), self.test_num))
        # stratified sampling (not put back)
        elif state == 3:
            skf = StratifiedKFold(n_splits=self.epoch_num, shuffle=True, random_state=5)
            for _, train_index in skf.split(self.trainsets_x, self.trainsets_y):
                train_slice.append(train_index.tolist())

            for _, test_index in skf.split(self.testsets_x, self.testsets_y):
                test_slice.append(test_index.tolist())
        else:
            raise Exception("[!] There is no option for " + state)

        # save index as .pkl file
        with open(self.tmp_path + '/{}_state{}_train({}).pkl'.format(self.name, state, self.epoch_num), 'wb') as f:
            pickle.dump(train_slice, f)
        with open(self.tmp_path + '/{}_state{}_test({}).pkl'.format(self.name, state, self.epoch_num), 'wb') as f:
            pickle.dump(test_slice, f)

    # get subsets by index
    def get_Isubset(self, train_slice, test_slice):
        # get training subset
        train_x = torch.from_numpy(np.array([self.trainsets_x[i] for i in train_slice]).reshape(
            (-1, self.channel_size, self.picture_size, self.picture_size))).float()
        train_y = torch.from_numpy(np.array([self.trainsets_y[i] for i in train_slice])).long()
        n_train = Data.TensorDataset(train_x, train_y)
        # get test subset
        test_x = torch.from_numpy(np.array([self.testsets_x[i] for i in test_slice]).reshape(
            (-1, self.channel_size, self.picture_size, self.picture_size))).float()
        test_y = torch.from_numpy(np.array([self.testsets_y[i] for i in test_slice])).long()
        n_test = Data.TensorDataset(test_x, test_y)

        return n_train, n_test

    # get data sets under different strategies
    def get_dataset(self, i):
        # replaceing old data;
        if self.divide_type in [1, 3, 5]:
            # get index based on different sampling
            train_slice = self.train_slice[i]
            test_slice = self.test_slice[i]
            n_train, n_test = self.get_Isubset(train_slice, test_slice)

            self.train = n_train
            self.test = n_test
        # replaying old data;
        elif self.divide_type in [2, 4, 6]:
            # get index based on different sampling
            train_slice = self.train_slice[i]
            test_slice = self.test_slice[i]
            n_train, n_test = self.get_Isubset(train_slice, test_slice)

            if i == 0:
                self.train = n_train
                self.test = n_test
            else:
                train_x = torch.cat((self.train[:][0], n_train[:][0]), dim=0)
                train_y = torch.cat((self.train[:][1], n_train[:][1]), dim=0)
                self.train = Data.TensorDataset(train_x, train_y)
                test_x = torch.cat((self.test[:][0], n_test[:][0]), dim=0)
                test_y = torch.cat((self.test[:][1], n_test[:][1]), dim=0)
                self.test = Data.TensorDataset(test_x, test_y)

        return self.train, self.test

    # get dataloader: s_state-decide whether to shuffle the dataset;
    def get_dataloader(self, data, train=False):
        datasets = data
        if self.worker_num >= 1:
            dataloader = DataLoader(dataset=datasets, batch_size=self.batch_num, shuffle=train,
                                    num_workers=self.worker_num, pin_memory=True)
        else:
            dataloader = DataLoader(dataset=datasets, batch_size=self.batch_num, shuffle=train)

        return dataloader

    # get the index of k-fold
    def get_k_fold(self, k):
        # total = self.train_num + self.test_num
        # # subblock size: total number/fold (rounded down)
        # fold_size = int(total / k)
        #
        # indexes = []
        #
        # for j in range(k - 1):
        #     tmp = [j for x in range(fold_size)]
        #     indexes += tmp
        #
        # indexes += [k - 1 for x in range(total - (fold_size * (k - 1)))]
        #
        # return indexes[:self.train_num], indexes[self.train_num:]
        return utils.get_k_fold(self.train_num, self.test_num, k)

    # divide data according to k1
    def get_k_fold_data(self, k1, trainsets, testsets, train_index, test_index):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in range(len(train_index)):
            if train_index[i] != k1:
                train_x.append(trainsets.tensors[0][i].numpy().tolist())
                train_y.append(trainsets.tensors[1][i].numpy().tolist())
            else:
                test_x.append(trainsets.tensors[0][i].numpy().tolist())
                test_y.append(trainsets.tensors[1][i].numpy().tolist())

        for i in range(len(test_index)):
            if test_index[i] != k1:
                train_x.append(testsets.tensors[0][i].numpy().tolist())
                train_y.append(testsets.tensors[1][i].numpy().tolist())
            else:
                test_x.append(testsets.tensors[0][i].numpy().tolist())
                test_y.append(testsets.tensors[1][i].numpy().tolist())

        n_train_x = torch.tensor(train_x)
        n_train_y = torch.tensor(train_y)
        n_train = Data.TensorDataset(n_train_x, n_train_y)
        n_test_x = torch.tensor(test_x)
        n_test_y = torch.tensor(test_y)
        n_test = Data.TensorDataset(n_test_x, n_test_y)

        return n_train, n_test
