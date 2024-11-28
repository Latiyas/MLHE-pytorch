import os, sys, pickle, time, torch, random, utils
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, KFold
import torchvision.transforms as transforms
from PIL import Image

from models import *


########## load CIFAR-10 data set ##########
def load_cifar10(data_path):
    # store training data, 50000*3072
    x_train = []
    y_train = []
    # read five training files
    for i in range(5):
        x, t = load_batch_cifar10(data_path, "data_batch_%d" % (i + 1))
        x_train.append(x)
        y_train.append(t)
    # read test file
    x_test, y_test = load_batch_cifar10(data_path, "test_batch")
    # combine training files into a matrix
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    x_train = x_train.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)
    return x_train, y_train, x_test, y_test


# load data
def load_batch_cifar10(data_path, filename, dtype="float 64"):
    data_dir_cifar10 = data_path
    # composite file path
    path = os.path.join(data_dir_cifar10, filename)
    # open file
    fi = open(path, 'rb')
    # read
    batch = pickle.load(fi, encoding="bytes")
    fi.close()
    # get features and labels
    data = batch[b'data']
    labels = batch[b'labels']

    return data, labels


# create a class that inherits torch.utils.data.Dataset
class CIFAR10(Dataset):
    # initialize some parameters
    def __init__(self, imgs, img_label, transform=None):
        self.imgs = imgs
        self.img_label = img_label
        self.transform = transform

    def __getitem__(self, index):
        # if self.img_label is not None:
        # Channels First->Channels Last
        pic = self.imgs[index].transpose((1, 2, 0))

        pic_grab = pic.reshape(32, 32, 3)

        img = Image.fromarray(np.uint8(pic_grab))
        label = np.array(self.img_label[index], dtype=int)
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.from_numpy(label)

    def __len__(self):
        return len(self.imgs)


# Restart training the model each iteration on CIFAR-10
class cifar10_utils:
    def __init__(self, args):
        # parameter
        self.args = args
        self.name = 'CIFAR10'
        self.train_times = 20
        self.batch_num = 512
        self.worker_num = 0
        self.learning_rate = 0.1
        self.channel_size = 3
        self.picture_size = 32
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
        self.train_num = int(50000 / self.epoch_num)
        self.test_num = int(10000 / self.epoch_num)

        self.trainsets_x, self.trainsets_y, self.testsets_x, self.testsets_y = load_cifar10(self.data_path)

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
        # choose ResNet-18
        model = ResNet18()
        return model

    # split--Multithreading
    def mul_scatter(self, o_models, scatter_num, dataloader):
        # number of threads
        thread_num = 1

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
        n_models = []
        for i in range(model_num):
            n_model = self.train_model(tmp[i], dataloader)
            n_models.append(deepcopy(n_model))

        return n_models

    # train model
    def train_model(self, model, dataloader):
        if self.args.gpu_model:
            model = model.to(self.args.gpu_option)

        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        for epoch in range(self.train_times):
            for i, data in enumerate(dataloader):
                # input data
                inputs, labels = data
                if self.args.gpu_model:
                    inputs = inputs.to(self.args.gpu_option)
                    labels = labels.to(self.args.gpu_option)
                inputs, labels = Variable(inputs), Variable(labels)

                # clear gradient
                optimizer.zero_grad()

                # forward + backward
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()

                # update parameters
                optimizer.step()

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
        # random sampling (not put back)
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
        n_train_x = np.array([self.trainsets_x[i] for i in train_slice])
        n_train_y = np.array([self.trainsets_y[i] for i in train_slice])
        # get test subset
        n_test_x = np.array([self.testsets_x[i] for i in test_slice])
        n_test_y = np.array([self.testsets_y[i] for i in test_slice])

        return n_train_x, n_train_y, n_test_x, n_test_y

    # get data sets under different strategies
    def get_dataset(self, i):
        # replaceing old data;
        if self.divide_type in [1, 3, 5]:
            # get index based on different sampling
            train_slice = self.train_slice[i]
            test_slice = self.test_slice[i]
            n_train_x, n_train_y, n_test_x, n_test_y = self.get_Isubset(train_slice, test_slice)

            self.train_x = n_train_x
            self.train_y = n_train_y
            self.test_x = n_test_x
            self.test_y = n_test_y
        # replaying old data;
        elif self.divide_type in [2, 4, 6]:
            # get index based on different sampling
            train_slice = self.train_slice[i]
            test_slice = self.test_slice[i]
            n_train_x, n_train_y, n_test_x, n_test_y = self.get_Isubset(train_slice, test_slice)

            if i == 0:
                self.train_x = n_train_x
                self.train_y = n_train_y
                self.test_x = n_test_x
                self.test_y = n_test_y
            else:
                train_x = np.vstack((self.train_x, n_train_x))
                train_y = np.hstack((self.train_y, n_train_y))
                test_x = np.vstack((self.test_x, n_test_x))
                test_y = np.hstack((self.test_y, n_test_y))

                self.train_x = train_x
                self.train_y = train_y
                self.test_x = test_x
                self.test_y = test_y

        return [self.train_x, self.train_y], [self.test_x, self.test_y]

    # get dataloader: s_state-decide whether to shuffle the dataset;
    def get_dataloader(self, data, train=False):
        data_x, data_y = data
        if train == True:
            # preprocess data set
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            # create a dataset based on MyDataset
            datasets = CIFAR10(data_x, data_y, transform=transform_train)
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            datasets = CIFAR10(data_x, data_y, transform=transform_test)

        if self.worker_num >= 1:
            dataloader = DataLoader(dataset=datasets, batch_size=self.batch_num, shuffle=train,
                                    num_workers=self.worker_num, pin_memory=True)
        else:
            dataloader = DataLoader(dataset=datasets, batch_size=self.batch_num, shuffle=train)

        return dataloader

    # get the index of k-fold
    def get_k_fold(self, k):
        return utils.get_k_fold(self.train_num, self.test_num, k)

    # divide data according to k1
    def get_k_fold_data(self, k1, trainsets, testsets, train_index, test_index):
        trainsets_x, trainsets_y = trainsets
        testsets_x, testsets_y = testsets

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for i in range(len(train_index)):
            if train_index[i] != k1:
                train_x.append(trainsets_x[i].tolist())
                train_y.append(trainsets_y[i])
            else:
                test_x.append(trainsets_x[i].tolist())
                test_y.append(trainsets_y[i])

        for i in range(len(test_index)):
            if test_index[i] != k1:
                train_x.append(testsets_x[i].tolist())
                train_y.append(testsets_y[i])
            else:
                test_x.append(testsets_x[i].tolist())
                test_y.append(testsets_y[i])

        return [np.array(train_x), np.array(train_y)], [np.array(test_x), np.array(test_y)]
