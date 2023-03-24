import os, sys, pickle, time, torch, random, utils
import copy
import numpy as np
from mnist import mnist_utils
from cifar10 import cifar10_utils
from sst2 import sst2_utils
from tinyimagenet import tinyimagenet_utils


# Monotone Learning
class MonotoneLearning:
    def __init__(self, args):
        # parameters
        self.args = args
        self.divide_type = args.divide_type
        self.o_state = self.args.output_model
        self.tmp_path = self.args.tmp_dir
        self.result_path = self.args.result_dir

        if self.args.dataset == 'MNIST':
            self.tools = mnist_utils(args)
        elif self.args.dataset == 'CIFAR10':
            self.tools = cifar10_utils(args)
        elif self.args.dataset == 'SST2':
            self.tools = sst2_utils(args)
        elif self.args.dataset == 'TinyImageNet':
            self.tools = tinyimagenet_utils(args)
        else:
            raise Exception("[!] There is no option for " + self.args.dataset)

    # Restart training the model each iteration
    def restarttraining(self):
        means = self.tools

        # save test error
        y_score = []

        for i in range(self.args.epochs):
            if self.args.process_model:
                utils.output_process(i)

            # get base model
            clf = means.get_model()

            # get training set and test set
            train, test = means.get_dataset(i)
            trainloader = means.get_dataloader(train, train=True)
            testloader = means.get_dataloader(test, train=False)

            h_set = means.train_model(clf, trainloader)

            bscore, _ = means.test_model(h_set, testloader)

            best_score = bscore

            y_score.append(best_score)

        return y_score

    # Retraining model each iteration
    def retraining(self):
        means = self.tools

        # save test error
        y_score = []

        # get base model
        n_set = means.get_model()

        for i in range(self.args.epochs):
            if self.args.process_model:
                utils.output_process(i)

            o_set = copy.deepcopy(n_set)

            # get training set and test set
            train, test = means.get_dataset(i)
            trainloader = means.get_dataloader(train, train=True)
            testloader = means.get_dataloader(test, train=False)

            n_set = means.train_model(o_set, trainloader)

            bscore, _ = means.test_model(n_set, testloader)

            best_score = bscore

            y_score.append(best_score)

        return y_score

    # Repetition MTsimple
    def MTsimple(self):
        means = self.tools

        # save test error
        y_score = []
        h_best = []

        for i in range(self.args.epochs):
            if self.args.process_model:
                utils.output_process(i)

            # get base model
            clf = means.get_model()

            # get training set and test set
            train, test = means.get_dataset(i)
            trainloader = means.get_dataloader(train, train=True)
            testloader = means.get_dataloader(test, train=False)

            h_set = means.train_model(clf, trainloader)

            nscore, _ = means.test_model(h_set, testloader)

            # initialize
            if i == 0:
                h_best = copy.deepcopy(h_set)
                best_score = nscore
            else:
                oscore, _ = means.test_model(h_best, testloader)

                # update the model according to the error
                if nscore > oscore:
                    h_best = copy.deepcopy(h_set)
                    best_score = nscore
                else:
                    best_score = oscore

            y_score.append(best_score)

        return y_score

    # Repetition MTht
    def MTht(self):
        means = self.tools

        # save test error
        y_score = []
        h_best = []

        for i in range(self.args.epochs):
            if self.args.process_model:
                utils.output_process(i)

            # get base model
            clf = means.get_model()

            # get training set and test set
            train, test = means.get_dataset(i)
            trainloader = means.get_dataloader(train, train=True)
            testloader = means.get_dataloader(test, train=False)

            h_set = means.train_model(clf, trainloader)

            nscore, ncontingency = means.test_model(h_set, testloader)

            # initialize
            if i == 0:
                h_best = copy.deepcopy(h_set)
                best_score = nscore
            else:
                oscore, ocontingency = means.test_model(h_best, testloader)

                # calculate mcnemar's test
                result_yy = 0
                result_yn = 0
                result_ny = 0
                result_nn = 0
                for j in range(len(ocontingency)):
                    if ocontingency[j] == True and ocontingency[j] == ncontingency:
                        result_yy += 1
                    elif ocontingency[j] == True and ocontingency[j] != ncontingency:
                        result_yn += 1
                    elif ocontingency[j] != True and ocontingency[j] == ncontingency:
                        result_nn += 1
                    else:
                        result_ny += 1

                mcnemar_result = utils.mcnemar_test([[result_yy, result_yn], [result_ny, result_nn]])

                # update the model according to mcnemar's test
                if mcnemar_result:
                    h_best = copy.deepcopy(h_set)
                    best_score = nscore
                else:
                    best_score = oscore

            y_score.append(best_score)

        return y_score

    # Repetition MTcv
    def MTcv(self):
        means = self.tools

        # K folds
        k = 5

        # save test error
        y_score = []
        h_bests = []

        train_indexes = []
        test_indexes = []

        for i in range(self.args.epochs):
            if self.args.process_model:
                utils.output_process(i)

            # get data
            otrain, otest = means.get_dataset(i)
            # divide K fold
            train_index, test_index = means.get_k_fold(k)
            # deal with A1 and A2 separately
            if self.divide_type == 1 or self.divide_type == 3 or self.divide_type == 5:
                train_indexes = train_index
                test_indexes = test_index
            else:
                train_indexes += train_index
                test_indexes += test_index

            h_sets = []
            nscores = []

            # initialize
            if i == 0:
                for j in range(k):
                    # get base model
                    clf = means.get_model()

                    # divide data according to j
                    train, test = means.get_k_fold_data(j, otrain, otest, train_indexes, test_indexes)

                    # get training set and test set
                    trainloader = means.get_dataloader(train, train=True)
                    testloader = means.get_dataloader(test, train=False)

                    h_set = means.train_model(clf, trainloader)

                    nscore, _ = means.test_model(h_set, testloader)

                    h_sets.append(h_set)
                    nscores.append(nscore)

                h_bests = copy.deepcopy(h_sets)
                best_score = max(nscores)
            else:
                oscores = []
                for j in range(k):
                    # get base model
                    clf = means.get_model()

                    train, test = means.get_k_fold_data(j, otrain, otest, train_indexes, test_indexes)

                    trainloader = means.get_dataloader(train, train=True)
                    testloader = means.get_dataloader(test, train=False)

                    h_set = means.train_model(clf, trainloader)

                    nscore, _ = means.test_model(h_set, testloader)

                    h_sets.append(h_set)
                    nscores.append(nscore)

                    oscore, _ = means.test_model(h_bests[j], testloader)
                    oscores.append(oscore)

                # update the model according to the mean error
                if np.mean(nscores) > np.mean(oscores):
                    h_bests = copy.deepcopy(h_sets)
                    best_score = max(nscores)
                else:
                    best_score = max(oscores)

            y_score.append(best_score)

        return y_score

    # Monotone Learning with Hypothesis Evolution
    def MLHE(self, split_num=2, retain_num=16):
        means = self.tools

        # y_score = []
        h_set = []
        c_data = []
        for i in range(self.args.epochs):
            if self.args.process_model:
                utils.output_process(i)

            # get training set and test set
            train, test = means.get_dataset(i)
            trainloader = means.get_dataloader(train, train=True)
            testloader = means.get_dataloader(test, train=False)

            # save test error
            scores = []

            # model update/save
            if i != 0:
                oset = copy.deepcopy(h_set)
                # split training
                h_scatter = means.mul_scatter(oset, split_num, trainloader)
                nset = copy.deepcopy(h_scatter)

                # test
                for j in range(len(nset)):
                    score, _ = means.test_model(nset[j], testloader)
                    scores.append(score)

                # sorted score
                s_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

                # retain
                if len(nset) <= retain_num:
                    h_set = copy.deepcopy(nset)
                else:
                    h_set = []
                    for j in range(retain_num):
                        # find high-accuracy hypotheses
                        h_set.append(nset[s_scores[j][0]])
            # initialize
            else:
                # training and testing the model based on current data
                h_scatter = means.mul_scatter([], split_num, trainloader)
                h_set = copy.deepcopy(h_scatter)

                # test
                for j in range(len(h_set)):
                    score, _ = means.test_model(h_set[j], testloader)
                    scores.append(score)

            # collecting result
            c_data.append(scores)

        return c_data