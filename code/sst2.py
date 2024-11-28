import os, sys, pickle, time, torch, random, utils
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import AutoTokenizer
from datasets import load_metric
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification


########## load SST-2 data set ##########
def load_sst2(path):
    df = pd.read_csv(path)
    return df


# Restart training the model each iteration on SST-2
class sst2_utils:
    def __init__(self, args):
        # parameter
        self.args = args
        self.name = 'SST2'
        self.train_times = 1
        self.batch_num = 128
        self.worker_num = 0
        self.learning_rate = 1e-5
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
        self.train_num = int(15000 / self.epoch_num)
        self.test_num = int(10000 / self.epoch_num)

        self.trainsets = load_sst2(self.data_path + '/csv_data/train.csv')
        self.testsets = load_sst2(self.data_path + '/csv_data/test.csv')

        # get index of sampling
        index_state = utils.getIndex(self.divide_type)

        train_path = self.tmp_path + '/{}_state{}_train({}).pkl'.format(self.name, index_state, self.epoch_num)
        test_path = self.tmp_path + '/{}_state{}_test({}).pkl'.format(self.name, index_state, self.epoch_num)

        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            self.generate_sample_index(index_state)

        self.train_slice = utils.load_data(train_path)
        self.test_slice = utils.load_data(test_path)

        # set AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # get the hyperparameters of the base model
    def get_paras(self):
        return self.epoch_num, self.train_times, self.learning_rate

    # get base model
    def get_model(self):
        # choose Distilbert-base-uncased
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
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
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        num_training_steps = self.train_times * len(dataloader)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                     num_training_steps=num_training_steps)

        model.train()
        for epoch in range(self.train_times):
            for batch in dataloader:
                # input data
                if self.args.gpu_model:
                    batch = {k: v.to(self.args.gpu_option) for k, v in batch.items()}
                else:
                    batch = {k: v.to('cpu') for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                # update parameters
                optimizer.step()
                lr_scheduler.step()

                # clear gradient
                optimizer.zero_grad()

        model.cpu()

        return model

    # test model
    def test_model(self, model, dataloader):
        if self.args.gpu_model:
            model = model.to(self.args.gpu_option)

        metric = load_metric("accuracy")
        model.eval()

        # define contingency table
        contingency = []
        for batch in dataloader:
            # input data
            if self.args.gpu_model:
                batch = {k: v.to(self.args.gpu_option) for k, v in batch.items()}
            else:
                batch = {k: v.to('cpu') for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

            for j in range(len(predictions)):
                # compare y and ^y
                if batch["labels"][j] == predictions[j]:
                    contingency.append(True)
                else:
                    contingency.append(False)

        # calculation top-1 accuracy
        correct_rate = metric.compute()['accuracy']

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
            for _, train_index in skf.split(self.trainsets['sentence'].values, self.trainsets['label'].values):
                train_slice.append(train_index.tolist())

            for _, test_index in skf.split(self.testsets['sentence'].values, self.testsets['label'].values):
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
        n_train_x = np.array([self.trainsets.iloc[i, 0] for i in train_slice])
        n_train_y = np.array([self.trainsets.iloc[i, 1] for i in train_slice])
        # get test subset
        n_test_x = np.array([self.testsets.iloc[i, 0] for i in test_slice])
        n_test_y = np.array([self.testsets.iloc[i, 1] for i in test_slice])

        # merge arrays -> transpose -> dataframe
        n_train = pd.DataFrame({'sentence': n_train_x, 'label': n_train_y})
        n_test = pd.DataFrame({'sentence': n_test_x, 'label': n_test_y})

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
                # combine data
                train = pd.DataFrame(np.vstack((self.train.values, n_train.values)))
                train.columns = ['sentence', 'label']

                test = pd.DataFrame(np.vstack((self.test.values, n_test.values)))
                test.columns = ['sentence', 'label']

                self.train = train
                self.test = test

        return self.train, self.test

    # tokenize function
    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence"], max_length=128, padding="max_length", truncation=True)

    # get dataloader: s_state-decide whether to shuffle the dataset;
    def get_dataloader(self, data, train=False):
        # convert from panda to dataset
        data = Dataset.from_pandas(data)
        tokenized_datasets = data.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        if self.worker_num >= 1:
            dataloader = DataLoader(dataset=tokenized_datasets, batch_size=self.batch_num, shuffle=train,
                                    num_workers=self.worker_num, pin_memory=True)
        else:
            dataloader = DataLoader(dataset=tokenized_datasets, batch_size=self.batch_num, shuffle=train)

        return dataloader

    # get the index of k-fold
    def get_k_fold(self, k):
        return utils.get_k_fold(self.train_num, self.test_num, k)

    # divide data according to k1
    def get_k_fold_data(self, k1, trainsets, testsets, train_index, test_index):
        n_train_x = []
        n_train_y = []
        n_test_x = []
        n_test_y = []

        for i in range(len(train_index)):
            if train_index[i] != k1:
                n_train_x.append(trainsets.iloc[i, 0])
                n_train_y.append(trainsets.iloc[i, 1])
            else:
                n_test_x.append(trainsets.iloc[i, 0])
                n_test_y.append(trainsets.iloc[i, 1])

        for i in range(len(test_index)):
            if test_index[i] != k1:
                n_train_x.append(testsets.iloc[i, 0])
                n_train_y.append(testsets.iloc[i, 1])
            else:
                n_test_x.append(testsets.iloc[i, 0])
                n_test_y.append(testsets.iloc[i, 1])

        # merge arrays -> transpose -> dataframe
        n_train = pd.DataFrame({'sentence': n_train_x, 'label': n_train_y})
        n_test = pd.DataFrame({'sentence': n_test_x, 'label': n_test_y})

        return n_train, n_test
