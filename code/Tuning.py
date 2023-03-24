import os, sys, pickle, time, torch, random, utils
import numpy as np
import matplotlib.pyplot as plt

mode = 'MLHE'


# boxplot-based learning curve
def Modified_curve(dataset, ruler):
    x = 0
    n_monotone = 0
    curve_y = []
    for i in range(len(dataset)):
        datas = dataset[i]

        # sort
        datas.sort(reverse=True)
        # number of iterations
        n = len(datas)

        # upper boundary Q_u=Q1
        q1 = (n + 1) / ruler
        if q1 < 1:
            Q1 = datas[0]
        elif q1 == int(q1):
            Q1 = datas[int(q1 - 1)]
        else:
            Q1 = datas[int(q1 - 1)] * (int(q1 + 1) - q1) + datas[int(q1)] * (q1 - int(q1))

        # lower boundaryQ_l=Q3
        q3 = (n + 1) * (ruler - 1) / ruler
        if q3 > n:
            Q3 = datas[int(n - 1)]
        elif q3 == int(q3):
            Q3 = datas[int(q3 - 1)]
        else:
            Q3 = datas[int(q3 - 1)] * (int(q3 + 1) - q3) + datas[int(q3)] * (q3 - int(q3))

        # judge
        if x <= Q3:
            x = Q3
            curve_y.append(x)
        elif x <= Q1 and x > Q3:
            curve_y.append(x)
        else:
            # non-monotonic update
            n_monotone += 1
            x = Q3
            curve_y.append(x)

    return curve_y, n_monotone


def Tuning(datastate, datasettype, split_num, retain_num):
    # data set type
    DatasetType = datasettype
    # different algorithms about monotone learning
    data_types = ['MLHE']
    # divide methods about data
    data_state = datastate

    # learning curve
    lc = {}
    monotones = []
    for data_type in data_types:
        path = '../user_data/tmp_data/{}/'.format(DatasetType)
        name = '{}_{}_state{}({},{})'.format(DatasetType, data_type, data_state, split_num, retain_num)
        targets = utils.find_files(path, name)
        if len(targets) == 0:
            print("Warning: there is no combination of ({}, {}) on {}.".format(split_num, retain_num, datasettype))
            continue
        elif len(targets) != 1:
            print("Warning: there are multiple matching files!")
            print("By default, {} is used for calculation.".format(targets[0]))

        tmp_path = path + targets[0]

        data = utils.load_data(tmp_path)
        # print(len(data))

        # MLHE algorithms
        if data_type == 'MLHE':
            raw_data = data[0]
            data_len = len(raw_data)

            # set /delta
            for confidence in [0.5, 0.8, 0.9]:
                ruler = float(2 / (1 - confidence))
                result, n_monotone = Modified_curve(raw_data, ruler)

                temp = {'data': result, 'monotone': int((data_len - n_monotone) * 100 / data_len)}
                lc['{}({})'.format(data_type, confidence)] = temp
                # lc.append({data_type + str(confidence): temp})
                monotones.append(temp['monotone'])

        lc['length'] = len(data[0])
        lc['monotones'] = monotones

    return lc


if __name__ == '__main__':
    # experimental data set(the results of some data sets can be output separately after modification)
    datasets = ['MNIST', 'CIFAR10', 'SST2', 'TinyImageNet']
    orders = [[2, 4], [2, 8], [2, 16], [2, 32], [3, 9], [3, 27], [4, 16]]

    for dataset in datasets:
        print('Probability of monotonic updates on {}.'.format(dataset))

        # # selectable scenarios: S1A1-4, S1A2-3, S2A1-6, S2A2-5
        options = [4, 6]
        for key in options:
            print("Case{}:".format(key))
            for order in orders:
                lc = Tuning(key, dataset, order[0], order[1])
                if lc:
                    print('({},{}):{}'.format(order[0], order[1], lc['monotones']))
            print("\n")

    print('end')
