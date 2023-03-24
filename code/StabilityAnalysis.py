import os, sys, pickle, time, torch, random, utils
import numpy as np


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


def Analysis(datastate, datasettype, split_num, retain_num):
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
        if len(targets) != 1:
            print("Warning: there are multiple matching files!")
            print("By default, {} is used for calculation.".format(targets[0]))

        tmp_path = path + targets[0]

        if not os.path.exists(tmp_path):
            continue

        data = utils.load_data(tmp_path)

        R5 = []
        R8 = []
        R9 = []

        if len(data) == 1:
            print("Note: there is only one result!")

        for i in range(len(data)):
            raw_data = data[i]
            # MLHE algorithms
            if data_type == 'MLHE':
                data_len = len(raw_data)

                # set /delta
                for confidence in [0.5, 0.8, 0.9]:
                    ruler = float(2 / (1 - confidence))
                    result, n_monotone = Modified_curve(raw_data, ruler)

                    temp = {'data': result, 'monotone': int((data_len - n_monotone) * 100 / data_len)}
                    lc['{}_{}({})'.format(data_type, i, confidence)] = temp
                    # lc.append({data_type + str(confidence): temp})
                    monotones.append(temp['monotone'])
                    if confidence == 0.5:
                        R5.append(int((data_len - n_monotone) * 100 / data_len))
                    elif confidence == 0.8:
                        R8.append(int((data_len - n_monotone) * 100 / data_len))
                    else:
                        R9.append(int((data_len - n_monotone) * 100 / data_len))

            lc['mean5'] = np.mean(R5)
            lc['std5'] = np.std(R5)
            lc['mean8'] = np.mean(R8)
            lc['std8'] = np.std(R8)
            lc['mean9'] = np.mean(R9)
            lc['std9'] = np.std(R9)

    return lc


if __name__ == '__main__':
    # experimental data set(the results of some data sets can be output separately after modification)
    datasets = ['MNIST', 'CIFAR10', 'SST2', 'TinyImageNet']

    for dataset in datasets:
        if dataset != 'TinyImageNet':
            order = [2, 16]
        else:
            order = [2, 8]

        print('Probability of monotonic updates on {}.'.format(dataset))
        print('Split:{}, Retain:{}.'.format(order[0], order[1]))

        # # selectable scenarios: S1A1-4, S1A2-3, S2A1-6, S2A2-5
        options = [4, 6, 3, 5]
        for key in options:
            lc = Analysis(key, dataset, order[0], order[1])
            print('Case {}: 0.5mean={}, 0.5std={}, 0.8mean={}, 0.8std={}, 0.9mean={}, 0.9std={}'.format(key, lc['mean5'], lc['std5'], lc['mean8'], lc['std8'], lc['mean9'], lc['std9']))
            print("")
        print("\n")

    print('end')