import os, sys, pickle, time, torch, random, utils
import numpy as np
import matplotlib.pyplot as plt


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


# average learning curve
def Average_data(dataset):
    result = []
    for i in range(len(dataset)):
        if i == 0:
            result = dataset[i]
        else:
            result = np.sum([result, dataset[i]], axis=0)
    result = [x / len(dataset) for x in result]

    return result


# count the number of non-monotonic updatas
def Nonmonotonicity(data):
    n_monotone = 0
    for i in range(1, len(data)):
        if data[i - 1] > data[i]:
            n_monotone += 1
    return n_monotone


# output learning curve
def output_result(lc, name='picture', datasettype='MNIST'):
    mode = 'MLHE'

    if not lc:
        raise Exception("[!] Error ")

    al_dict = {mode+'(0.5)': 'r', mode+'(0.8)': 'g', mode+'(0.9)': 'b', 'retraining': 'r', 'IR': 'y', 'MTsimple': 'c',
               'MTht': 'darkviolet', 'MTcv': 'orchid'}

    x_list = range(lc['length'])

    for x in al_dict:
        if x in lc:
            y = lc[x]['data']
            plt.plot(x_list, y, color=al_dict[x], linestyle='-', label=x)

    plt.legend()
    if lc['length'] == 50:
        plt.xticks([0, 10, 20, 30, 40, 50])
    else:
        plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    if datasettype == 'MNIST':
        plt.ylim((0.05, 1.025))
    elif datasettype == 'CIFAR10':
        plt.ylim((0.1, 0.96))
    elif datasettype == 'SST2':
        plt.ylim((0.4, 0.975))
    elif datasettype == 'TinyImageNet':
        plt.ylim((0.15, 0.95))

    # save picture
    plt.savefig('../prediction_result/{}.jpg'.format(name), dpi=300)
    plt.close()


# load .pkl file
def load_data(data_path):
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def Analysis(datastate, datasettype='MNIST'):
    # data set type
    DatasetType = datasettype
    # different algorithms about monotone learning
    data_types = ['retraining', 'MTsimple', 'MLHE']
    # divide methods about data
    data_state = datastate

    # learning curve
    lc = {}
    monotones = []
    for data_type in data_types:
        path = '../user_data/tmp_data/{}/'.format(DatasetType)
        name = '{}_{}_state{}'.format(DatasetType, data_type, data_state)
        targets = utils.find_files(path, name)
        if len(targets) != 1:
            print("Warning: there are multiple matching files!")
            print("By default, {} is used for calculation.".format(targets[0]))

        tmp_path = path + targets[0]


        if not os.path.exists(tmp_path):
            continue

        raw_data = load_data(tmp_path)
        # print(len(raw_data))

        # MLHE algorithms
        if data_type == 'MLHE':
            raw_data = raw_data[0]
            data_len = len(raw_data)

            # set /delta
            for confidence in [0.5, 0.8, 0.9]:
                ruler = float(2 / (1 - confidence))
                result, n_monotone = Modified_curve(raw_data, ruler)

                temp = {'data': result, 'monotone': int((data_len - n_monotone) * 100 / data_len)}
                lc['{}({})'.format(data_type, confidence)] = temp
                # lc.append({data_type + str(confidence): temp})
                monotones.append(temp['monotone'])

        # IR algorithms
        elif data_type == 'retraining':
            raw_data = raw_data[0]
            data_len = len(raw_data)
            result = raw_data
            n_monotone = Nonmonotonicity(result)

            temp = {'data': result, 'monotone': int((data_len - n_monotone) * 100 / data_len)}
            lc['IR'] = temp
            # lc.append({data_type: temp})
            monotones.append(temp['monotone'])
        # MT algorithms
        else:
            data_len = len(raw_data[0])

            # average data
            result = Average_data(raw_data)
            n_monotone = Nonmonotonicity(result)

            temp = {'data': result, 'monotone': int((data_len - n_monotone) * 100 / data_len)}
            lc[data_type] = temp
            # lc.append({data_type: temp})
            monotones.append(temp['monotone'])

        lc['length'] = data_len
        lc['monotones'] = monotones

    return lc


if __name__ == '__main__':
    # experimental data set(the results of some data sets can be output separately after modification)
    datasets = ['MNIST', 'CIFAR10', 'SST2', 'TinyImageNet']

    # output pictures(outdated): 0-show; 1-save
    s_state = 1
    # final result path
    save_path = '../prediction_result/'

    for dataset in datasets:
        print('Probability of monotonic updates on {}.'.format(dataset))

        # # selectable scenarios: S1A1-4, S1A2-3, S2A1-6, S2A2-5
        options = [4, 6, 3, 5]
        for key in options:
            lc = Analysis(key, dataset)
            output_result(lc, '{}_{}'.format(dataset, key), dataset)
            print('Case {}:{}'.format(key, lc['monotones']))

        print("\n")

    print('end')
