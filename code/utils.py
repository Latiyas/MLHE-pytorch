import os, sys, pickle, time, torch, random
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar


# load .pkl file
def load_data(data_path):
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


# output result
def output_results(data, args, i):
    if args.output_model == 0:
        if args.execution_num == 1:
            name = "({}){}".format(args.execution_num, time.strftime("%m-%d-%H", time.localtime()))
        else:
            name = "({}_{}){}".format(i + 1, args.execution_num, time.strftime("%m-%d-%H", time.localtime()))

        if args.device_id != 'None':
            name = name + args.device_id

        if args.algo_type == 'MLHE':
            file_path = "/{}/{}_{}_state{}({},{}){}.pkl".format(args.dataset, args.dataset, args.algo_type,
                                                                args.divide_type, args.split_num, args.retain_num, name)
        else:
            file_path = "/{}/{}_{}_state{}{}.pkl".format(args.dataset, args.dataset, args.algo_type, args.divide_type,
                                                         name)
        # save as .pkl file
        with open(args.tmp_dir + file_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        # draw
        plt.figure(figsize=(7, 4))
        plt.plot(data, 'b', lw=1.5)
        plt.plot(data, 'ro')
        plt.grid(True)
        plt.axis('tight')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Learning curve of {} under the state{}'.format(args.algo_type, args.divide_type))
        plt.savefig(args.result_dir + '/{} under the state{}.png'.format(args.algo_type, args.divide_type))


# mcnemar's test
def mcnemar_test(table):
    result = mcnemar(table, exact=True)
    # confidence level
    alpha = 0.05
    if result.pvalue > alpha:
        return False
    else:
        return True


# output process status
def output_process(i):
    print("epoch:{}".format(i))
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))


# get data index according to division type
def getIndex(divide_type):
    if divide_type == 1 or divide_type == 2:
        index_state = 1
    elif divide_type == 3 or divide_type == 4:
        index_state = 2
    elif divide_type == 5 or divide_type == 6:
        index_state = 3
    else:
        raise Exception("[!] There is no option for " + divide_type)

    return index_state


# get the index of k-fold
def get_k_fold(train_num, test_num, k):
    total = train_num + test_num
    # subblock size: total number/fold (rounded down)
    fold_size = int(total / k)

    indexes = []

    for j in range(k - 1):
        tmp = [j for x in range(fold_size)]
        indexes += tmp

    indexes += [k - 1 for x in range(total - (fold_size * (k - 1)))]

    return indexes[:train_num], indexes[train_num:]


# look for .pkl files that begins with XX
def find_files(path, name):
    files = os.listdir(path)
    targets = []
    for f in files:
        if name in f and f.endswith('.pkl'):
            targets.append(f)
    return targets