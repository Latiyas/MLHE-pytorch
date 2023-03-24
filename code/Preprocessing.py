import os, sys, pickle, time, torch, random, utils
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

if __name__ == '__main__':
    """
    expand data from SST-2 by resampling and scrambling
    """
    # raw data path
    source_path = "../data/SST2/tsv_data/"
    # processed data path
    save_path = "../data/SST2/csv_data/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # loop to convert tsv file to csv file
    with open(source_path + 'train.tsv', 'r', encoding='utf-8') as tsv_file:
        df = pd.read_csv(tsv_file, sep='\t')
        data_t, data_f = df[df['label'] == 1], df[df['label'] == 0]
        # combine data
        train = pd.DataFrame(np.vstack((data_t[:7500].values, data_f[:7500].values)))
        test = pd.DataFrame(np.vstack((data_t[7500:12500].values, data_f[7500:12500].values)))

        train.columns = ['sentence', 'label']
        test.columns = ['sentence', 'label']

        train = shuffle(train, random_state=1)
        test = shuffle(test, random_state=1)

        # save file
        train.to_csv(save_path + 'train.csv', index=False, sep=',')
        test.to_csv(save_path + 'test.csv', index=False, sep=',')

    print('end')
