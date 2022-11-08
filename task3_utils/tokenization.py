import random
import time

import sklearn
import torch
import numpy as np
import pickle
import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def loadDataset():
    with open('task3_utils/dataset.pkl', 'rb') as fr:
        train_set_X, train_set_Y, test_set_X, test_set_Y = pickle.load(fr)
    # scalar = MinMaxScaler()
    scalar = StandardScaler()
    train_shape = train_set_X.shape
    test_shape = test_set_X.shape

    train_set_X = train_set_X.reshape(-1, train_shape[-1])
    test_set_X = test_set_X.reshape(-1, test_shape[-1])

    scalar.fit(train_set_X)
    # train_set_X_ = scalar.transform(train_set_X)
    # test_set_X_ = scalar.transform(test_set_X)

    train_set_X_ = train_set_X
    test_set_X_ = test_set_X

    return train_set_X_, test_set_X_, scalar

def find_best_clusterNum():
    os.chdir('/home/vincentqin/PycharmProjects/Multihot_Embedding')
    train_set_X, test_set_X_, scalar = loadDataset()
    random.seed(6200)
    ind_list = list(range(len(train_set_X)))
    random.shuffle(ind_list)
    sample_inds = ind_list[: int(0.1 * len(ind_list))]
    # X_train_old = pd.read_csv('task3_utils/X_train.csv')

    train_set_X_sam = train_set_X[sample_inds]

    # max_score_cnum_list = list(range(10))
    max_score_cnum = 0.
    max_score = 0.

    for cluster_num in range(100, 10000, 100):
        t0 = time.time()
        km = KMeans(n_clusters=cluster_num, max_iter=50)
        km.fit(train_set_X_sam)
        score = silhouette_score(train_set_X_sam, km.labels_, metric='euclidean')
        if score > max_score:
            max_score = score
            max_score_cnum = cluster_num
        print(f'cluster_num:{cluster_num}, silhouette_score:{score}, time:{time.time() - t0}')

    print('--------------')
    print(max_score_cnum)


def corpus_generate(data):
    with open('task3_utils/corpus3.txt', 'w') as fw:
        for i in range(len(data)):
            line = ''
            for j in range(len(data[i])):
                if j != len(data[i]) - 1:
                    line += f'{data[i][j]} '
                else:
                    line += f'{data[i][j]}\n'

            fw.write(line)

def generate_corpus_tokens():

    seed = 6200
    cluster_num = 4800 # 4800
    random.seed(6200)
    os.chdir('/home/vincentqin/PycharmProjects/Multihot_Embedding')
    train_set_X, test_set_X, scalar = loadDataset()
    ind_list = list(range(len(train_set_X)))
    random.shuffle(ind_list)
    sample_inds = ind_list[: int(0.1 * len(ind_list))]

    t0 = time.time()
    train_set_X_sam = train_set_X[sample_inds]
    km = KMeans(n_clusters=cluster_num, max_iter=50, random_state=seed)
    km.fit(train_set_X_sam)
    score = silhouette_score(train_set_X_sam, km.labels_, metric='euclidean')
    print(f'cluster_num:{cluster_num}, silhouette_score:{score}, time:{time.time() - t0}')

    # generate corpus
    labeled_train_set_X = km.predict(train_set_X).reshape(-1, 128)
    labeled_test_set_X = km.predict(test_set_X).reshape(-1, 128)

    # corpus_x = labeled_train_set_X[: int(0.1 * len(labeled_train_set_X))]

    corpus_generate(labeled_train_set_X)

    with open('task3_utils/dataset.pkl', 'rb') as fr:
        train_set_X, train_set_Y, test_set_X, test_set_Y = pickle.load(fr)

    with open('task3_utils/dataset_tokens3.pkl', 'wb') as fw:
        data = (labeled_train_set_X, train_set_Y, labeled_test_set_X, test_set_Y)
        pickle.dump(data, fw)




if __name__ == '__main__':
    # find_best_clusterNum()
    generate_corpus_tokens()

