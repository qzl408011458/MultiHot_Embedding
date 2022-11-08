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



def find_best_clusterNum():
    os.chdir('/home/vincentqin/PycharmProjects/Multihot_Embedding')
    with open('task2_utils/df.pkl', 'rb') as f:
        in_train, in_test, out_train, out_test = pickle.load(f)

    train_set_X, train_set_Y = in_train[:, :-1], in_train[:, -1]
    test_set_X, test_set_Y = in_test[:, :-1], in_test[:, -1]

    samples_train, seq = train_set_X.shape
    samples_test, seq = test_set_X.shape

    random.seed(5112)
    ind_list = list(range(len(train_set_X)))
    random.shuffle(ind_list)
    sample_inds = ind_list[: int(1 * len(ind_list))]


    train_set_X_sam = train_set_X[sample_inds].reshape(-1, 1)

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
    with open('task2_utils/corpus2.txt', 'w') as fw:
        for i in range(len(data)):
            line = ''
            for j in range(len(data[i])):
                if j != len(data[i]) - 1:
                    line += f'{data[i][j]} '
                else:
                    line += f'{data[i][j]}\n'

            fw.write(line)



def generate_corpus_tokens():
    seed = 5112
    cluster_num = 2580  # 2580
    random.seed(seed)
    os.chdir('/home/vincentqin/PycharmProjects/Multihot_Embedding')
    with open('task2_utils/df.pkl', 'rb') as f:
        in_train, in_test, out_train, out_test = pickle.load(f)

    train_set_X, train_set_Y = in_train[:, :-1], in_train[:, -1]
    test_set_X, test_set_Y = in_test[:, :-1], in_test[:, -1]

    samples_train, seq = train_set_X.shape
    samples_test, seq = test_set_X.shape

    ind_list = list(range(len(train_set_X)))
    random.shuffle(ind_list)
    sample_inds = ind_list[: int(1 * len(ind_list))]

    t0 = time.time()
    train_set_X_sam = train_set_X[sample_inds].reshape(-1, 1)
    km = KMeans(n_clusters=cluster_num, max_iter=50, random_state=seed)
    km.fit(train_set_X_sam)
    score = silhouette_score(train_set_X_sam, km.labels_, metric='euclidean')
    print(f'cluster_num:{cluster_num}, silhouette_score:{score}, time:{time.time() - t0}')

    # generate corpus
    labeled_train_set_X = km.predict(train_set_X.reshape(-1, 1)).reshape(samples_train, seq)
    labeled_test_set_X = km.predict(test_set_X.reshape(-1, 1)).reshape(samples_test, seq)



    # corpus_x = labeled_train_set_X[: int(0.1 * len(labeled_train_set_X))]

    corpus_generate(labeled_train_set_X)

    with open('task2_utils/dataset_tokens2.pkl', 'wb') as fw:
        data = (labeled_train_set_X, train_set_Y, labeled_test_set_X, test_set_Y)
        pickle.dump(data, fw)

    # with open('task2_utils/centroid.pkl', 'wb') as fw:
    #     pickle.dump(km.cluster_centers_, fw)


if __name__ == '__main__':
    # find_best_clusterNum()
    generate_corpus_tokens()




