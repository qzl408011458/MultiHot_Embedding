import pickle
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn.functional as F

with open('dataset_tokens.pkl', 'rb') as fr:
    train_set_X, train_set_Y, test_set_X, test_set_Y = pickle.load(fr)



# get embedding
with open('word_embedding_task2.txt', 'r') as fr:
    train_set_X, train_set_Y, test_set_X, test_set_Y = \
        torch.tensor(train_set_X, dtype=torch.long), train_set_Y, torch.tensor(test_set_X, dtype=torch.long), test_set_Y


    line = fr.readline().strip()
    tokens, emb_size = line.split(' ')
    tokens, emb_size = int(tokens), int(emb_size)
    line = fr.readline().strip()

    embed_matrix = np.zeros((tokens, emb_size))

    train_set_X_oh = F.one_hot(train_set_X, num_classes=tokens).to(torch.float64)
    test_set_X_oh = F.one_hot(test_set_X, num_classes=tokens).to(torch.float64)


    while line:
        l = line.split(' ')
        token_id, embeddding = l[0], l[1:]
        embed_matrix[int(token_id)] = np.array(embeddding, dtype=np.float)
        line = fr.readline().strip()
    print()
    embed_matrix = torch.tensor(embed_matrix)
    train_set_X_e = torch.matmul(train_set_X_oh, embed_matrix)
    test_set_X_e = torch.matmul(test_set_X_oh, embed_matrix)
    print()
    train_set_X = np.array(train_set_X_e)
    test_set_X = np.array(test_set_X_e)

    with open('dataset_s2v_t2.pkl', 'wb') as fw:
        data = (train_set_X, train_set_Y, test_set_X, test_set_Y)
        pickle.dump(data, fw)


