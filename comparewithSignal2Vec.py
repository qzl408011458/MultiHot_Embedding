import os




if __name__ == '__main__':
    # the procedures to get embeddings and train gru model with signal2vec include:
    # 1. run task3_utils/tokenization.py to get 'corpus.txt'
    # 2. run word2vec/word2vec.py to get 'word_embedding_task3.txt
    # 3. run task3_utils/genS2VDataset.py to get 'dataset_s2v_t3.pkl
    # 4. run the following command
    # note: task2 uses a similar way to obtain the embeddings
    task = 2
    if task == 3:
        # test_acc:0.7478991596638656
        os.system('python train_task3.py --module s2v --epoch 1000')
    if task == 2:
        # rmse: 463.10162142345297, smape: 70.91170247971436
        os.system('python train_task2.py --module s2v --epoch 100')



