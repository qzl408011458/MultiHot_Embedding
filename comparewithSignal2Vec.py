import os




if __name__ == '__main__':
    # the procedures to get embeddings and train gru model with signal2vec include:
    # 1. run task3_utils/tokenization.py 2. run word2vec/word2vec.py 3. run task3_utils/genS2VDataset.py
    # 4. run the following command
    os.system('python train_task3.py --module s2v --epoch 1000')

