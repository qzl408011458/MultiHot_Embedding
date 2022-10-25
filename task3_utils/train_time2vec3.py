import argparse
import pickle
import random
import time
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim
import torch.nn.functional as F
from task3_utils.mydataset import myDataset
from task3_utils.LSTM_Model_tModule import LSTMClassifier
import os





# def discrete(data_train, data_test,bins=100):
#     train_shape = data_train.shape
#     test_shape = data_test.shape
#     data_train = data_train.reshape(-1, train_shape[-1])
#     data_test = data_test.reshape(-1, test_shape[-1])
#     if args.module == 'efde':
#         est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
#         est.fit(data_train)
#         data_train = torch.from_numpy(est.transform(data_train))
#         data_test = torch.from_numpy(est.transform(data_test))
#     if args.module == 'ewde':
#         est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
#         est.fit(data_train)
#         data_train = torch.from_numpy(est.transform(data_train))
#         data_test = torch.from_numpy(est.transform(data_test))
#     return data_train.reshape(train_shape), data_test.reshape(test_shape)

def test(best_path, model, fw_record, testloader):
    model.load_state_dict(torch.load(best_path))
    correct, total = 0, 0
    print('Start testing the best model')
    with torch.no_grad():
        model.eval()
        for i, batch_data in enumerate(testloader):
            X_batch = batch_data['data'].to(device)
            y_batch = batch_data['label'].to(device).long()

            y_pred = model(X_batch)
            class_predictions = F.log_softmax(y_pred, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (class_predictions == y_batch).sum().item()

    acc = correct / total
    print('-------------------------------------------------')
    print('-------------------------------------------------')
    test_logging = 'test_acc:{}'.format(acc)
    print('-------------------------------------------------')
    print('-------------------------------------------------')
    print(test_logging)
    fw_record.write(test_logging)
    return acc

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def transform_scalar(train_set_X, test_set_X):
    scalar = None
    if args.scalar == 'minmax':
        scalar = MinMaxScaler()
    if args.scalar == 'standard':
        scalar = StandardScaler()
    if scalar:
        train_shape = train_set_X.shape
        test_shape = test_set_X.shape
        train_set_X = train_set_X.reshape(-1, train_shape[-1])
        test_set_X = test_set_X.reshape(-1, test_shape[-1])
        scalar.fit(train_set_X)
        train_set_X = scalar.transform(train_set_X).reshape(train_shape)
        test_set_X = scalar.transform(test_set_X).reshape(test_shape)
    return train_set_X, test_set_X

def loadDataset():
    with open('task3_utils/dataset.pkl', 'rb') as fr:
        train_set_X, train_set_Y, test_set_X, test_set_Y = pickle.load(fr)
    train_set_X, test_set_X = transform_scalar(train_set_X, test_set_X)
    # if args.module != 'direct':
    #     train_set_X, test_set_X = discrete(train_set_X, test_set_X, args.bins)
    if not torch.is_tensor(train_set_X):
        train_set_X = torch.tensor(train_set_X)
    if not torch.is_tensor(test_set_X):
        test_set_X = torch.tensor(test_set_X)
    train_val_split = int(len(train_set_Y) * 0.9)
    trainset = myDataset(train_set_X[:train_val_split], train_set_Y[:train_val_split])
    valset = myDataset(train_set_X[train_val_split:], train_set_Y[train_val_split:])
    testset = myDataset(test_set_X, test_set_Y)
    return trainset, valset, testset

def train_test():
    print('=========================================================================')
    print('trial {} starts'.format(0))
    lr = 0.001
    n_epochs = args.epoch
    patience, patience_counter = 50, 0
    args.scalar = 'standard'
    args.bins = 200
    args.intv = 5
    args.hid_siz = 128
    print('params:')
    params_str = 'module_{}, scalar_{}, bins_{}, intv_{}, hid_size_{}'.format(
        args.module, args.scalar, args.bins, args.intv, args.hid_siz)
    print(params_str)
    print('=========================================================================')
    trainset, valset, testset = loadDataset()
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(valset, shuffle=False, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=False, batch_size=batch_size)
    seed = 6200
    seed_torch(seed)

    best_acc = 0
    model = LSTMClassifier(rnn=args.rnn, dropout=args.dropout, hidden_dim=args.hid_siz,
                           emb_size=args.emb_siz, module=args.module, bins=args.bins,
                           inv=args.intv, total=args.bins * 2)


    print(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('Start model training...')
    record_name = params_str
    fw_record = open(dirpath + '/{}.txt'.format(record_name), 'w', encoding='utf-8')

    for epoch in range(1, n_epochs + 1):
        # initialize losses
        loss_train_total = 0
        loss_val_total = 0
        # Training loop
        for i, batch_data in enumerate(trainloader):
            model.train()
            X_batch = batch_data['data'].to(device)
            y_batch = batch_data['label'].to(device).long()
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss_train_total += loss.cpu().detach().item() * batch_size
            loss.backward()
            optimizer.step()
        loss_train_total = loss_train_total / len(trainset)

        # Validation loop
        with torch.no_grad():
            for i, batch_data in enumerate(valloader):
                model.eval()
                X_batch = batch_data['data'].to(device)
                y_batch = batch_data['label'].to(device).long()

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss_val_total += loss.cpu().detach().item() * batch_size

        loss_val_total = loss_val_total / len(valset)

        # Validation Accuracy
        correct, total = 0, 0
        with torch.no_grad():
            model.eval()
            for i, batch_data in enumerate(valloader):
                X_batch = batch_data['data'].to(device)
                y_batch = batch_data['label'].to(device).long()
                y_pred = model(X_batch)
                class_predictions = F.log_softmax(y_pred, dim=1).argmax(dim=1)
                total += y_batch.size(0)
                correct += (class_predictions == y_batch).sum().item()
        acc = correct / total
        if epoch % 5 == 0:
            logging = f'Epoch: {epoch:3d}. Train Loss: {loss_train_total:.4f}. Val Loss: {loss_val_total:.4f} Acc.: {acc:2.2%}'
            print(logging)

        if acc > best_acc:
            patience_counter = 0
            best_acc = acc
            model_saveName = '{}_hid_{}_{}_scal_{}_b{}_intv{}_s{}.pth'. \
                format(args.rnn, args.hid_siz, args.module, args.scalar,
                       args.bins, args.intv, seed)
            best_path = os.path.join(dirpath, model_saveName)
            torch.save(model.state_dict(), best_path)
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
    test_acc = test(best_path, model, fw_record, testloader)
    fw_record.close()
    return test_acc

def main(tVec_dim, hid_siz):

    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--hid_siz', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.75)
    parser.add_argument('--method', type=str, default='general')
    parser.add_argument('--saveModelName', '-sm', type=str, default='best')
    parser.add_argument('--batch', '-batch', type=int, default=64)
    parser.add_argument('--rnn', type=str, default='gru', help='lstm or gru')
    parser.add_argument('--module', '-module', type=str, default='t2v',
                        )
    parser.add_argument('--scalar', type=str, default='minmax', help='select from minmax, standard or None')
    parser.add_argument('--tVec_dim', type=int, default=16)
    parser.add_argument('--emb_siz', type=int, default=300)
    parser.add_argument('--board', '-board', type=str, default='offBoard', help='onBoard, offBoard')
    parser.add_argument('--attribute', '-attr', type=str, default='t', help='attr to be predicted: o, d, t')
    parser.add_argument('--pred_time', '-pred_time', action='store_true')
    parser.add_argument('--mode', '-mode', type=str, default='train')
    parser.add_argument('--time_target', '-tim_tar', type=str, default='dis', help='output form of time: dis con')
    parser.add_argument('--epoch', '-epoch', type=int, default=100, help='number of training epoch')
    parser.add_argument('--multi', '-multi', type=int, help='select scenario')
    parser.add_argument('--bd', '-bd', type=str, help='yes no')
    parser.add_argument('--intv', '-intv', type=int, default=0)
    parser.add_argument('--bins', '-bins', type=int, default=0)

    global scalar, args, dirpath, device, batch_size

    args = parser.parse_args()
    args.tVec_dim = tVec_dim
    args.hid_siz = hid_siz
    batch_size = args.batch

    os.chdir('/home/vincentqin/PycharmProjects/Multihot_Embedding')

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # for dirname, _, filenames in os.walk('/kaggle/input'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device == torch.device('cuda'):
        print( torch.cuda.get_device_properties( device ) )

    seed=6200
    dirpath = 'modelSave_task3/{}_{}_scal_{}_b{}_intv{}_s{}_{}'. \
        format(args.rnn, args.module, args.scalar, args.bins, args.intv,
               seed, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
    os.makedirs(dirpath)

    train_test()


if __name__ == '__main__':
    main(tVec_dim=16, hid_siz=300)
