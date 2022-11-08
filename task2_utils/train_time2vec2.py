import os
import platform
# sysstr = platform.system()
# if sysstr =="Windows":
#     import sys
#     sys.path.append('C:\\ana\\envs\\bert123\\Lib\\site-packages')

import torch.nn.functional as F
import torch.utils.data as tud
import sklearn
import random
import math
import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import time
from sklearn.preprocessing import KBinsDiscretizer




class TLSTM3(nn.Module):
    def __init__(self, in_dim, hid_dim, tVec_dim=16, device='cuda'):
        super(TLSTM3, self).__init__()
        self.fc_gate_i = nn.Sequential(
            nn.Linear(in_dim + hid_dim, hid_dim),
            nn.Sigmoid()
        )
        # self.fc_gate_f = nn.Sequential(
        #     nn.Linear(in_dim + hid_dim, hid_dim),
        #     nn.Sigmoid()
        # )

        self.fc_content_c_ = nn.Sequential(
            nn.Linear(in_dim + hid_dim, hid_dim),
            nn.Tanh()
        )

        self.fc_gate_o = nn.Sequential(
            nn.Linear(in_dim + hid_dim + tVec_dim + 1, hid_dim),
            nn.Sigmoid()
        )

        self.device = device
        self.hid_dim = hid_dim

        # Time2Vec modules
        self.fc_gate_t1 = nn.Sequential(
            nn.Linear(in_dim + tVec_dim + 1, hid_dim),
            nn.Sigmoid()
        )

        self.fc_gate_t2 = nn.Sequential(
            nn.Linear(in_dim + tVec_dim + 1, hid_dim),
            nn.Sigmoid()
        )

        self.emb_t_non_pd = nn.Linear(1, 1)
        self.emb_t_pd = nn.Linear(1, tVec_dim)


    def forward(self, x):
        bat, seq, feats = x.size()
        h = torch.zeros(bat, self.hid_dim, device=self.device)
        c = torch.zeros(bat, self.hid_dim, device=self.device)

        for j in range(seq):
            x_j = x[:, j]

            # time 2 vector
            t_ = torch.tensor([[seq - j]], dtype=torch.float32, device=self.device).repeat(bat, 1)
            t_j = torch.cat([self.emb_t_non_pd(t_), torch.sin(self.emb_t_pd(t_))], dim=-1)

            # time gate module
            gate_t1 = self.fc_gate_t1(torch.cat([x_j, t_j], dim=-1))
            gate_t2 = self.fc_gate_t2(torch.cat([x_j, t_j], dim=-1))

            # LSTM module
            gate_i = self.fc_gate_i(torch.cat([x_j, h], dim=-1))
            # gate_f = self.fc_content_f(torch.cat([x_j, h], dim=-1))

            content_c_ = self.fc_content_c_(torch.cat([x_j, h], dim=-1))

            content_c__ = (1 - gate_i * gate_t1) * c + gate_i * gate_t1 * content_c_

            c = (1 - gate_i) * c + gate_i * gate_t2 * content_c_

            # c = gate_i * c + gate_f * content_c_

            gate_o = self.fc_gate_o(torch.cat([x_j, t_j, h], dim=-1))
            h = gate_o * torch.tanh(content_c__)

        return h




class flow_pred(nn.Module):
    def __init__(self, module, hidden=60,
                 tVec_dim=16):
        super(flow_pred, self).__init__()
        self.module = module
        self.hidden = hidden



        self.tlstm = TLSTM3(1, self.hidden, tVec_dim)



        self.fc = nn.Sequential(nn.Linear(self.hidden, 50),
                                nn.ReLU(),
                                nn.Linear(50, 50),
                                nn.ReLU(),
                                nn.Linear(50, 1))


    def forward(self, x):
        x = x.float()

        hid = self.tlstm(x.unsqueeze(-1))


        out = self.fc(hid.squeeze())
        return out.squeeze(-1).squeeze(-1)

def discrete(data_train, data_test,bins=100):
    if args.module == 'efde':
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        est.fit(data_train.reshape(-1).unsqueeze(-1).repeat(1,4))
        data_train[:,:-1] = torch.from_numpy(est.transform(data_train[:,:-1]))
        data_test[:,:-1] = torch.from_numpy(est.transform(data_test[:,:-1]))
    if args.module == 'ewde':
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        est.fit(data_train.reshape(-1).unsqueeze(-1).repeat(1,4))
        data_train[:,:-1] = torch.from_numpy(est.transform(data_train[:, :-1]))
        data_test[:,:-1] = torch.from_numpy(est.transform(data_test[:, :-1]))
    return data_train, data_test

def train(model, data, data_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data,data_test = discrete(data, data_test,args.bins)
    data = data.to(device)
    data_test = data_test.to(device)
    data_train = data[:,:-1]
    data_target = data[:,-1]
    train_dataset = tud.TensorDataset(data_train, data_target)

    train_loader = tud.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True
    )

    loss_fn = nn.MSELoss()
    learning_rate = 0.0009
    # learning_rate = 0.2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = 0
    acc_lst = []
    time_start = time.time()
    for epoch in range(args.epoch):
        model.train()
        for batch_data, batch_tar in train_loader:
            model.train()
            out = model(batch_data)
            optimizer.zero_grad()
            loss = loss_fn(out, batch_tar.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            count = count + 1
            # % 10
            if count % 2 == 0:
                _, acc= tesnet(model, data_test)
                acc_lst.append(acc)

                time_end = time.time()
                time_used = time_end - time_start
                # % 300
                if count % 3 == 0:
                    print('epoch', epoch, 'time:', time_used, 'loss:', loss.item(), 'test_loss', acc)
                if len(acc_lst) > 2:

                        if acc_lst[-1] == min(acc_lst):
                            save_path = os.path.join(dirpath, model.module + '.th')
                            torch.save(model.state_dict(), save_path)

def tesnet(model, data):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    if args.mode == 'train':
        data_train = data[:int(len(data) / 3), :-1]
        data_target = data[:int(len(data) / 3), -1]
        # data_train = data[:1500, :-1]
        # data_target = data[:1500, -1]
    else:
        # data_train = data[1500:, :-1]
        # data_target = data[1500:, -1]
        data_train = data[int(len(data) / 3):, :-1]
        data_target = data[int(len(data) / 3):, -1]
    out = model(data_train)

    with torch.no_grad():
        out = model(data_train)
        out = inverse_norm(out, scaler)
        data_target = inverse_norm(data_target, scaler)
        res_rmse = rmse(out, data_target)
        res_mape = mape(out, data_target)
    return res_rmse, res_mape


def mape(out,tar):
    temp = out - tar
    temp = torch.abs(temp)
    temp_sum = (torch.abs(out) + torch.abs(tar))/2
    temp = temp.div(temp_sum)
    res = temp.sum().item()/temp.shape[0]*100
    return res


def rmse(out, data_target):
    res = out - data_target
    result = math.sqrt(res.mul(res).sum().item()/res.shape[0])
    return result

def inverse_norm(x, scaler):
    y = torch.zeros(x.shape[0],scaler.n_features_in_).cuda()
    y[:,-1] = x
    k = scaler.inverse_transform(y.cpu().numpy())
    res = k[:,-1]
    return torch.from_numpy(res)

def main(tVec_dim, hid_siz):
    seed = 5112  # 5112 # 51
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--core', '-c', type=str, default='cpu', help='cpu or gpu')
    parser.add_argument('--path', '-p', type=str, default='raw_data', help=
    'path of the data file on the cluster')
    parser.add_argument('--fileName', '-f', nargs='+', type=str, default=['train_test_user.pkl',
                                                                          'batch_ch_user.pkl'],
                        help='file1: train_test_bd.pkl; file2: batch_ch_bd.pkl')
    parser.add_argument('--tVec_dim', '-t', type=int, default=16)
    parser.add_argument('--hid_siz', '-hid', type=int, default=300)
    parser.add_argument('--dropout', '-drop', type=float, default=0.2)
    parser.add_argument('--method', '-method', type=str, default='general')
    parser.add_argument('--batch', '-batch', type=int, default=50)
    parser.add_argument('--module',  type=str, default='t2v')
    parser.add_argument('--board', '-board', type=str, default='offBoard', help='onBoard, offBoard')
    parser.add_argument('--attribute', '-attr', type=str, default='t', help='attr to be predicted: o, d, t')
    parser.add_argument('--pred_time', '-pred_time', action='store_true')

    parser.add_argument('--mode', '-mode', type=str, default='train')

    parser.add_argument('--time_target', '-tim_tar', type=str, default='dis', help='output form of time: dis con')
    parser.add_argument('--epoch', '-epoch', type=int, default=20, help='number of training epoch')
    parser.add_argument('--multi', '-multi', type=int, help='select scenario')
    parser.add_argument('--bd', '-bd', type=str, help='yes no')
    parser.add_argument('--scalar', type=str, default='standard', help='minmax, standard')

    start_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    global scaler, args, dirpath

    args = parser.parse_args()
    args.tVec_dim = tVec_dim
    args.hid_siz = hid_siz
    args.multi = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set the main project's path
    os.chdir('/home/vincentqin/PycharmProjects/Multihot_Embedding')
    with open('task2_utils/df.pkl', 'rb') as f:
        in_train, in_test, out_train, out_test = pickle.load(f)

    if args.scalar == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    if args.scalar == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()


    scaler.fit(in_train.reshape(-1).unsqueeze(-1).repeat(1,5))
    in_train[:, -1] = torch.from_numpy(scaler.transform(in_train)[:, -1])
    in_test[:, -1] = torch.from_numpy(scaler.transform(in_test)[:, -1])
    if args.module == 't2v':
        in_train[:, :-1] = torch.from_numpy(scaler.transform(in_train)[:, :-1])
        in_test[:, :-1] = torch.from_numpy(scaler.transform(in_test)[:, :-1])

    model = flow_pred(module=args.module, hidden=args.hid_siz, tVec_dim=args.tVec_dim).cuda()
    dirpath = 'modelSave_task2/{}_scal_{}_s{}_{}'. \
        format(args.module, args.scalar,
               seed, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
    os.makedirs(dirpath)

    record_name = 'record.txt'
    fw_record = open(dirpath + '/{}.txt'.format(record_name), 'w', encoding='utf-8')



    if args.mode == "train":
        train(model,in_train, in_test)
    # if args.mode == 'test':
        model_save_path = os.path.join(dirpath, model.module + '.th')
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        res_rmse, res_mape = tesnet(model, in_test)
        # print('module: ', args.module, 'rmse: ', res_rmse, 'smape: ', res_mape)

        test_logging = f'module:{args.module},' \
                       f'rmse:{res_rmse},smape:{res_mape}'
        print(test_logging)
        fw_record.write(test_logging)


if __name__ == '__main__':
    main(tVec_dim=16)