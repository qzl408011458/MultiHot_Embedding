import os
import platform

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

seed = 5112  # 5112 # 51
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


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







class direct_lay(nn.Module):
    def __init__(self):
        super(direct_lay, self).__init__()
    def forward(self,x):
        return x.unsqueeze(-1)

class soft_lay(nn.Module):
    def __init__(self, t=10.0, emb_siz=100):
        super(soft_lay, self).__init__()
        self.t = t
        self.emb_siz = emb_siz

        self.lay1 = nn.Sequential(
            nn.Linear(1, int(args.bins), bias=False),
            nn.LeakyReLU())
        self.lin = nn.Linear(int(args.bins), int(args.bins), bias=False)
        self.lay2 = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Linear(int(args.bins), self.emb_siz, bias=False))

    def forward(self,x):
        y = self.lay1(x)
        y = self.lin(y)+0.1*y
        y = y*self.t
        y = self.lay2(y)
        return y

class flow_pred(nn.Module):
    def __init__(self, module, hidden=60,
                 emb_siz=200, inv=5, total=1000, num_station=None):
        super(flow_pred, self).__init__()
        self.module = module
        self.hidden = hidden
        self.emb_siz = emb_siz
        self.inv = inv
        self.total = total

        if self.module == 't2v':
            self.tlstm = TLSTM3(1, self.hidden)

        if self.module == 's2v':
            self.gru = nn.GRU(emb_siz, self.hidden, batch_first=True)


        if self.module == 'ewde' or self.module == 'efde':
            self.tim_emb_lay = nn.Linear(int(self.total*3), self.emb_siz,bias=False)
        if self.module == 'fe':
            self.tim_emb_lay = nn.Linear(1, self.emb_siz,bias=False)

        if self.module == 'ad':
            self.tim_emb_lay = soft_lay(t=0.91, emb_siz=self.emb_siz)
        self.gru = nn.GRU(self.emb_siz, self.hidden, batch_first = True)
        if self.module == 'direct':
            self.gru = nn.GRU(1, self.hidden, batch_first=True)


        self.fc = nn.Sequential(nn.Linear(self.hidden, 50),
                                nn.ReLU(),
                                nn.Linear(50, 50),
                                nn.ReLU(),
                                nn.Linear(50, 1))


    def process_t(self, t_round, intv_min):
        # multi-hot-embedding
        total = self.total
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t_round = F.one_hot(t_round, num_classes=total)
        comp_ten = torch.zeros(t_round.shape).to(device)
        t_round = torch.cat((comp_ten, t_round, comp_ten), dim=-1)
        t_fin = torch.zeros(t_round.shape).to(device)
        for i in range(int(intv_min) + 1):
            t_temp = torch.roll(t_round, i, -1)
            t_fin = t_fin + t_temp

        for i in range(int(intv_min) + 1):
            t_temp = torch.roll(t_round, -i, -1)
            t_fin = t_fin + t_temp
        t_fin = t_fin - t_round
        return t_fin.to(device)

    def forward(self, x):
        x = x.float()
        if 't2v' not in self.module and 's2v' not in self.module:
            if self.module == 'ewde' or self.module == 'efde':
                x = x.long()
                tim_emb = self.tim_emb_lay(self.process_t(x, self.inv))

            if self.module == 'fe':
                tim_emb = self.tim_emb_lay(x.unsqueeze(-1))

            if self.module == 'direct':
                tim_emb = x.unsqueeze(-1)
            if self.module == 'ad':
                tim_emb = self.tim_emb_lay(x.unsqueeze(-1))

            _, hid = self.gru(tim_emb)


        if self.module == 't2v':
            hid = self.tlstm(x.unsqueeze(-1))

        if self.module == 's2v':
            _, hid = self.gru(x)

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
    if args.module != 's2v':
        data,data_test = discrete(data, data_test,args.bins)
        data = data.to(device)
        data_test = data_test.to(device)
        data_train = data[:,:-1]
        data_target = data[:,-1]
    else:
        data_train, data_target = data
        data_train = torch.tensor(data_train, dtype=torch.float32).to(device)
        data_target = torch.tensor(data_target, dtype=torch.float32).to(device)



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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
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

    if args.mode == 'train':
        if args.module != 's2v':
            data = data.to(device)
            data_train = data[:int(len(data) / 3), :-1]
            data_target = data[:int(len(data) / 3), -1]
        else:
            data_train, data_target = data
            data_train = torch.tensor(data_train, dtype=torch.float32).to(device)
            # data_target = torch.tensor(data_target, dtype=torch.float32).to(device)
        # data_train = data[:1500, :-1]
        # data_target = data[:1500, -1]
    else:
        # data_train = data[1500:, :-1]
        # data_target = data[1500:, -1]
        if args.module != 's2v':
            data = data.to(device)
            data_train = data[:int(len(data) / 3), :-1]
            data_target = data[:int(len(data) / 3), -1]
        else:
            data_train, data_target = data
            data_train = torch.tensor(data_train, dtype=torch.float32).to(device)
            # data_target = torch.tensor(data_target, dtype=torch.float32).to(device)

    if args.module != 's2v':
        with torch.no_grad():
            out = model(data_train)
            out = inverse_norm(out, scaler)
            data_target = inverse_norm(data_target, scaler)
            res_rmse = rmse(out, data_target)
            res_mape = mape(out, data_target)
    else:
        with torch.no_grad():
            out = model(data_train)
            out = inverse_norm(out, scaler)
            if 'cuda' in str(data_target.device):
                data_target = data_target.to('cpu')

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='select args')
    parser.add_argument('--core', '-c', type=str, default='cpu', help='cpu or gpu')
    parser.add_argument('--path', '-p', type=str, default='raw_data', help=
    'path of the data file on the cluster')
    parser.add_argument('--fileName', '-f', nargs='+', type=str, default=['train_test_user.pkl',
                                                                          'batch_ch_user.pkl'],
                        help='file1: train_test_bd.pkl; file2: batch_ch_bd.pkl')
    parser.add_argument('--emb_siz', '-t', type=int, default=200)
    parser.add_argument('--hid_siz', '-hid', type=int, default=300)
    parser.add_argument('--dropout', '-drop', type=float, default=0.2)
    parser.add_argument('--method', '-method', type=str, default='general')
    parser.add_argument('--batch', '-batch', type=int, default=50)
    parser.add_argument('--module',  type=str, default='s2v',
                        help='ewde, fe, direct, efde, ad, t2v, s2v')
    parser.add_argument('--board', '-board', type=str, default='offBoard', help='onBoard, offBoard')
    parser.add_argument('--attribute', '-attr', type=str, default='t', help='attr to be predicted: o, d, t')
    parser.add_argument('--pred_time', '-pred_time', action='store_true')

    parser.add_argument('--mode', '-mode', type=str, default='train')

    parser.add_argument('--time_target', '-tim_tar', type=str, default='dis', help='output form of time: dis con')
    parser.add_argument('--epoch', '-epoch', type=int, default=100, help='number of training epoch')
    parser.add_argument('--multi', '-multi', type=int, help='select scenario')
    parser.add_argument('--bd', '-bd', type=str, help='yes no')
    parser.add_argument('--maxv', '-maxv', type=float, default=5000000)  # 单位是小时
    parser.add_argument('--minv', '-minv', type=float, default=0)
    parser.add_argument('--scalar', type=str, default='standard', help='minmax, standard')
    parser.add_argument('--bins', '-bins', type=int, default=300)
    parser.add_argument('--intv', '-intv', type=float, default=0)  # 单位是小时

    start_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    args = parser.parse_args()
    args.multi = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.module != 's2v':
        with open('task2_utils/df.pkl', 'rb') as f:
            in_train, in_test, out_train, out_test = pickle.load(f)

        if args.scalar == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
        if args.scalar == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()

        scaler.fit(in_train.reshape(-1).unsqueeze(-1).repeat(1,5))
        in_train[:, -1] = torch.from_numpy(scaler.transform(in_train)[:, -1])
        in_test[:, -1] = torch.from_numpy(scaler.transform(in_test)[:, -1])
        if args.module == 'fe' or args.module == 'direct' or args.module == 'ad':
            in_train[:, :-1] = torch.from_numpy(scaler.transform(in_train)[:, :-1])
            in_test[:, :-1] = torch.from_numpy(scaler.transform(in_test)[:, :-1])
    else:
        with open('task2_utils/df.pkl', 'rb') as f:
            in_train, in_test, out_train, out_test = pickle.load(f)
        if args.scalar == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
        if args.scalar == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()

        scaler.fit(in_train.reshape(-1, 1))
        with open('task2_utils/dataset_s2v_t2.pkl', 'rb') as fr:
            train_set_X, train_set_Y, test_set_X, test_set_Y = pickle.load(fr)

            train_set_Y = scaler.transform(train_set_Y.reshape(-1, 1)).reshape(-1)
            in_train = (train_set_X, train_set_Y)
            in_test = (test_set_X, test_set_Y)


    model = flow_pred(module=args.module, total=args.bins*2, inv=args.intv, hidden=args.hid_siz, emb_siz=args.emb_siz).cuda()
    dirpath = 'modelSave_task2/{}_scal_{}_b{}_intv{}_s{}_{}'. \
        format(args.module, args.scalar, args.bins, args.intv,
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

        test_logging = f'module:{args.module},bins:{args.bins}' \
                       f'intv:{args.intv},rmse:{res_rmse},smape:{res_mape}'
        # print('module: ', args.module,'bins: ', args.bins, 'intv: ', args.intv, 'rmse: ', res_rmse, 'smape: ', res_mape)
        print(test_logging)
        fw_record.write(test_logging)


