import os
import torch.nn.functional as F
import torch.utils.data as tud
import sklearn
import random
import math
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from sklearn.preprocessing import KBinsDiscretizer


seed = 5112
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class soft_lay(nn.Module):
    def __init__(self, t=100, emb_siz=100):
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

class house_pred(nn.Module):
    def __init__(self, module, hidden=60,
                 emb_siz=100, inv=5, total=1000, num_feature=8):
        super(house_pred, self).__init__()
        self.module = module
        self.hidden = hidden
        self.emb_siz = emb_siz
        self.num_feature = num_feature

        self.inv = inv
        self.total = total

        if self.module == 'ewde' or self.module == 'efde':
            self.tim_emb_lay = nn.Linear(int(self.total*3), self.emb_siz,bias=False)

        if self.module == 'fe':
            self.tim_emb_lay = nn.Sequential(
                                             nn.Linear(1, self.emb_siz,bias=False))

        if self.module == 'direct':
            self.tim_emb_lay = nn.Linear(1,self.emb_siz)

        if self.module == 'ad':
            self.tim_emb_lay = soft_lay(t=0.5, emb_siz=self.emb_siz)


        self.module_lst = nn.ModuleList()

        for i in range(self.num_feature):
            self.module_lst.append(self.tim_emb_lay)

        self.fc = nn.Sequential(nn.Linear(self.emb_siz*self.num_feature, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 50),
                                    nn.ReLU(),
                                    nn.Linear(50, 25),
                                    nn.ReLU(),
                                    nn.Linear(25, 1))

        if self.module == 'direct':
            self.fc = nn.Sequential(nn.Linear(self.num_feature, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 50),
                                    nn.ReLU(),
                                    nn.Linear(50, 25),
                                    nn.ReLU(),
                                    nn.Linear(25, 1))

    def process_t(self, t_round, intv_min):
        total = self.total
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        t_round = F.one_hot(t_round,num_classes = total)

        comp_ten = torch.zeros(t_round.shape).to(device)

        t_round = torch.cat((comp_ten,t_round,comp_ten), dim = -1)
        t_fin = torch.zeros(t_round.shape).to(device)

        for i in range(int(intv_min)+1):
            t_temp = torch.roll(t_round, i, -1)
            t_fin = t_fin+t_temp

        for i in range(int(intv_min)+1):
            t_temp = torch.roll(t_round, -i, -1)
            t_fin = t_fin+t_temp

        t_fin = t_fin-t_round
        return t_fin.to(device)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y = x.float()
        # tim_emb_all = torch.zeros(y.shape[0],self.emb_siz).to(device)
        for i in range(self.num_feature):
            x = y[:,i]
            if self.module == 'ewde' or self.module == 'efde':
                x = x.long()
                tim_emb = self.module_lst[i](self.process_t(x, self.inv))
                # tim_emb = self.module_lst[i](x)

            if self.module == 'fe':
                tim_emb = self.module_lst[i](x.unsqueeze(-1))

            if self.module == 'direct':
                tim_emb = x.unsqueeze(-1)

            if self.module == 'ad':
                tim_emb = self.module_lst[i](x.unsqueeze(-1))

            if i == 0:
                tim_emb_all = tim_emb
            else:
                tim_emb_all = torch.cat((tim_emb_all,tim_emb), dim = -1)

        out = self.fc(tim_emb_all)

        return out.squeeze(-1).squeeze(-1)

def discrete(data_train, data_test,bins=100):
    if args.module == 'efde':
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        est.fit(data_train[:,:-1])
        data_train[:,:-1] = torch.from_numpy(est.transform(data_train[:,:-1]))
        data_test[:,:-1] = torch.from_numpy(est.transform(data_test[:,:-1]))
    if args.module == 'ewde':
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        est.fit(data_train[:, :-1])
        data_train[:, :-1] = torch.from_numpy(est.transform(data_train[:, :-1]))
        data_test[:, :-1] = torch.from_numpy(est.transform(data_test[:, :-1]))

    return data_train, data_test

def train(model, data, data_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data,data_test = discrete(data, data_test,args.bins)
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

            optimizer.step()
            count = count + 1

            if count % 10 == 0:
                _, acc= tesnet(model, data_test)
                acc_lst.append(acc)

                time_end = time.time()
                time_used = time_end - time_start
                if count%300==0:
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
        data_train = data[:15000, :-1]
        data_target = data[:15000, -1]
    else:
        data_train = data[15000:, :-1]
        data_target = data[15000:, -1]

    with torch.no_grad():
        out = model(data_train)

        out = inverse_norm(out, scaler)
        data_target = inverse_norm(data_target, scaler)

        res_rmse = rmse(out, data_target)
        res_mape = mape(out, data_target)
    return res_rmse, res_mape

def mape(out,tar):
    # tar = tar+0.0000001
    temp = out - tar
    # temp = temp.div(tar)
    temp = torch.abs(temp)
    temp_sum = (torch.abs(out) + torch.abs(tar))/2
    # temp_sum = torch.abs(tar)
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
    parser.add_argument('--emb_siz', type=int, default=50)
    parser.add_argument('--hid_siz', type=int, default=60)

    parser.add_argument('--dropout', '-drop', type=float, default=0.2)
    parser.add_argument('--method', '-method', type=str, default='general')
    parser.add_argument('--saveModelName', '-sm', type=str, default='best')
    parser.add_argument('--batch', '-batch', type=int, default=100)
    parser.add_argument('--module', type=str, default='efde',
                        help='ewde, fe, direct, efde, ad')
    parser.add_argument('--scalar', type=str, default='minmax', help='minmax, standard')
    parser.add_argument('--board', '-board', type=str, default='offBoard', help='onBoard, offBoard')
    parser.add_argument('--attribute', '-attr', type=str, default='t', help='attr to be predicted: o, d, t')
    parser.add_argument('--pred_time', '-pred_time', action='store_true')
    parser.add_argument('--mode', '-mode', type=str, default='train')
    parser.add_argument('--time_target', '-tim_tar', type=str, default='dis', help='output form of time: dis con')
    parser.add_argument('--epoch', '-epoch', type=int, default=20, help='number of training epoch')
    parser.add_argument('--intv', '-intv', type=float, default=0)  # 单位是小时
    parser.add_argument('--bins', '-bins', type=int, default=300)

    args = parser.parse_args()
    args.saveModelName = args.attribute + ',' + args.module + ',' + args.board + ',' + str(
        args.pred_time) + ',' + args.time_target

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing()
    if args.scalar == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    if args.scalar == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()

    data = torch.from_numpy(housing.data)
    target = torch.from_numpy(housing.target)

    data_all = torch.cat((data, target.unsqueeze(-1)), dim=-1)
    data_all = data_all[torch.randperm(data_all.shape[0])]

    num_of_train = int(data_all.shape[0] * 0.8)

    data_train = data_all[:num_of_train]
    data_test = data_all[num_of_train:]

    scaler.fit(data_train)
    data_train[:,-1] = torch.from_numpy(scaler.transform(data_train)[:,-1])
    data_test[:,-1] = torch.from_numpy(scaler.transform(data_test)[:,-1])
    if args.module == 'continuous' or args.module == 'direct' or args.module == 'ad':
        data_train[:, :-1] = torch.from_numpy(scaler.transform(data_train)[:, :-1])
        data_test[:, :-1] = torch.from_numpy(scaler.transform(data_test)[:, :-1])

    model = house_pred(module=args.module, total=args.bins*2,
                       inv=args.intv, hidden=args.hid_siz, emb_siz=args.emb_siz).cuda()

    dirpath = 'modelSave_task1/{}_scal_{}_b{}_intv{}_s{}_{}'. \
        format(args.module, args.scalar, args.bins, args.intv,
               seed, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
    os.makedirs(dirpath)
    if args.mode == "train":
        train(model,data_train, data_test)
    # if args.mode == 'test':
        model_save_path = os.path.join(dirpath, model.module + '.th')
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        res_rmse, res_mape = tesnet(model, data_test)
        print('module: ', args.module,'bins: ', args.bins, 'intv: ', args.intv, 'rmse: ', res_rmse, 'smape: ', res_mape)



