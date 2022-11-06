import torch.nn as nn
import torch
import torch.nn.functional as F

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

class LSTMClassifier(nn.Module):
    def __init__(self, rnn='lstm', input_dim=10, hidden_dim=256, num_layers=2, output_dim=9,
                 dropout=0, module=None, emb_size=100, bins=300, inv=5, total=200,
                 t=5.6, hid_layers=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.module = module
        self.rnn = rnn

        if self.module != 't2v' and self.module != 's2v':
            if self.module != 'direct':
                self.in_rl = in_RepreLayer(module, emb_size=emb_size, total=total, inv=inv,
                                           bins=bins, num_feature=input_dim, device='cuda', t=t, hid_layers=hid_layers)
                self.rnn_in_dim = input_dim*emb_size
            else:
                self.rnn_in_dim = input_dim
            if self.rnn == 'lstm':
                self.lstm = nn.LSTM(input_size=self.rnn_in_dim, hidden_size=hidden_dim,
                                    num_layers=num_layers, batch_first=True, dropout=dropout)
            if self.rnn == 'gru':
                self.gru = nn.GRU(input_size=self.rnn_in_dim, hidden_size=hidden_dim,
                                    num_layers=num_layers, batch_first=True, dropout=dropout)

        else:
            if self.module == 't2v':
                self.tlstm = TLSTM3(input_dim, hidden_dim)
            if self.module == 's2v':
                self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_dim,
                                    num_layers=num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        if self.module != 't2v' and self.module != 's2v':
            if self.module != 'direct':
                X = self.in_rl(X)
            if self.rnn == 'lstm':
                hidden_features, (h_n, c_n) = self.lstm(X)  # (h_0, c_0) default to zeros
            if self.rnn == 'gru':
                hidden_features, h_n = self.gru(X)
            hidden_features = hidden_features[:, -1, :]
        if self.module == 't2v':
            hidden_features = self.tlstm(X)
        if self.module == 's2v':
            hidden_features, h_n = self.gru(X)
            hidden_features = hidden_features[:, -1, :]

          # index only the features produced by the last LSTM cell
        out = self.fc(hidden_features)
        return out

class Linear_times(nn.Module):
    def __init__(self, times=100):
        self.times = times
        super(Linear_times, self).__init__()
    def forward(self, x):
        return x * self.times

class soft_layer(nn.Module):
    def __init__(self, t=5.6, bins=50, emb_size=100):
        super(soft_layer, self).__init__()
        self.t = t
        self.emb_size = emb_size
        self.layer1 = nn.Sequential(
            nn.Linear(1, bins, bias=False),
            nn.LeakyReLU()
        )
        self.lin = nn.Linear(bins, bins, bias=False)
        self.layer2 = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Linear(bins, emb_size, bias=False)
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.lin(y) + 0.1 * y
        y = y * self.t
        y = self.layer2(y)
        return y

class in_RepreLayer(nn.Module):
    def __init__(self, module, emb_size=20, total=100, t=5.6,
                 inv=2, bins=50, num_feature=10, device='cuda', hid_layers=0):
        super(in_RepreLayer, self).__init__()
        self.module = module
        self.emb_size = emb_size
        self.hid_layers = hid_layers

        self.inv = inv
        self.total = total
        self.bins = bins
        self.num_feature = num_feature
        if self.module == 'ewde' or self.module == 'efde':
            map_bin = torch.tensor(range(bins), device=device)
            map_bin = F.one_hot(map_bin, num_classes=total)
            padding = torch.zeros(map_bin.shape, device=device)
            map_bin = torch.cat([padding, map_bin, padding], dim=-1)
            temp = torch.zeros(map_bin.shape, device=device)
            ori = map_bin.clone()

            for i in range(1, self.inv + 1):
                temp = torch.roll(ori, i, -1)
                map_bin += temp
            for i in range(1, self.inv + 1):
                temp = torch.roll(ori, -i, -1)
                map_bin += temp
            self.map_bin = map_bin.float()
            self.tim_emb_lay = nn.ModuleList()

            for layer_i in range(hid_layers + 1):
                if layer_i == 0:
                    self.tim_emb_lay.append(nn.Linear(int(self.total * 3), self.emb_size, bias=False))
                else:
                    self.tim_emb_lay.append(nn.Linear(self.emb_size, self.emb_size, bias=False))

        if self.module == 'fe':
            self.tim_emb_lay = nn.Sequential(nn.Linear(1, 150, bias=False),
                                             # nn.ReLU(),
                                             nn.Linear(150, 50, bias=False),
                                             # nn.ReLU(),
                                             nn.Linear(50, self.emb_size, bias=False))
        if self.module == 'direct':
            print()

        if self.module == 'ad':
            self.tim_emb_lay = soft_layer(t=t, bins=bins, emb_size=emb_size)

        self.module_lst = nn.ModuleList()
        for i in range(self.num_feature):
            self.module_lst.append(self.tim_emb_lay)

    def forward(self, x):
        out_emb_list = []
        for i in range(self.num_feature):
            # x_i = x[:, :, i: i+1]
            x_i = x[:, :, i]
            if self.module == 'ewde' or self.module == 'efde':
                x_i = x_i.long()
                xf_i = F.one_hot(x_i, num_classes=self.bins).float()
                x_i = torch.matmul(xf_i, self.map_bin)
                for layer_j in range(self.hid_layers + 1):
                    x_i = self.module_lst[i][layer_j](x_i)
                tim_emb = x_i

            if self.module == 'fe':
                tim_emb = self.module_lst[i](x_i.unsqueeze(-1))

            if self.module == 'direct':
                tim_emb = self.module_lst[i](x_i.unsqueeze(-1))

            if self.module == 'ad':
                tim_emb = self.module_lst[i](x_i.unsqueeze(-1))
            out_emb_list.append(tim_emb)

        out_emb = torch.cat(out_emb_list, dim=-1)
        return out_emb