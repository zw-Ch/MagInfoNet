import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from sklearn.metrics import r2_score


class MagInfoNet(nn.Module):
    def __init__(self, edge_weight, num_nodes, out_dim, gnn_style, num_layers, hid_dim):
        super(MagInfoNet, self).__init__()
        self.linear_at = nn.Sequential(nn.Linear(2, 1000), nn.Linear(1000, 6000))
        self.linear_t = nn.Sequential(nn.Linear(1, 1000), nn.Linear(1000, 6000))
        self.edge_weight = nn.Parameter(edge_weight)
        self.gnn_style = gnn_style
        self.hid_dim = hid_dim
        self.cnn1 = nn.Conv2d(1, hid_dim, kernel_size=(1, 3), padding=(0, 1))
        self.cnn2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 3), padding=(0, 1))
        self.cnn3 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 3), padding=(0, 1))
        self.cnn4 = nn.Conv2d(hid_dim, 1, kernel_size=(1, 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1))
        self.bn1 = nn.BatchNorm2d(hid_dim)
        self.bn2 = nn.BatchNorm2d(hid_dim)
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.linear2 = nn.Linear(3, 300)
        self.cnn5 = nn.Conv2d(1, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn6 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn7 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn8 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn9 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn10 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn11 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn12 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn13 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn14 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))
        self.cnn15 = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 5), padding=(0, 2))

        self.pool2 = nn.MaxPool2d(kernel_size=(3, 10))
        self.gnn_1, self.gnn_2 = gnn_layers(hid_dim, hid_dim, 1, gnn_style)
        self.lstm1 = nn.LSTM(hid_dim, 1, num_layers=num_layers, batch_first=True)
        self.gru1 = nn.GRU(hid_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.gru2 = nn.GRU(hid_dim, 1, num_layers=num_layers, batch_first=True)
        self.last = nn.Linear(num_nodes, out_dim)

    def forward(self, x, edge_index, ps_at, p_t):
        h_at = self.linear_at(ps_at).unsqueeze(1)
        h_t = self.linear_t(p_t).unsqueeze(1)

        h_x = self.cnn1(x.unsqueeze(1))
        h_x_0 = h_x
        h_x = self.pre(self.bn1(h_x))
        h_x = self.cnn2(h_x)
        h_x = self.pre(self.bn1(h_x))
        h_x = self.cnn3(h_x)
        h_x = self.pre(self.bn1(h_x))
        h_x = h_x + h_x_0
        h_x = self.cnn4(self.bn1(h_x))
        h_x = self.pool1(h_x)
        h_x = h_x.squeeze(1)

        out = torch.cat((h_x, h_at, h_t), dim=1)
        out = self.cnn5(out.unsqueeze(1))
        out_0 = out
        out = self.pre(self.bn2(out))
        out = self.cnn6(out)
        out = self.pre(self.bn2(out))
        out = self.cnn7(out)
        out = out + out_0

        out_1 = out
        out = self.cnn8(self.pre(self.bn2(out)))
        out = self.cnn9(self.pre(self.bn2(out)))
        out = out + out_1

        out_2 = out
        out = self.cnn10(self.pre(self.bn2(out)))
        out = self.cnn11(self.pre(self.bn2(out)))
        out = out + out_2

        out_3 = out
        out = self.cnn12(self.pre(self.bn2(out)))
        out = self.cnn13(self.pre(self.bn2(out)))
        out = out + out_3

        # out_4 = out
        # out = self.cnn14(self.pre(self.bn2(out)))
        # out = self.cnn15(self.pre(self.bn2(out)))
        # out = out + out_4

        out = self.pool2(out)

        out = out.view(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
        put = self.gnn_batch(out, edge_index)
        # put, (_, _) = self.lstm1(self.pre(put))
        # put, (_) = self.gru1(put)
        # put, (_) = self.gru2(put)
        put = self.last(put.view(put.shape[0], -1))
        return put

    def gnn_batch(self, x, edge_index):         
        batch_size = x.shape[0]             
        h_all = None
        for i in range(batch_size):             
            x_one = x[i, :, :]
            h = self.gnn_1(x_one, edge_index)
            h = self.gnn_2(self.pre(h), edge_index)
            h = h.unsqueeze(0)
            if h_all is None:
                h_all = h
            else:
                h_all = torch.cat((h_all, h), dim=0)
        return h_all


def get_adm(adm_style, n, k):
    if adm_style == "ts_un":
        adm_1 = time_series_graph_k(n, k)
        adm_2 = adm_1.T
        adm = adm_1 + adm_2 / 2
    else:
        raise TypeError("Unknown adm_style, got {}".format(adm_style))
    return adm


def time_series_graph_k(n, k):
    adm = np.zeros(shape=(n, n))
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    else:
        for i in range(n):
            if i < (n - k):
                for k_one in range(1, k + 1):
                    adm[i, i + k_one] = 1.
            else:
                for k_one in range(1, k + 1):
                    if (k_one + i) >= n:
                        pass
                    else:
                        adm[i, i + k_one] = 1.
    return adm


def tran_adm_to_edge_index(adm):
    if isinstance(adm, np.ndarray):
        u, v = np.nonzero(adm)
        num_edges = u.shape[0]
        edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])
        edge_weight = np.zeros(shape=u.shape)
        for i in range(num_edges):
            edge_weight_one = adm[u[i], v[i]]
            edge_weight[i] = edge_weight_one
        edge_index = torch.from_numpy(edge_index.astype(np.int64))
        edge_weight = torch.from_numpy(edge_weight.astype(np.float32))
        return edge_index, edge_weight
    elif torch.is_tensor(adm):
        u, v = torch.nonzero(adm, as_tuple=True)
        num_edges = u.shape[0]
        edge_index = torch.cat((u.view(1, -1), v.view(1, -1)))
        edge_weight = torch.zeros(size=u.shape)
        for i in range(num_edges):
            edge_weight[i] = adm[u[i], v[i]]
        return edge_index, edge_weight


def get_r2_score(y1, y2, axis):
    if (type(y1) is np.ndarray) & (type(y2) is np.ndarray):      # numpy数组类型
        pass
    elif (torch.is_tensor(y1)) & (torch.is_tensor(y2)):          # pytorch张量类型
        y1 = y1.detach().cpu().numpy()
        y2 = y2.detach().cpu().numpy()
    else:
        raise TypeError("type of y1 and y must be the same, but got {} and {}".format(type(y1), type(y2)))
    if y1.shape != y2.shape:
        raise ValueError("shape of y1 and y2 must be the same, but got {} and {}".format(y1.shape, y2.shape))
    if y1.ndim == 1:
        y1 = np.expand_dims(y1, axis=1)
        y2 = np.expand_dims(y2, axis=1)
    elif y1.ndim == 2:
        pass
    else:
        raise ValueError("y1 and y2 must be 1d or 2d, but got {}d".format(y1.ndim))
    if axis == 0:
        num_col = y1.shape[0]
    elif axis == 1:
        num_col = y1.shape[1]
    else:
        raise TypeError("axis must be equal as 0 or 1, but got {}".format(axis))
    r2_all = 0
    for i in range(num_col):
        if axis == 0:
            y1_one = y1[i, :]
            y2_one = y2[i, :]
        elif axis == 1:
            y1_one = y1[:, i]
            y2_one = y2[:, i]
        else:
            raise TypeError("axis must be equal as 0 or 1, but got {}".format(axis))
        r2_one = r2_score(y1_one, y2_one)
        r2_all = r2_all + r2_one
    r2 = r2_all / num_col
    return r2


def gnn_layers(in_dim, hid_dim, out_dim, gnn_style):
    if gnn_style == "GCN":
        gnn_1 = gnn.GCNConv(in_dim, hid_dim)
        gnn_2 = gnn.GCNConv(hid_dim, out_dim)
    elif gnn_style == "Cheb":
        gnn_1 = gnn.ChebConv(in_dim, hid_dim, K=2)
        gnn_2 = gnn.ChebConv(hid_dim, out_dim, K=2)
    elif gnn_style == "GIN":
        gnn_1 = gnn.GraphConv(in_dim, hid_dim)
        gnn_2 = gnn.GraphConv(hid_dim, out_dim)
    elif gnn_style == "GraphSage":
        gnn_1 = gnn.SAGEConv(in_dim, hid_dim)
        gnn_2 = gnn.SAGEConv(hid_dim, out_dim)
    elif gnn_style == "TAG":
        gnn_1 = gnn.TAGConv(in_dim, hid_dim)
        gnn_2 = gnn.TAGConv(hid_dim, out_dim)
    elif gnn_style == "UniMP":
        gnn_1 = gnn.TransformerConv(in_dim, hid_dim)
        gnn_2 = gnn.TransformerConv(hid_dim, out_dim)
    elif gnn_style == "GAT":
        gnn_1 = gnn.GATConv(in_dim, hid_dim)
        gnn_2 = gnn.GATConv(hid_dim, out_dim)
    elif gnn_style == "ARMA":
        gnn_1 = gnn.ARMAConv(in_dim, hid_dim)
        gnn_2 = gnn.ARMAConv(hid_dim, out_dim)
    else:
        raise TypeError("Unknown Type of gnn_style!")
    return gnn_1, gnn_2
