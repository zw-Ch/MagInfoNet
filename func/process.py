import os
import numpy as np
import torch
import os.path as osp
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


class Chunk(Dataset):
    def __init__(self, num, train, num_train, idx, root, name):
        super(Chunk, self).__init__()
        self.num = num
        self.root = root
        self.save_address = osp.join(root, str(num))
        self.name = name
        self.df, self.dtfl = self.get_merge()
        self.data, self.sm, self.pl, self.sl, self.index = self.get_sample()
        self.df = self.df.iloc[self.index, :]
        self.num_train = num_train
        self.length = self.data.shape[2]
        self.train = train
        self.idx = idx
        self.get_train_or_test()

    def get_train_or_test(self):
        self.data = self.data[self.idx, :, :]
        self.sm = self.sm[self.idx]
        self.pl = self.pl[self.idx]
        self.sl = self.sl[self.idx]
        self.index = self.index[self.idx]
        self.df = self.df.iloc[self.idx, :]
        return None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data, sm, pl, sl, index = self.data[idx, :, :], self.sm[idx], self.pl[idx], self.sl[idx], self.index[idx]
        return data, sm, pl, sl, index

    def get_merge(self):
        csv_address = osp.join(self.root, self.name + ".csv")
        file_address = osp.join(self.root, self.name + ".hdf5")
        df = pd.read_csv(csv_address)
        dtfl = h5py.File(file_address, 'r')
        return df, dtfl

    def get_sample(self):
        if not osp.exists(self.save_address):
            os.makedirs(self.save_address)
        data_address = osp.join(self.save_address, "data.pt")
        sm_address = osp.join(self.save_address, "sm.pt")
        p_arrival_address = osp.join(self.save_address, "p_arrival.pt")
        s_arrival_address = osp.join(self.save_address, "s_arrival.pt")
        index_address = osp.join(self.save_address, "index.pt")
        if (osp.exists(data_address) & osp.exists(sm_address) & osp.exists(p_arrival_address) &
                osp.exists(s_arrival_address) & (osp.exists(index_address))):
            data = torch.load(data_address)
            sm = torch.load(sm_address)
            p_arrival = torch.load(p_arrival_address)
            s_arrival = torch.load(s_arrival_address)
            index = torch.load(index_address)
        else:
            trace_name = self.df.loc[:, "trace_name"].values.reshape(-1)
            source_magnitude = self.df.loc[:, "source_magnitude"].values.reshape(-1)
            index = np.random.choice(trace_name.shape[0], self.num, replace=False).tolist()

            sm, p_arrival, s_arrival = [], [], []
            ev_list = self.df['trace_name'].to_list()
            data = np.zeros(shape=(self.num, 3, 6000))
            for c, i in enumerate(index):
                ev_one = ev_list[i]
                dataset_one = self.dtfl.get('data/' + str(ev_one))
                data_one = np.array(dataset_one)
                data_one = np.expand_dims(data_one.T, axis=0)
                data[c, :, :] = data_one
                sm_one = source_magnitude[i]

                p_arrival_one = dataset_one.attrs['p_arrival_sample']
                s_arrival_one = dataset_one.attrs['s_arrival_sample']

                sm.append(sm_one), p_arrival.append(p_arrival_one), s_arrival.append(s_arrival_one)

            data = torch.from_numpy(data).float()
            index = torch.FloatTensor(index).int()
            sm = torch.FloatTensor(sm).float()
            p_arrival = torch.FloatTensor(p_arrival)
            s_arrival = torch.FloatTensor(s_arrival)

            torch.save(data, data_address)
            torch.save(index, index_address)
            torch.save(sm, sm_address)
            torch.save(p_arrival, p_arrival_address)
            torch.save(s_arrival, s_arrival_address)
        if self.num != data.shape[0]:
            raise ValueError("data.shape[0] is not equal to num. Please delete the file saved before and run again!")
        return data, sm, p_arrival, s_arrival, index


def get_train_or_test_idx(num, num_train):
    idx_all = np.arange(num)
    idx_train = np.random.choice(num, num_train, replace=False)
    idx_test = list(set(idx_all) - set(idx_train))
    return idx_train, idx_test


def df_info(df, r_w=1, r_at=1000):
    p_w, s_w = df.p_weight.values / r_w, df.s_weight.values / r_w
    p_w, s_w = tran_nan(p_w, 0), tran_nan(s_w, 0)     # 将nan值替换为1
    p_at, s_at = df.p_arrival_sample.values / r_at, df.s_arrival_sample.values / r_at
    ps_w = np.hstack([p_w.reshape(-1, 1), s_w.reshape(-1, 1)])
    ps_at = np.hstack([p_at.reshape(-1, 1), s_at.reshape(-1, 1)])
    ps_w = torch.from_numpy(ps_w).float()
    ps_at = torch.from_numpy(ps_at).float()
    return ps_w, ps_at


# Convert "nan" in the array x , to "value"
def tran_nan(x, value):
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    elif x.ndim == 2:
        pass
    else:
        raise ValueError("x must be 1d or 2d, but got {}d!".format(x.ndim))
    num_col = x.shape[1]
    x_new = np.zeros(shape=x.shape)
    for i in range(num_col):
        x_one = x[:, i]
        nan_idx = np.isnan(x_one)
        nan_idx = np.argwhere(nan_idx == True).reshape(-1)
        if nan_idx.shape[0] == 0:
            pass
        else:
            x_one[nan_idx] = value
        x_new[:, i] = x_one
    return x_new


def remain_sm_type(data, df, label, style):
    smt = df.source_magnitude_type.values.reshape(-1)
    idx = np.argwhere(smt == style).reshape(-1)
    data = data[idx, :, :]
    label = label[idx]
    df = df.iloc[idx, :]
    return data, label, df


class SelfData(Dataset):
    def __init__(self, data, label, *args):
        super(SelfData, self).__init__()
        self.data = data
        self.label = label
        self.args = args
        self.data_else = self.get_data_else()

    def get_data_else(self):
        num = len(self.args)
        if num != 0:
            data_else = []
            for i in range(num):
                data_else_one = self.args[i]
                data_else.append(data_else_one)
        else:
            data_else = None
        return data_else

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.data.dim() == 3:
            data_one = self.data[item, :, :]
        elif self.data.dim() == 4:
            data_one = self.data[item, :, :, :]
        else:
            raise ValueError("data.dim() must be 3 or 4, but got {}".format(self.data.dim()))
        if self.label.dim() == 1:
            label_one = self.label[item]
        elif self.label.dim() == 2:
            label_one = self.label[item, :]
        else:
            raise ValueError("label.dim() must be 1 or 2, but got {}".format(self.label.dim()))
        return_all = [data_one, label_one]
        data_else_one = []
        if self.data_else is not None:
            num = len(self.data_else)
            for i in range(num):
                x = self.data_else[i]
                if x.dim() == 2:
                    x_one = x[item, :]
                elif x.dim() == 1:
                    x_one = x[item]
                else:
                    raise ValueError("data_else dim() must be 1 or 2, but got {}".format(x_one.dim()))
                data_else_one.append(x_one)
        return_all = return_all + data_else_one
        return_all.append(item)
        return_all = tuple(return_all)
        return return_all


def prep(train, test, prep_style, be_torch=False):
    if prep_style == "sta":
        preprocessor = StandardScaler()
    elif prep_style == "min":
        preprocessor = MinMaxScaler()
    else:
        raise TypeError("Unknown Type of prep_style!")
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)
    preprocessor.fit(train)
    train_prep = preprocessor.transform(train)
    test_prep = preprocessor.transform(test)
    if be_torch:
        train_prep = torch.from_numpy(train_prep).float()
        test_prep = torch.from_numpy(test_prep).float()
    return train_prep, test_prep, preprocessor

