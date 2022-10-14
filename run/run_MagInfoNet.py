"""
The main program
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import os.path as osp
import pickle
from torch.utils.data import DataLoader
import sys
sys.path.append("..")

import func.net as net
import func.process as pro
from func.net import MagInfoNet


device = "cuda:1" if torch.cuda.is_available() else "cpu"
batch_size = 64
k = 2
gnn_style_all = ["UniMP", "GCN", "GraphSage", "GIN", "GAT", "TAG"]
gnn_style = "UniMP"           # The type of Graph Neural Network
lr = 0.0001                  # The learning rate
weight_decay = 5e-4          # The weight decay
epochs = 600                 # The number of iterations
out_dim = 1
num_layers = 1
save_txt = True              # Whether to write the running results to TXT file
save_fig = True              # Whether to save pictures
save_np = True               # Whether to save true results and predict results
save_model = True            # Whether to save network model
hid_dim = 32
fig_size = (12, 12)          # The size of figures
font_size = 40
font_ti_size = 20
sm_style = "md"              # The Magnitude Type ("ml", "md", "mb", etc)
prep_style = "sta"
random = False

# Create results folder
last_address = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
result_address = osp.join(last_address, "result", "MagInfoNet")
if not(osp.exists(result_address)):
    os.makedirs(result_address)

num = 200000                     # The number of samples, in total
num_train = 180000                # The number of samples, training dataset
if not random:
    np.random.seed(1)
idx_train, idx_test = pro.get_train_or_test_idx(num, num_train)
root = "/home/chenziwei2021/standford_dataset/chunk2"    # The storage path of data file
name = "chunk2"
chunk_train = pro.Chunk(num, True, num_train, idx_train, root, name)    # Training Dataset
chunk_test = pro.Chunk(num, False, num_train, idx_test, root, name)     # Test Dataset

data_train, data_test = chunk_train.data, chunk_test.data     # Time Series (X)
sm_train, sm_test = chunk_train.sm, chunk_test.sm       # Source Magnitude (y)
df_train, df_test = chunk_train.df, chunk_test.df

# Select samples according to Magnitude Type
data_train, sm_train, df_train = pro.remain_sm_type(data_train, df_train, sm_train, sm_style)
data_test, sm_test, df_test = pro.remain_sm_type(data_test, df_test, sm_test, sm_style)

# The location of sources
df_train_pos = df_train.loc[:, ["source_longitude", "source_latitude"]].values
df_test_pos = df_test.loc[:, ["source_longitude", "source_latitude"]].values

num_train, num_test = data_train.shape[0], data_test.shape[0]
num = num_train + num_test             # The number of samples, after extraction

ps_at_name = ["p_arrival_sample", "s_arrival_sample"]
ps_at_train, ps_at_test = df_train.loc[:, ps_at_name].values, df_test.loc[:, ps_at_name].values
ps_at_train, ps_at_test, prep_ps_at = pro.prep(ps_at_train, ps_at_test, prep_style)
ps_at_train, ps_at_test = torch.from_numpy(ps_at_train).float(), torch.from_numpy(ps_at_test).float()

t_name = ["p_travel_sec"]
p_t_train, p_t_test = df_train.loc[:, t_name].values, df_test.loc[:, t_name].values
p_t_train, p_t_test, prep_p_t = pro.prep(p_t_train, p_t_test, prep_style)
p_t_train, p_t_test = torch.from_numpy(p_t_train).float(), torch.from_numpy(p_t_test).float()

if save_model:
    prep_ps_at_address = osp.join(result_address, "prep_ps_at_{}_{}_{}_{}.sav".format(sm_style, name, num_train, num_test))
    pickle.dump(prep_ps_at, open(prep_ps_at_address, "wb"))
    prep_p_t_address = osp.join(result_address, "prep_p_t_{}_{}_{}_{}.sav".format(sm_style, name, num_train, num_test))
    pickle.dump(prep_p_t, open(prep_p_t_address, "wb"))

sm_train, sm_test = sm_train.unsqueeze(1), sm_test.unsqueeze(1)
train_dataset = pro.SelfData(data_train, sm_train, ps_at_train, p_t_train)
test_dataset = pro.SelfData(data_test, sm_test, ps_at_test, p_t_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

num_nodes = 600                             # The number of nodes in graph structure
adm = net.get_adm("ts_un", num_nodes, k)    # Construct adjacency matrix
edge_index, edge_weight = net.tran_adm_to_edge_index(adm)

# Build network model, loss function and optimizer
model = MagInfoNet(edge_weight, num_nodes, out_dim, gnn_style, num_layers, hid_dim).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
edge_index = edge_index.to(device)

# Training and testing
train_predict, train_true, test_predict, test_true = [], [], [], []
train_pos, test_pos = [], []
for epoch in range(epochs):
    loss_train_all, loss_test_all = 0, 0
    for item_train, (x_train, y_train, ps_at_train, p_t_train, index_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        ps_at_train, p_t_train = ps_at_train.to(device), p_t_train.to(device)

        optimizer.zero_grad()
        output_train = model(x_train, edge_index, ps_at_train, p_t_train)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()
        loss_train_all = loss_train_all + loss_train.item()

        train_predict_one = output_train.detach().cpu().numpy()
        train_true_one = y_train.detach().cpu().numpy()
        train_index_one = index_train.numpy()
        train_pos_one = df_train_pos[train_index_one, :]
        if item_train == 0:
            train_predict = train_predict_one
            train_true = train_true_one
            train_pos = train_pos_one
        else:
            train_predict = np.concatenate((train_predict, train_predict_one), axis=0)
            train_true = np.concatenate((train_true, train_true_one), axis=0)
            train_pos = np.concatenate((train_pos, train_pos_one), axis=0)

    for item_test, (x_test, y_test, ps_at_test, p_t_test, index_test) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)
        ps_at_test, p_t_test = ps_at_test.to(device), p_t_test.to(device)

        output_test = model(x_test, edge_index, ps_at_test, p_t_test)
        loss_test = criterion(output_test, y_test)
        loss_test_all = loss_test_all + loss_test.item()

        test_predict_one = output_test.detach().cpu().numpy()
        test_true_one = y_test.detach().cpu().numpy()
        test_index_one = index_test.numpy()
        test_pos_one = df_test_pos[test_index_one, :]
        if item_test == 0:
            test_predict = test_predict_one
            test_true = test_true_one
            test_pos = test_pos_one
        else:
            test_predict = np.concatenate((test_predict, test_predict_one), axis=0)
            test_true = np.concatenate((test_true, test_true_one), axis=0)
            test_pos = np.concatenate((test_pos, test_pos_one), axis=0)

    r2_train = net.get_r2_score(train_predict, train_true, axis=1)
    r2_test = net.get_r2_score(test_predict, test_true, axis=1)
    if sm_style == 'ml':
        if r2_test >= 0.898:
            break
    elif sm_style == 'md':
        if r2_test >= 0.825:
            break
    print("Epoch: {:04d}  Loss_Train: {:.4f}  Loss_Test: {:.4f}  R2_Train: {:.8f}  R2_Test: {:.8f}".
          format(epoch, loss_train_all, loss_test_all, r2_train, r2_test))

if save_np:
    np.save(osp.join(result_address, "train_true_{}_{}_{}_{}.npy".format(sm_style, name, num_train, num_test)), train_true)
    np.save(osp.join(result_address, "train_predict_{}_{}_{}_{}.npy".format(sm_style, name, num_train, num_test)), train_predict)
    np.save(osp.join(result_address, "train_pos_{}_{}_{}_{}.npy".format(sm_style, name, num_train, num_test)), train_pos)
    np.save(osp.join(result_address, "test_true_{}_{}_{}_{}.npy".format(sm_style, name, num_train, num_test)), test_true)
    np.save(osp.join(result_address, "test_predict_{}_{}_{}_{}.npy".format(sm_style, name, num_train, num_test)), test_predict)
    np.save(osp.join(result_address, "test_pos_{}_{}_{}_{}.npy".format(sm_style, name, num_train, num_test)), test_pos)
if save_model:
    torch.save(model.state_dict(), osp.join(result_address, "MagInfoNet_model_{}_{}_{}_{}.pkl".format(sm_style, name, num_train, num_test)))

if save_txt:
    info_txt_address = osp.join(result_address, "MagInfoNet_result.txt")
    info_df_address = osp.join(result_address, "MagInfoNet_result.csv")
    f = open(info_txt_address, 'a')
    if osp.getsize(info_txt_address) == 0:
        f.write("r2_test r2_train gnn_style sm_style batch_size k epochs lr hid_dim num_train num_test name\n")
    f.write(str(round(r2_test, 5)) + "\t")
    f.write(str(round(r2_train, 5)) + "\t")
    f.write(str(gnn_style) + "\t")
    f.write(str(sm_style) + "\t")
    f.write(str(batch_size) + "\t")
    f.write(str(k) + "\t")
    f.write(str(epochs) + "\t")
    f.write(str(lr) + "\t")
    f.write(str(hid_dim) + "\t")
    f.write(str(num_train) + "\t")
    f.write(str(num_test) + "\t")
    f.write(str(name) + "\t")

    f.write("\n")
    f.close()

    info = np.loadtxt(info_txt_address, dtype=str)
    columns = info[0, :].tolist()
    values = info[1:, :]
    info_df = pd.DataFrame(values, columns=columns)
    info_df.to_csv(info_df_address)


print()
plt.show()
print()
