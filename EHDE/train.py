# @date:    2021/11/21
# @author:  Zhiyuan Lu
# @email:   luzy@bupt.edu.cn

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import numpy as np
from utils import process
from models import GNN
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import accuracy
from torchmetrics.functional import auroc
from torch.utils.data import DataLoader
# from torchsummary import summary

# ============================
# GPU configuration
# ============================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_printoptions(threshold=np.inf)

# torch.set_printoptions(threshold=np.inf)

# ============================
# hyper-parameters
# ============================
# PATH = './data/acm_mini/ACM_miniPvsA.txt'
# PATH = './data/acm_mini/ACM_miniPvsC.txt'
# PATH = './data/imdb/MD.txt'
# PATH = './data/imdb/MA.txt'
# PATH = './data/lastfm_mini/UA_mini.txt'
# PATH = './data/lastfm_mini/AT_mini.txt'
# PATH = './data/freebase/new_ma.txt'
PATH = './data/freebase/new_md.txt'
USE_DE = True
NUM_NEIGHBOR = 5
EPOCH = 10
BATCH_SIZE = 64
K_HOP = 2  # Consider K_HOP ego graph when computing HDE.
EMB_DIM = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
MAX_DIST = K_HOP + 1
TASK = "link_prediction"
if TASK == "link_prediction":
    node_set_size = 2
elif TASK == "meta_prediction":
    node_set_size = 3
type2idx = {
    'M': 0,
    'D': 1,
}
node_type = len(type2idx)
random.seed(0)

# specify the feature dimension
if USE_DE:
    # num_fea = (K_HOP + 2) * 4 + node_type  # TODO 怎么算的？
    num_fea = (node_type * (MAX_DIST + 1)) * node_set_size + node_type
    # num_fea = 32 + node_type
else:
    num_fea = node_type

model = GNN.LP(input_dim=num_fea, output_dim=EMB_DIM/2, num_neighbor=NUM_NEIGHBOR).to(DEVICE)
criterion = nn.BCELoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# initialization
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')

# y_pred  = torch.FloatTensor([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1])
# y_label = torch.IntTensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# acc = accuracy(y_pred, y_label)
# auc = auroc(y_pred, y_label, pos_label=1)
# print("acc:{} auc:{}".format(acc, auc))




if __name__ == '__main__':
    g_train, g_val, g_test = process.load_data(PATH, test_ratio=0.1)
    train_acc_list = []
    val_acc_list = []
    t1 = time.time()
    data_A_train, data_B_train, data_y_train = process.batch_data(g_train,
                                                                  K_HOP,
                                                                  USE_DE,
                                                                  node_type,
                                                                  type2idx,
                                                                  BATCH_SIZE,
                                                                  num_fea,
                                                                  NUM_NEIGHBOR,
                                                                  MAX_DIST)
    t2 = time.time()
    print('time:', t2 - t1)
    data_A_train = data_A_train.reshape(-1, data_A_train.shape[2])
    data_B_train = data_B_train.reshape(-1, data_B_train.shape[2])
    data_y_train = data_y_train.reshape(-1)
    # np.save('./cache_ma/data_A_train', data_A_train)
    # np.save('./cache_ma/data_B_train', data_B_train)
    # np.save('./cache_ma/data_y_train', data_y_train)
    #
    data_A_val, data_B_val, data_y_val = process.batch_data(g_val,
                                                            K_HOP,
                                                            USE_DE,
                                                            node_type,
                                                            type2idx,
                                                            BATCH_SIZE,
                                                            num_fea,
                                                            NUM_NEIGHBOR,
                                                            MAX_DIST)
    data_A_val = data_A_val.reshape(-1, data_A_val.shape[2])
    data_B_val = data_B_val.reshape(-1, data_B_val.shape[2])
    data_y_val = data_y_val.reshape(-1)
    # np.save('./cache_ma/data_A_val', data_A_val)
    # np.save('./cache_ma/data_B_val', data_B_val)
    # np.save('./cache_ma/data_y_val', data_y_val)
    #
    data_A_test, data_B_test, data_y_test = process.batch_data(g_test,
                                                            K_HOP,
                                                            USE_DE,
                                                            node_type,
                                                            type2idx,
                                                            BATCH_SIZE,
                                                            num_fea,
                                                            NUM_NEIGHBOR,
                                                            MAX_DIST)
    data_A_test = data_A_test.reshape(-1, data_A_test.shape[2])
    data_B_test = data_B_test.reshape(-1, data_B_test.shape[2])
    data_y_test = data_y_test.reshape(-1)
    # np.save('./cache_ma/data_A_test', data_A_test)
    # np.save('./cache_ma/data_B_test', data_B_test)
    # np.save('./cache_ma/data_y_test', data_y_test)

    # data_A_train = np.load('./cache_ma/data_A_train.npy', allow_pickle=True)
    # data_B_train = np.load('./cache_ma/data_B_train.npy', allow_pickle=True)
    # data_y_train = np.load('./cache_ma/data_y_train.npy', allow_pickle=True)
    # data_A_val = np.load('./cache_ma/data_A_val.npy', allow_pickle=True)
    # data_B_val = np.load('./cache_ma/data_B_val.npy', allow_pickle=True)
    # data_y_val = np.load('./cache_ma/data_y_val.npy', allow_pickle=True)
    # data_A_test = np.load('./cache_ma/data_A_test.npy', allow_pickle=True)
    # data_B_test = np.load('./cache_ma/data_B_test.npy', allow_pickle=True)
    # data_y_test = np.load('./cache_ma/data_y_test.npy', allow_pickle=True)

    train_dataset = process.HDEDataset(data_A_train, data_B_train, data_y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for ep in range(EPOCH):
        # for (train_batch_A_fea, train_batch_B_fea, train_batch_y) in zip(data_A_train, data_B_train, data_y_train):
        for it, [train_batch_A_fea, train_batch_B_fea, train_batch_y] in enumerate(train_loader):
            # train
            model.train()
            # train_batch_A_fea = torch.tensor(train_batch_A_fea).to(torch.float32).to(DEVICE)
            # train_batch_B_fea = torch.tensor(train_batch_B_fea).to(torch.float32).to(DEVICE)
            # train_batch_y = torch.FloatTensor(train_batch_y).to(DEVICE)
            train_batch_A_fea = train_batch_A_fea.to(torch.float32).to(DEVICE)
            train_batch_B_fea = train_batch_B_fea.to(torch.float32).to(DEVICE)
            train_batch_y = train_batch_y.to(torch.float32).to(DEVICE)
            logits = model(train_batch_A_fea, train_batch_B_fea)
            loss = criterion(logits.squeeze(), train_batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            # pred = logits.argmax(dim=1)
            # train_acc = accuracy_score(train_batch_y.cpu().numpy(), pred.cpu().numpy())
            # train_auc = roc_auc_score(train_batch_y.cpu().numpy(), pred.cpu().numpy())
            train_acc = accuracy(logits.squeeze(), train_batch_y.int().squeeze()).item()
            train_auc = auroc(logits.squeeze(), train_batch_y.int().squeeze(), pos_label=1).item()
            train_acc_list.append(train_acc)

            # val
            model.eval()
            val_batch_A_fea = data_A_val
            val_batch_B_fea = data_B_val
            val_batch_y = data_y_val
            val_batch_A_fea = torch.tensor(val_batch_A_fea).to(torch.float32).to(DEVICE)
            val_batch_B_fea = torch.tensor(val_batch_B_fea).to(torch.float32).to(DEVICE)
            val_batch_y = torch.FloatTensor(val_batch_y).to(DEVICE)
            logits = model(val_batch_A_fea, val_batch_B_fea)
            # pred = logits.argmax(dim=1)
            # val_acc = accuracy_score(val_batch_y.cpu().numpy(), pred.cpu().numpy())
            # val_auc = roc_auc_score(val_batch_y.cpu().numpy(), pred.cpu().numpy())
            val_acc = accuracy(logits.squeeze(), val_batch_y.int().squeeze()).item()
            val_auc = auroc(logits.squeeze(), val_batch_y.int().squeeze(), pos_label=1).item()
            val_acc_list.append(val_acc)

            print('ep:{}  train_acc:{}  train_auc:{}  train_loss{} val_acc:{}  val_auc:{}'
                  .format(ep, train_acc, train_auc, loss.item(), val_acc, val_auc))

    plt.plot(val_acc_list)
    plt.show()
    exit()
