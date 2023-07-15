# @date:    2021/11/21
# @author:  Zhiyuan Lu
# @email:   luzy@bupt.edu.cn

import torch
import random
from utils import HDE
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch.utils.data import Dataset

mini_batch = []
random.seed(0)
np.random.seed(0)


# def load_data(path, test_ratio=0.2):
#     """
#     load data for link prediction task using DGL library
#     :param path: data set path
#     :param test_ratio: ratio of test set size
#     :return: split graph
#     """
#     with open(path, 'r') as f:
#         edge_list = []
#         for line in f.readlines():
#             line = line.strip().split(' ')
#             src_indx = int(line[0][1:])
#             targ_indx = int(line[1][1:])
#             edge_list.append([src_indx, targ_indx])
#
#     # train/val/test split:
#     random.shuffle(edge_list)
#     train_len = int(len(edge_list) * (1 - 2 * test_ratio))
#     val_len = int(len(edge_list) * test_ratio)
#     test_len = len(edge_list) - train_len - val_len
#     train_edge_list = np.array(edge_list[0:train_len])
#     val_edge_list = np.array(edge_list[train_len: train_len + val_len])
#     test_edge_list = np.array(edge_list[train_len + val_len:])
#
#     # generate train/val/test graph:
#     train_graph_dict = {('M', 'edge', 'A'): (torch.tensor(train_edge_list[:, 0]),
#                                              torch.tensor(train_edge_list[:, 1]))}
#     val_graph_dict = {('M', 'edge', 'A'): (torch.tensor(val_edge_list[:, 0]),
#                                              torch.tensor(val_edge_list[:, 1]))}
#     test_graph_dict = {('M', 'edge', 'A'): (torch.tensor(test_edge_list[:, 0]),
#                                            torch.tensor(test_edge_list[:, 1]))}
#     g_train = dgl.heterograph(train_graph_dict)
#     g_val = dgl.heterograph(val_graph_dict)
#     g_test = dgl.heterograph(test_graph_dict)
#
#     return g_train, g_val, g_test

def load_data(path, test_ratio=0.2):
    """
    Load data for link predictio task using networkx graph data structure
    :param path:
    :param test_ratio:
    :return:
    """
    # sp， train size
    # val size = test size
    G = nx.Graph()
    sp = 1 - test_ratio * 2

    edge_list = []

    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            G.add_edge(line[0], line[1])
            edge_list.append(line)

    # TODO test only
    # cut_num = 5000
    # edge_list = [edge_list[i] for i in range(cut_num)]

    num_edge = len(edge_list)
    sp1 = int(num_edge * sp)
    sp2 = int(num_edge * test_ratio)
    print(num_edge, sp, sp2)
    # random.shuffle(edge_list) # 为了的DE保持一致
    G_train = nx.Graph()
    G_val = nx.Graph()
    G_test = nx.Graph()

    G_train.add_edges_from(edge_list[:sp1])
    G_val.add_edges_from(edge_list[sp1:sp1 + sp2])
    G_test.add_edges_from(edge_list[sp1 + sp2:])
    print(
        f"all edge: {len(G.edges)}, train edge: {len(G_train.edges)}, val edge: {len(G_val.edges)}, test edge: {len(G_test.edges)}")
    return G_train, G_val, G_test


# def batch_data(g,
#                k_hop,
#                use_de,
#                node_type,
#                type2idx,
#                batch_size,
#                num_fea,
#                num_neighbor):
#     """
#     Generate data and compute HDE in each iteration.
#     :param g: Input graph.
#     :param k_hop: K hop neighbor for egog graph sampling.
#     :param use_de: Use distance encoding or not.
#     :param node_type: Number of node type.
#     :param type2idx: Node type encoding dictionary
#     :param batch_size: Batch size.
#     :param num_fea: Dimension of feature vector
#     :param num_neighbor: Number of sampling neighbor.
#     :return:Data that can feed into the network
#     """
#     edge = list(g.edges)
#     nodes = list(g.nodes)
#     num_batch = int(len(edge) / batch_size)
#     random.shuffle(edge)
#     for idx in range(num_batch):
#         batch_edge = edge[idx * batch_size:(idx + 1) * batch_size]
#         batch_label = [1.0] * batch_size
#         batch_A_fea = []
#         batch_B_fea = []
#         batch_x = []
#         batch_y = []
#
#         for (bx, by) in zip(batch_edge, batch_label):
#             posA, posB = HDE.subgraph_sampling_with_DE_node_pair(g,
#                                                                  bx,
#                                                                  use_de,
#                                                                  node_type,
#                                                                  type2idx,
#                                                                  num_fea,
#                                                                  num_neighbor,
#                                                                  K_HOP=k_hop)
#             batch_A_fea.append(posA)
#             batch_B_fea.append(posB)
#             # batch_x.append(tmp_pos)
#             # tmpB = np.asarray(HDE.subgraph_sampling_with_DE_node_pair(G, bx[1]), dtype=np.float32)
#             # batch_B_fea.append(tmpB)
#             batch_y.append(np.asarray(by, dtype=np.float32))
#
#             # neg
#             # batch_A_fea.append(tmpA)
#             # TODO do not consider sampling pos as neg
#             neg_tmpB_id = random.choice(nodes)
#             # tmp_neg = np.asarray(HDE.subgraph_sampling_with_DE_node_pair(G, [bx[0], neg_tmpB_id]),
#             #                      dtype=np.float32)
#             # batch_x.append(tmp_neg)
#             negA, negB = HDE.subgraph_sampling_with_DE_node_pair(g,
#                                                                  [bx[0], neg_tmpB_id],
#                                                                  use_de,
#                                                                  node_type,
#                                                                  type2idx,
#                                                                  num_fea,
#                                                                  num_neighbor,
#                                                                  K_HOP=k_hop)
#             batch_A_fea.append(negA)
#             batch_B_fea.append(negB)
#             batch_y.append(np.asarray(0.0, dtype=np.float32))
#
#         yield np.asarray(np.squeeze(batch_A_fea)), np.asarray(np.squeeze(batch_B_fea)), np.asarray(
#             batch_y).reshape(batch_size * 2, 1)

def batch_data(g,
               k_hop,
               use_de,
               node_type,
               type2idx,
               batch_size,
               num_fea,
               num_neighbor,
               max_dist):
    """
    Generate all the data and compute HDE all at once.
    :param g: Input graph.
    :param k_hop: K hop neighbor for egog graph sampling.
    :param use_de: Use distance encoding or not.
    :param node_type: Number of node type.
    :param type2idx: Node type encoding dictionary
    :param batch_size: Batch size.
    :param num_fea: Dimension of feature vector
    :param num_neighbor: Number of sampling neighbor.
    :return:Data that can feed into the network
    """
    edge = list(g.edges)
    nodes = list(g.nodes)
    num_batch = int(len(edge) * 2 / batch_size)
    random.shuffle(edge)
    data = []
    edge = edge[0: num_batch * batch_size // 2]
    m_rand = np.random.rand(3, num_fea // 8)
    for bx in tqdm(edge):
        posA, posB = HDE.subgraph_sampling_with_DE_node_pair(g,
                                                             bx,
                                                             use_de,
                                                             node_type,
                                                             type2idx,
                                                             num_fea,
                                                             num_neighbor,
                                                             max_dist,
                                                             k_hop,
                                                             m_rand)
        data.append([posA, posB, 1])

        neg_tmpB_id = random.choice(nodes)
        negA, negB = HDE.subgraph_sampling_with_DE_node_pair(g,
                                                             [bx[0], neg_tmpB_id],
                                                             use_de,
                                                             node_type,
                                                             type2idx,
                                                             num_fea,
                                                             num_neighbor,
                                                             max_dist,
                                                             k_hop,
                                                             m_rand)
        data.append([negA, negB, 0])

    # random.shuffle(data)
    data = np.array(data)
    data = data.reshape(num_batch, batch_size, 3)
    data_A = data[:, :, 0].tolist()
    data_B = data[:, :, 1].tolist()
    data_y = data[:, :, 2].tolist()
    for i in range(len(data_A)):
        for j in range(len(data[0])):
            data_A[i][j] = data_A[i][j].tolist()
            data_B[i][j] = data_B[i][j].tolist()
    data_A = np.squeeze(np.array(data_A))
    data_B = np.squeeze(np.array(data_B))
    data_y = np.squeeze(np.array(data_y))
    return data_A, data_B, data_y


def gen_fea_batch(G, root, fea_dict, K_HOP, USE_DE, NODE_TYPE, type2idx, NUM_FEA, NUM_NEIGHBOR):
    fea_batch = []
    mini_batch.append([root])
    # 两个相对位置的onehot
    if USE_DE:
        a = [0] * (NUM_FEA - NODE_TYPE) + HDE.type_encoder(root, NODE_TYPE, type2idx)
        # a = fea_dict[root].tolist() + HDE.type_encoder(root, NODE_TYPE, type2idx)
    else:
        a = HDE.type_encoder(root, NODE_TYPE, type2idx)
    fea_batch.append(np.asarray(a,  # [0] * NODE_TYPE +
                                dtype=np.float32
                                ).reshape(-1, NUM_FEA)
                     )
    # 1-ord
    # if len(G.neighbors(node)) < 1:
    #     print(node)
    # 邻居集合补上自己，因为subG可能有孤立点
    ns_1 = [list(np.random.choice(list(G.neighbors(node)) + [node],
                                  NUM_NEIGHBOR,
                                  replace=True))
            for node in mini_batch[-1]]
    mini_batch.append(ns_1[0])
    if USE_DE:
        de_1 = [
            np.concatenate([fea_dict[dest], np.asarray(HDE.type_encoder(dest, NODE_TYPE, type2idx))], axis=0)
            for dest in ns_1[0]
        ]
    else:
        de_1 = [
            np.asarray(HDE.type_encoder(dest, NODE_TYPE, type2idx))
            for dest in ns_1[0]
        ]

    fea_batch.append(np.asarray(de_1,
                                dtype=np.float32).reshape(1, -1)
                     )
    # 2-order
    ns_2 = [list(np.random.choice(list(G.neighbors(node)) + [node],
                                  NUM_NEIGHBOR,
                                  replace=True))
            for node in mini_batch[-1]]
    de_2 = []
    for i in range(len(ns_2)):
        tmp = []
        for j in range(len(ns_2[0])):
            if USE_DE:
                tmp.append(
                    # fea_dict[ns_2[i][j]] + HDE.type_encoder(ns_2[i][j], NODE_TYPE, type2idx)
                    np.concatenate([fea_dict[ns_2[i][j]], np.asarray(HDE.type_encoder(ns_2[i][j], NODE_TYPE, type2idx))], axis=0)
                )
            else:
                tmp.append(
                    # fea_dict[ns_2[i][j]] + HDE.type_encoder(ns_2[i][j], NODE_TYPE, type2idx)
                    np.asarray(HDE.type_encoder(ns_2[i][j], NODE_TYPE, type2idx))
                )
        de_2.append(tmp)

    fea_batch.append(np.asarray(de_2,
                                dtype=np.float32).reshape(1, -1)
                     )

    # 返回值：拼接（root节点自身的type encoding的特征;五个一阶邻居与关于AB节点的HDE;25个二阶邻居关于AB节点的HDE）
    # 这个作为最终输入到网络中的一个A节点的表示
    return np.concatenate(fea_batch, axis=1)


class HDEDataset(Dataset):
    """
    下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, a_data, b_data, y_data):
        self.a_data = a_data
        self.b_data = b_data
        self.y_data = y_data
        self.len = y_data.shape[0]

    def __getitem__(self, index):
        return self.a_data[index], self.b_data[index], self.y_data[index]

    def __len__(self):
        return self.len
