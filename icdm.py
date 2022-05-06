# import pylab


import numpy as np

import networkx as nx

from tqdm import tqdm
# print(nx.__version__)
# import dgl
#
# dgl.dataloading.MultiLayerNeighborSampler(

import random
import numpy as np
import tensorflow as tf

from itertools import combinations

random.seed(0)  # 确保各个算法的 数据，邻居等信息一致，保证公平


# for ACM,  only use Author-Paper-Conf
USE_DE = 1

PATH ='./ds/imdb/MA.txt'

type2idx = {
    'M': 0,
    'A': 1,
    # 'C': 2,
    # 'T': 3
}


def load_ACM(test_ratio=0.2):
    # sp， train size
    # val size = test size
    G = nx.Graph()
    sp = 1 - test_ratio * 2

    NODE_TYPE = len(type2idx)
    edge_list = []

    # with open('./data/ACM_org_edge/TvsP.txt', 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip().split(' ')
    #         G.add_edge(line[0], line[1])
    #         edge_list.append(line)

    with open(PATH, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            G.add_edge(line[0], line[1])
            edge_list.append(line)

    # with open('./data/ACM_org_edge/edges_PC.txt', 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip().split(' ')
    #         edge_list.append(line)
    #         G.add_edge(line[0], line[1])

    # with open('./data/ACM_org_edge/PvsL.txt', 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip().split(' ')
    #         edge_list.append(line)
    #         G.add_edge(line[0], line[1])

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
    return G_train, G_val, G_test, NODE_TYPE, type2idx


G_train, G_val, G_test, NODE_TYPE, type2idx = load_ACM(test_ratio=0.3)

# print('ss')
# for node in G_train.nodes:
#     if len(list(G_train.neighbors(node))) < 1:
#         print(node)
#
# for node in G_test.nodes:
#     if len(list(G_test.neighbors(node))) < 1:
#         print(node)
#
# print('ss')


# def load_mini():
#     G = nx.Graph()
#     NODE_TYPE = 2
#     type2idx = {
#         'A': 0,
#         'B': 1
#     }
#     G.add_edge('A1', 'B2')
#     # G.add_edge('A1', 'A3')
#     G.add_edge('A2', 'B2')
#     G.add_edge('A3', 'B1')
#     G.add_edge('A3', 'B4')
#     G.add_edge('A4', 'B1')
#     G.add_edge('B1', 'B2')
#     G.add_edge('A1', 'B3')
#     G.add_edge('A1', 'B1')
#     G.add_edge('A1', 'A5')
#     G.add_edge('B4', 'B2')
#     return G, NODE_TYPE, type2idx
#

# G, NODE_TYPE, type2idx = load_mini()

# print(G.nodes)  # , G.nodes().index('A1'))
# for idx, val in enumerate(G.nodes):
#     print(f"node: {val} idx: {idx}", sep=' ')

# nx.single_source_shortest_path_length(G, node, cutoff=K)


NUM_NEIGHBOR = 5
mini_batch = []
fea_batch = []

EPOCH = 200
BATCH_SIZE = 32

K_HOP = 2  # 聚合K_HOP的邻居

EMB_DIM = 128
if USE_DE:
    NUM_FEA = (K_HOP + 2) * 4 + NODE_TYPE
else:
    NUM_FEA = NODE_TYPE
initializer = tf.contrib.layers.xavier_initializer(uniform=False)
regularizer = tf.contrib.layers.l2_regularizer(0.0)


# v1


# def dist_encoder(G, src, dest):
#     #
#     pass
def dist_encoder(src, dest, G, K_HOP, one_hot=True):
    # 计算在各个类型下的SPD=最少出现次数
    paths = list(nx.all_simple_paths(G, src, dest, cutoff=K_HOP+2))
    cnt = [K_HOP + 1] * NODE_TYPE  # 超过max_spd的默认截断
    # print(src, dest, paths)
    for path in paths:
        res = [0] * NODE_TYPE
        for i in range(1, len(path)):
            tmp = path[i][0]
            res[type2idx[tmp]] += 1
        # print(path, res)
        for k in range(NODE_TYPE):
            cnt[k] = min(cnt[k], res[k])
    # print(cnt)
    # pass
    if one_hot:
        # pass
        one_hot_list = [np.eye(K_HOP + 2, dtype=np.float64)[cnt[i]]
                        for i in range(NODE_TYPE)]
        return np.concatenate(one_hot_list)
    return cnt


# nx.all_simple_paths()
def type_encoder(node):
    res = [0] * NODE_TYPE
    res[type2idx[node[0]]] = 1.0
    return res


def gen_fea_batch(G, root, fea_dict, hop):
    fea_batch = []
    mini_batch.append([root])
    # 两个相对位置的onehot
    if USE_DE:
        a = [0] * (K_HOP + 2) * 4 + type_encoder(root)
    else:
        a = type_encoder(root)
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
            np.concatenate([fea_dict[dest], np.asarray(type_encoder(dest))], axis=0)
            for dest in ns_1[0]
        ]
    else:
        de_1 = [
            np.asarray(type_encoder(dest))
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
                    # fea_dict[ns_2[i][j]] + type_encoder(ns_2[i][j])
                    np.concatenate([fea_dict[ns_2[i][j]], np.asarray(type_encoder(ns_2[i][j]))], axis=0)
                )
            else:
                tmp.append(
                    # fea_dict[ns_2[i][j]] + type_encoder(ns_2[i][j])
                    np.asarray(type_encoder(ns_2[i][j]))
                )
        de_2.append(tmp)

    # de_2 = [
    #     dist_encoder(G, root, dest) + type_encoder(dest) if USE_DE else type_encoder(dest)
    #     for dest in ns_1[0]
    # ]
    fea_batch.append(np.asarray(de_2,
                                dtype=np.float32).reshape(1, -1)
                     )

    return np.concatenate(fea_batch, axis=1)


# v2
# print(G.degree('A1'), G.degree('A3'))

def subgraph_sampling_with_DE_node_pair(G, node_pair, K_HOP=3):
    # print('edge_DE .... ')
    [A, B] = node_pair
    A_ego = nx.ego_graph(G, A, radius=K_HOP)
    # print(nx.shortest_path_length(A_ego, A))
    B_ego = nx.ego_graph(G, B, radius=K_HOP)
    sub_G_for_AB = nx.compose(A_ego, B_ego)
    sub_G_for_AB.remove_edges_from(combinations(node_pair, 2))

    sub_G_nodes = sub_G_for_AB.nodes
    # print(sub_G_nodes)
    # 子图中所有点到 node pair的距离，
    SPD_based_on_node_pair = {}
    if USE_DE:
        for node in sub_G_nodes:
            # if node in node_pair: # 不要跳过node-pair， 聚合图里可能有A和B
            #     continue
            # print(node, node_pair)
            tmpA = dist_encoder(A, node, sub_G_for_AB, K_HOP)
            tmpB = dist_encoder(B, node, sub_G_for_AB, K_HOP)
            # TODO 这里 求和还是拼接？
            # SPD_based_on_node_pair[node] = np.sum([tmpA, tmpB], axis=0)
            SPD_based_on_node_pair[node] = np.concatenate([tmpA, tmpB], axis=0)
            # np.concatenate([tmpA, tmpB])

        # print(node, tmp)
    # print(SPD_based_on_node_pair)

    # A he B 的聚合图
    A_fea_batch = gen_fea_batch(sub_G_for_AB, A,
                                SPD_based_on_node_pair, K_HOP)
    B_fea_batch = gen_fea_batch(sub_G_for_AB, B,
                                SPD_based_on_node_pair, K_HOP)
    # return SPD_based_on_node_pair
    return A_fea_batch, B_fea_batch


# x, y = subgraph_sampling_with_DE_node_pair(G_train, ['A11097', 'P11564'], K_HOP=K_HOP)
#
# print('sss')


# dist_encoder('A1', 'B1', G)


# print(subgraph_sampling_with_DE('P0'))
#

def batch_data(G,
               # edge, label,
               batch_size=3):
    edge = list(G.edges)
    nodes = list(G.nodes)
    num_batch = int(len(edge) / batch_size)
    random.shuffle(edge)
    for idx in range(num_batch):
        # TODO add shuffle and random sample
        batch_edge = edge[idx * batch_size:(idx + 1) * batch_size]
        batch_label = [1.0] * batch_size
        # label[idx * batch_size:(idx + 1) * batch_size]

        batch_A_fea = []
        batch_B_fea = []
        batch_x = []
        batch_y = []
        #
        # neg_batch_A_fea = []
        # neg_batch_B_fea = []
        # neg_batch_y = []

        # for (edge, label) in zip(batch_edge, batch_label):
        for (bx, by) in zip(batch_edge, batch_label):
            # print(bx, by)

            # pos
            posA, posB = subgraph_sampling_with_DE_node_pair(G, bx, K_HOP=K_HOP)
            batch_A_fea.append(posA)
            batch_B_fea.append(posB)
            # batch_x.append(tmp_pos)
            # tmpB = np.asarray(subgraph_sampling_with_DE_node_pair(G, bx[1]), dtype=np.float32)
            # batch_B_fea.append(tmpB)
            batch_y.append(np.asarray(by, dtype=np.float32))

            # neg
            # batch_A_fea.append(tmpA)
            # TODO do not consider sampling pos as neg
            neg_tmpB_id = random.choice(nodes)
            # tmp_neg = np.asarray(subgraph_sampling_with_DE_node_pair(G, [bx[0], neg_tmpB_id]),
            #                      dtype=np.float32)
            # batch_x.append(tmp_neg)
            negA, negB = subgraph_sampling_with_DE_node_pair(G, [bx[0], neg_tmpB_id], K_HOP=K_HOP)
            batch_A_fea.append(negA)
            batch_B_fea.append(negB)
            batch_y.append(np.asarray(0.0, dtype=np.float32))

        # batch_pos_fea = np.squeeze(batch_pos_fea)
        # batch_neg_fea = np.squeeze(batch_neg_fea)

        yield np.asarray(np.squeeze(batch_A_fea)), np.asarray(np.squeeze(batch_B_fea)), np.asarray(
            batch_y).reshape(batch_size * 2, 1)


# print('s')
# A, B, label = batch_data(G_train).__next__()
# #
# print('ss')


# split data
def split(G, split=0.8):
    edge_list = list(G.edges)
    num_edge = len(edge_list)
    sp = int(num_edge * split)
    train_edge = edge_list[:sp]
    train_label = [1.0] * sp  # np.ones(sp)

    test_edge = edge_list[sp:]
    test_label = [1.0] * (num_edge - sp)  # np.ones(sp)
    # train_data = (train_edge, train_label]
    # test_data = [test_edge, test_label]
    return train_edge, train_label, test_edge, test_label
    # return train_edge, test_edge




def decode_node_attr(infos, hash_size_list, is_hash=False):
    # decode arbitrary num of node attr, len(infos) can be arbitrary number
    # work for both user and item
    fea_val_list = [tf.decode_csv(info,
                                  [[" "], [" "]],
                                  ":")[1]
                    for info in infos]
    if is_hash:
        fea_hash_list = [tf.string_to_hash_bucket(i, j)
                         for (i, j) in zip(fea_val_list, hash_size_list)]
        return fea_hash_list
    return fea_val_list


def GNN(fea, model='meirec'):
    """
    :param fea: fea_batch, [[0, 0, 4], [[0, 1, 1], [0, 1, 4], [0, 1, 4], [0, 1, 1], [1, 0, 1]]]
    :return:
    """
    with tf.variable_scope(name_or_scope='gnn', reuse=tf.AUTO_REUSE):
        # node = fea[0]
        # neigh = fea[1]
        # fea_size = neigh.shape[1]  # neigh.get_shape().as_list()[1]
        # fea_emb_mat = tf.
        node = fea[:, :NUM_FEA]
        neigh1 = fea[:, NUM_FEA:NUM_FEA * (NUM_NEIGHBOR + 1)]
        neigh1 = tf.reshape(neigh1, [-1, NUM_NEIGHBOR, NUM_FEA])

        neigh2 = fea[:, NUM_FEA * (NUM_NEIGHBOR + 1):]
        neigh2 = tf.reshape(neigh2, [-1, NUM_NEIGHBOR, NUM_NEIGHBOR, NUM_FEA])
        if model == 'meirec':
            # agg 2-ord
            neigh2_agg = tf.reduce_mean(neigh2, axis=2)
            tmp = tf.concat(
                [neigh1, neigh2_agg],
                axis=2
            )
            tmp = tf.layers.dense(tmp, 64,
                                  activation=tf.nn.elu,
                                  use_bias=True,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  name='tmp_proj'
                                  )

            # agg
            emb = tf.concat(
                [
                    node, tf.reduce_mean(tmp, axis=1)
                ],
                axis=1
            )
        emb = tf.layers.dense(emb, 64,
                              activation=tf.nn.elu,
                              use_bias=True,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name='emb_proj'
                              )
        emb = tf.layers.dense(emb, 64,
                              activation=tf.nn.elu,
                              use_bias=True,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name='emb_proj_2'
                              )

        # node_proj = tf.get_variable('node_proj',
        #                             [NUM_FEA, EMB_DIM],
        #                             initializer=initializer,
        #                             )
        # neigh_proj = tf.get_variable('neigh_proj',
        #                              [NUM_FEA, EMB_DIM],
        #                              initializer=initializer)
        # neigh_proj = tf.reshape(neigh_proj, [-1, NUM_NEIGHBOR, EMB_DIM])
        # emb = tf.concat([
        #     tf.matmul(node, node_proj),
        #     tf.reduce_mean(
        #         tf.reshape(tf.matmul(neigh, neigh_proj), [-1, NUM_NEIGHBOR, EMB_DIM]),
        #         axis=0, keep_dims=True)
        # ],
        #     axis=1)
        return emb


def LP(n1, n2, label):
    n1_emb = GNN(n1)
    n2_emb = GNN(n2)


    pred = tf.layers.dense(tf.concat([n1_emb, n2_emb], axis=1),
                           32,
                           activation=tf.nn.elu,
                           use_bias=True,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizer,
                           name='pred_layer'
                           )
    pred = tf.layers.dense(pred,
                           1,
                           activation=None,
                           use_bias=True,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizer,
                           name='pred_layer_2'
                           )
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label,
                                                                  logits=pred))
    auc, auc_op = tf.metrics.auc(labels=label,
                                 predictions=tf.nn.sigmoid(pred))
    # tf.metrics.ac
    return pred, loss, auc, auc_op, n1_emb, n2_emb


def tmp():
    for idx, val in enumerate(user_hash_size_list):
        all_emb_mat['user_{}_emb_mat'.format(idx)] = tf.get_variable('user_{}_emb_mat'.format(idx),
                                                                     [val, EMB_DIM],
                                                                     initializer=initializer)
    for idx, val in enumerate(item_hash_size_list):
        all_emb_mat['item_{}_emb_mat'.format(idx)] = tf.get_variable('item_{}_emb_mat'.format(idx),
                                                                     [val, EMB_DIM],
                                                                     initializer=initializer)
    u_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], u_info_hash[i])
        for i in range(len(u_info_hash))
    ]
    v_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], v_info_hash[i])
        for i in range(len(v_info_hash))
    ]

    u_fea_final = cat_fea_emb_list(u_fea_emb_list)
    v_fea_final = cat_fea_emb_list(v_fea_emb_list)

    batch_y = tf.expand_dims(batch_y, axis=1)

    i_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['item_{}_emb_mat'.format(i)], i_info_hash[i])
        for i in range(len(i_info_hash))
    ]
    i_fea_final = cat_fea_emb_list(i_fea_emb_list)
    print('u, v, i, shape: ', u_fea_final.shape, v_fea_final.shape, i_fea_final.shape)
    # # =========================================  u, v  friends embedding
    uf_fea_emb_list = [
        tf.nn.embedding_lookup(
            all_emb_mat['user_{}_emb_mat'.format(i)], uf_info_hash[i])
        for i in range(len(uf_info_hash))
    ]


# shape=(None, (NUM_NEIGHBOR + 1) * NUM_FEA)
A_holder = tf.placeholder(tf.float32,
                          shape=(None, (NUM_NEIGHBOR * NUM_NEIGHBOR + NUM_NEIGHBOR + 1) * NUM_FEA), name='a')
B_holder = tf.placeholder(tf.float32,
                          shape=(None, (NUM_NEIGHBOR * NUM_NEIGHBOR + NUM_NEIGHBOR + 1) * NUM_FEA), name='b')
y_holder = tf.placeholder(tf.float32, shape=(None, 1), name='y')

pred, loss, auc, auc_op, A_emb, B_emb = LP(A_holder, B_holder, y_holder)

op = tf.train.AdamOptimizer(0.001).minimize(loss)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

plot_x = []
plot_y = []

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    for ep in range(EPOCH):
        # train
        if 1:
            batch_A_fea, batch_B_fea, batch_y = batch_data(G_train, BATCH_SIZE).__next__()
            tra_A_emb, tra_B_emb, _, tra_pred, tra_loss, tra_auc_op, tra_auc = sess.run(
                [A_emb, B_emb, op, pred, loss, auc_op, auc],
                feed_dict={
                    A_holder: batch_A_fea,
                    B_holder: batch_B_fea,
                    y_holder: batch_y
                })
            print(USE_DE, PATH, ep, 'train: ', tra_loss, tra_auc)

        # val
        if 1:
            val_batch_A_fea, val_batch_B_fea, val_batch_y = batch_data(G_val, BATCH_SIZE).__next__()
            val_A_emb, val_B_emb, val_pred, val_loss, val_auc_op, val_auc = sess.run(
                [A_emb, B_emb, pred, loss, auc_op, auc],
                feed_dict={
                    A_holder: val_batch_A_fea,
                    B_holder: val_batch_B_fea,
                    y_holder: val_batch_y
                })
            print(ep, "val: ", val_loss, val_auc)

        # test
        if 1:
            test_batch_A_fea, test_batch_B_fea, test_batch_y = batch_data(G_test, BATCH_SIZE).__next__()
            test_A_emb, test_B_emb, test_pred, test_loss, test_auc_op, test_auc = sess.run(
                [A_emb, B_emb, pred, loss, auc_op, auc],
                feed_dict={
                    A_holder: test_batch_A_fea,
                    B_holder: test_batch_B_fea,
                    y_holder: test_batch_y
                })
            print(ep, "test: ", test_loss, test_auc)

# import matplotlib.pyplot as plt
#
# plt.plot(plot_x, plot_y, label='second line')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()

#
#
# path = nx.shortest_path(G, source='A1', target='B4')
# print(path)
# sg = nx.generators.ego.ego_graph(G, 'A1', 2)
# sg2 = nx.generators.ego.ego_graph(G, 'B4', 2)
# # nx.singl                  e_source_shortest_path_length
# print(sg)
# print(sg2)
# print('---- dgl -----')
# dgl_sg = dgl.from_networkx(sg)
# print(dgl_sg, dgl_sg.nodes(), dgl_sg.edges(), sep='\n')
#
# dgl_sg2 = dgl.from_networkx(sg2)
# print(dgl_sg2, dgl_sg2.nodes(), dgl_sg2.edges(), sep='\n')
#
# import dgl
# import dgl.nn as dglnn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
# dataloader = dgl.dataloading.NodeDataLoader(
#     g, train_nids, sampler,
#     batch_size=1024,
#     shuffle=True,
#     drop_last=False,
#     num_workers=4)

# dgl_G = dgl.from_networkx(G)
# print(dgl_G)
# import dgl
# import torch as th
#
# # 创建一个具有3种节点类型和3种边类型的异构图
# graph_data = {
#     ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
#     ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
#     ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
# }
# g = dgl.heterograph(graph_data)
# g.ntypes
# g.etypes
# g.canonical_etypes
# print(g)
# # dgl.sampling.sample_neighbors(g, nodes
# sg = dgl.node_subgraph(g, {'drug': [1, 2], 'gene': [2]})
# print('subgraph', sg)
# print(g.nodes('drug'), g.nodes('gene'))
# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
# dataloader = dgl.dataloading.NodeDataLoader(g,
#                                             # {'drug': [1]},
#                                             {'gene': [2]},
#                                             sampler,
#                                             batch_size=5,
#                                             shuffle=True,
#                                             drop_last=False,
#                                             num_workers=4)
# input_nodes, output_nodes, blocks = next(iter(dataloader))
# print(len(input_nodes), len(output_nodes), len(blocks))
# print(input_nodes)
# print(output_nodes)
# print(blocks)

# if 0:
#     def dist_encoder(path):
#         res = [0] * NODE_TYPE
#         # omit root node
#         # print(path)
#         for i in range(1, len(path)):
#             tmp = path[i][0]
#             res[type2idx[tmp]] += 1
#         return res
#
#
#     de = dist_encoder(path)
#
#     dist = nx.shortest_path_length(G, source='A1', target='B4')
#
#     print('源节点为A1，终点为B4：', path, " dist: ", dist, " DE for {}: ".format(path[0]), de)
#
#     print('给网路设置布局...')
#     pos = nx.shell_layout(G)
#     print('画出网络图像：')
#     nx.draw(G, pos, with_labels=True, node_color='white',
#             edge_color='red', node_size=400, alpha=0.5)
#     pylab.title('Self_Define Net', fontsize=15)
#     pylab.show()
