import networkx as nx
import numpy as np
from utils import process
from itertools import combinations


def dist_encoder(src, dest, G, K_HOP, NODE_TYPE, type2idx, max_dist, one_hot=True):
    # 计算在各个类型下的SPD=最少出现次数
    paths = list(nx.all_simple_paths(G, src, dest, cutoff=max_dist+1))
    cnt = [max_dist] * NODE_TYPE  # 超过max_spd的默认截断
    for path in paths:
        res = [0] * NODE_TYPE
        for i in range(1, len(path)):
            tmp = path[i][0]
            res[type2idx[tmp]] += 1
        # print(path, res)
        for k in range(NODE_TYPE):
            cnt[k] = min(cnt[k], res[k])
    if one_hot:
        # pass
        one_hot_list = [np.eye(max_dist + 1, dtype=np.float64)[cnt[i]]
                        for i in range(NODE_TYPE)]
        return np.concatenate(one_hot_list)
    return cnt


def type_encoder(node, NODE_TYPE, type2idx):
    res = [0] * NODE_TYPE
    res[type2idx[node[0]]] = 1.0
    return res


def subgraph_sampling_with_DE_node_pair(G,
                                        node_pair,
                                        USE_DE,
                                        NODE_TYPE,
                                        type2idx,
                                        NUM_FEA,
                                        NUM_NEIGHBOR,
                                        max_dist,
                                        K_HOP,
                                        m_rand):
    [A, B] = node_pair
    A_ego = nx.ego_graph(G, A, radius=K_HOP)
    B_ego = nx.ego_graph(G, B, radius=K_HOP)
    sub_G_for_AB = nx.compose(A_ego, B_ego)
    sub_G_for_AB.remove_edges_from(combinations(node_pair, 2))

    sub_G_nodes = sub_G_for_AB.nodes

    value_a, idx_dict = gen_params(sub_G_for_AB, type2idx, K_HOP, A, m_rand)
    value_b, idx_dict = gen_params(sub_G_for_AB, type2idx, K_HOP, B, m_rand)
    # 子图中所有点到 node pair的距离，
    SPD_based_on_node_pair = {}
    if USE_DE:
        # 分别计算egograph中所有节点关于节点集(A;B)的HDE
        for node in sub_G_nodes:
            # tmpA = dist_encoder(A, node, sub_G_for_AB, K_HOP, NODE_TYPE, type2idx, max_dist)
            # tmpB = dist_encoder(B, node, sub_G_for_AB, K_HOP, NODE_TYPE, type2idx, max_dist)
            tmpA = fast_dist_enc(node, A, value_a, idx_dict)
            tmpB = fast_dist_enc(node, B, value_b, idx_dict)
            SPD_based_on_node_pair[node] = np.concatenate([tmpA, tmpB], axis=0)

        # print(node, tmp)
    # print(SPD_based_on_node_pair)

    # A he B 的聚合图
    A_fea_batch = process.gen_fea_batch(sub_G_for_AB,
                                        A,
                                        SPD_based_on_node_pair,
                                        K_HOP,
                                        USE_DE,
                                        NODE_TYPE,
                                        type2idx,
                                        NUM_FEA,
                                        NUM_NEIGHBOR)
    B_fea_batch = process.gen_fea_batch(sub_G_for_AB,
                                        B,
                                        SPD_based_on_node_pair,
                                        K_HOP,
                                        USE_DE,
                                        NODE_TYPE,
                                        type2idx,
                                        NUM_FEA,
                                        NUM_NEIGHBOR)
    # return SPD_based_on_node_pair
    return A_fea_batch, B_fea_batch


def build_type_list(g, type2idx):
    node_list = g.nodes()
    type_dic = {}
    idx_dict = {}
    type_list = []
    for i, node in enumerate(node_list):
        one_hot = np.eye(len(type2idx))[type2idx[node[0]]]
        type_list.append(one_hot)
        type_dic[node] = one_hot
        idx_dict[node] = i
    type_list = np.array(type_list)

    return type_list, type_dic, idx_dict


def gen_params(g, type2idx, k_hop, node, m_rand):
    k_order = k_hop + 2
    type_list, type_dict, idx_dict = build_type_list(g, type2idx)
    adj = nx.to_scipy_sparse_matrix(g)
    epsilon = 1e-6
    adj = adj / (adj.sum(1) + epsilon)

    # compute k order reachability matrix
    w_k = []
    for i in range(k_order):
        w_k.append(adj)
        adj = adj @ adj

    is_target = np.zeros([len(g), 1])
    is_target[idx_dict[node], 0] = 1
    query = np.hstack([is_target, type_list])

    query = query @ m_rand

    value = []
    for w in w_k:
        value.append(w @ query)

    return value, idx_dict


def fast_dist_enc(src, dst, value, idx_dict):
    landing_p_list = []
    for value_i in value:
        value_i = np.array(value_i)
        landing_p_list.append(value_i[idx_dict[src]])

    dist_enc = np.concatenate(landing_p_list)
    return dist_enc
