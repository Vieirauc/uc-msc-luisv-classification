from cfg_extraction_constants import CFG_FILE, LABEL
from cfg_feature_extraction import obtain_cfg_data_structures, obtain_node_types
from cfg_parsing import read_graph
import numpy as np
import os
import pandas as pd

import dgl
import torch as th
import torch.nn as nn
from dgl.nn import SortPooling

from save_cfg_features import write_cfgs_to_file

data_directory = "output-data"
file_cfg_data_mask = "functions-cfg-{}.csv"
projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]


def analyze_dot_cfg(cfg):
    print(cfg)  # prints the dot file
    print(cfg.get_name())

    for node in cfg.get_nodes():
        print(node)
        print(node.get_label())
        print(node.get_name())
    for edge in cfg.get_edges():
        print(edge)
        print(edge.get_source())
        print(edge.get_destination())


def obtain_identity(A):
    return np.identity(A.shape[0])


def obtain_diagonal_matrix(A_tilde):
    D = np.zeros(A_tilde.shape)
    for i in range(D.shape[0]):
        D[i][i] = np.sum(A_tilde[i, :])
    return D


def calculate_graph_convolution_layer(A, X, t=1):
    I = obtain_identity(A)
    A_tilde = A + I
    print(A_tilde)
    D_tilde = obtain_diagonal_matrix(A_tilde)
    print(D_tilde)

    # W: Trainable graph convolution parameters
    W1 = np.matrix([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    W2 = np.matrix([[0.0, 1.0, -2.0, 2.0], [1.0, 1.0, 7.0, -2.0], [1.0, 0.0, -1.0, 4.0]])
    W = [W1, W2]
    print(W1)
    print(W2)
    print(X)
    # Zt+1 = f(D_tilde ^-1 A_tilde Zt Wt)
    # Z0 = X
    D_inv = np.linalg.inv(D_tilde)
    print("D_inv\n", D_inv)
    D_A = np.dot(D_inv, A_tilde)
    print("D_A\n", D_A)
    D_A_X = np.dot(D_A, X)
    print("D_A_X\n", D_A_X)
    Ztp = np.dot(D_A_X, W[t-1])

    # f: nonlinear activation function (RELU)
    Ztp = np.maximum(Ztp, 0.0)
    print("Ztp{}\n".format(t), Ztp, "\n", Ztp.shape)

    # Ztp is the output activation matrix
    return Ztp


def read_cfg_file(project):
    filepath = os.path.join(data_directory, file_cfg_data_mask.format(project))
    df = pd.read_csv(filepath)
    df = df[df[CFG_FILE].notnull()]

    node_types = {}
    dataset_samples = []
    for index, row in df.iterrows():
        cfg_filepath = row[CFG_FILE]

        A, X, cfg_nx = obtain_cfg_data_structures(cfg_filepath)

        if A is not None:
            dataset_samples.append((cfg_filepath, row[LABEL], A.shape[0], list(A.getA1()), list(X.getA1())))

        if (index + 1) % 10 == 0:
            write_cfgs_to_file(project, dataset_samples)

        # TODO talvez daqui pra frente seja eliminado
        graphs = read_graph(cfg_filepath)

        if graphs is not None:
            cfg = graphs[0]
            analyze_dot_cfg(cfg)
            cfg_node_types = obtain_node_types(cfg)
            for cfg_node_type in cfg_node_types:
                if cfg_node_type not in node_types:
                    node_types[cfg_node_type] = cfg_node_types[cfg_node_type]
                else:
                    node_types[cfg_node_type] += cfg_node_types[cfg_node_type]
        print(index)

    write_cfgs_to_file(project, dataset_samples)
    node_types = dict(sorted(node_types.items(), key=lambda item: item[1], reverse=True))
    print(node_types)
    for item in node_types.items():
        print(item)
    print(len(node_types))


def extract_data_from_file(cfg_directory, cfg_filename):
    graph, X, Z1_t = None, None, None

    cfg_filepath = os.path.join(cfg_directory, cfg_filename)
    A, X, cfg_nx = obtain_cfg_data_structures(cfg_filepath)

    if A is not None:
        Z1_t = obtain_graph_convolution_layers(A, X)
        print("Z1:t of graph from file\n", Z1_t)

        graph = dgl.convert.from_networkx(cfg_nx)
        print("indegree", cfg_nx.in_degree())
        print("outdegree", cfg_nx.out_degree())
    return graph, X, Z1_t


def obtain_graph_convolution_layers(A, X, max_t=2):
    Z = []
    Zi_minus1 = X
    for i in range(0, max_t):
        Zi = calculate_graph_convolution_layer(A, Zi_minus1, t=i + 1)
        Z.append(Zi)
        Zi_minus1 = Zi
    Z1_t = Z[0]
    for i in range(0, max_t - 1):
        Z1_t = np.concatenate((Z1_t, Z[i + 1]), axis=1)

    return Z1_t


def obtain_sample_data():
    A = np.matrix([[0.0, 1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
    X = np.matrix([[2.0, 3.0], [1.0, 5.0], [7.0, 4.0], [8.0, 9.0], [6.0, 1.0]])
    Z1_t = obtain_graph_convolution_layers(A, X)
    print(Z1_t)

    # Sample graph from Yan2019 paper
    src_ids = th.tensor([i - 1 for i in [1, 1, 2, 3, 3]])
    dst_ids = th.tensor([i - 1 for i in [2, 4, 3, 4, 5]])
    g1 = dgl.graph((src_ids, dst_ids))

    return g1, X, Z1_t


def extract_sortpooling_layer(graph, Z1_t_th):
    # https://docs.dgl.ai/en/0.6.x/api/python/nn.pytorch.html#sortpooling
    # O k aqui é o quantidade de vértices que serão utilizados.
    # No artigo Zhang2018, o valor é definido como 60% dos grafos tendo mais de k vértices
    print("before_sortpooling\n", Z1_t_th)
    sortpool = SortPooling(k=3)
    sortpool_feats = sortpool(graph, Z1_t_th)
    print("sortpool_feats\n", sortpool_feats)
    return sortpool_feats


def extract_adaptive_max_pool(Z1_t_th):

    # The adaptive max pooling allows having different kernel sizes according to the input,
    # but the output always has the same dimensions.
    m = nn.AdaptiveMaxPool1d(3)  # THIS ALMOST WORK, BUT IT HAS A DIFFERENT STRIDE IN THE PAPER!
    amp = m(Z1_t_th)
    print("amp (AdaptiveMaxPool1d)\n", amp)
    return amp


def main():
    graph, X, Z1_t = extract_data_from_file("cfg-data", "6-cfg.dot")
    if Z1_t is not None:
        Z1_t_th = th.from_numpy(Z1_t)
        extract_sortpooling_layer(graph, Z1_t_th)
        extract_adaptive_max_pool(Z1_t_th)

    graph, X, Z1_t = obtain_sample_data()
    Z1_t_th = th.from_numpy(Z1_t)
    extract_sortpooling_layer(graph, Z1_t_th)
    extract_adaptive_max_pool(Z1_t_th)

    for project in projects[2:4]:
        read_cfg_file(project)


if __name__ == "__main__":
    main()
