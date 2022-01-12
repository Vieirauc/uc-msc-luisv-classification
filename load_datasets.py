# %%
import pandas as pd
import sys
import os
import numpy as np
from io import StringIO
import dgl
import torch
# %%

def convert_to_adjancency(row):
    return np.array(eval(row['adjacency_matrix'])).reshape(row['size'],-1)

def convert_to_np_array(row):
    return np.array(eval(row))

def create_dgl_graphs(row):
    src, dest = np.nonzero(row['adjacency_matrix'])
    execution_graph = dgl.graph((src, dest))
    execution_graph.ndata['features'] = torch.Tensor(row['feature_matrix'].reshape(row['size'],-1))
    return execution_graph


def load_dataset(path):
    df = pd.read_csv(path + '.csv', sep=';', usecols = ['label', 'size', 'adjacency_matrix', 'feature_matrix'], dtype={'label' : np.bool_, 'size' : np.int32, 'adjacency_matrix': str, 'feature_matrix': str})
    df['adjacency_matrix'] = df[['adjacency_matrix', 'size']].apply(convert_to_adjancency, axis=1)
    df['feature_matrix'] = df['feature_matrix'].apply(convert_to_np_array)
    df['graphs'] = df.apply(create_dgl_graphs, axis=1)
    return df

if __name__ == '__main__':
    print('Loading')
    dataset_name = sys.argv[1]
    if not os.path.isfile(dataset_name + '.pkl'):
        df = load_dataset(dataset_name)
        df = df.to_pickle(sys.argv[1] + '.pkl')
    else:
        print('Pickle object already exists!')
# %%
