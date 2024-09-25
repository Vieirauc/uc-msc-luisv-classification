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
    # We add the identity matrix to the adjacency matrix
    np.fill_diagonal(row['adjacency_matrix'], 1)
    src, dest = np.nonzero(row['adjacency_matrix'])
    
    # Ensure src and dest are int64
    src = src.astype(np.int64)
    dest = dest.astype(np.int64)

    # Convert src and dest to PyTorch tensors
    src = torch.tensor(src, dtype=torch.int64)
    dest = torch.tensor(dest, dtype=torch.int64)
    
    # print(row)

    # Debugging statements
    #print(f"src dtype: {src.dtype}, dest dtype: {dest.dtype}")
    #print(f"src shape: {src.shape}, dest shape: {dest.shape}")
    #print(f"src: {src}, dest: {dest}")
    
    execution_graph = dgl.graph((src, dest))
    #print(execution_graph.num_nodes())
    try:
        execution_graph.ndata['features'] = torch.Tensor(row['feature_matrix'].reshape(row['size'],-1))
    except:
        return np.nan
    return execution_graph


def load_dataset(path):
    df = pd.read_csv(path + '.csv', sep=';', usecols = ['label', 'size', 'adjacency_matrix', 'feature_matrix'], dtype={'label' : np.bool_, 'size' : np.int32, 'adjacency_matrix': str, 'feature_matrix': str})
    df['adjacency_matrix'] = df[['adjacency_matrix', 'size']].apply(convert_to_adjancency, axis=1)
    df['feature_matrix'] = df['feature_matrix'].apply(convert_to_np_array)
    df['graphs'] = df.apply(create_dgl_graphs, axis=1)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# %%

if __name__ == '__main__':
    print('Loading')
    dataset_name = sys.argv[1]
    if not os.path.isfile(dataset_name + '.pkl'):
        df = load_dataset(dataset_name)
        df = df.to_pickle(sys.argv[1] + '.pkl')
    else:
        print('Pickle object already exists!')
# %%
