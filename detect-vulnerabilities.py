# %%
import numpy as np
import pandas as pd
import os
from load_datasets import load_dataset
import sys
import dgl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from dgl.nn import SortPooling
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.pytorch import GraphConv, GATConv
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#from torchvision import transforms

# %%

project = 'linux' # 'gecko-dev'#'linux'
version = 'v0.4'

dataset_name = 'datasets/cfg-dataset-{}-{}'.format(project, version)
# dataset_name = 'datasets/cfg-dataset-gecko-dev-v0.3'
if not os.path.isfile(dataset_name + '.pkl'):
    df = load_dataset(dataset_name)
    df = df.to_pickle(sys.argv[1] + '.pkl')
else:
    df = pd.read_pickle(dataset_name + '.pkl')

# %%

ZNORM = "znorm"
MINMAX = "minmax"
normalization = MINMAX #ZNORM

SORTPOOLING = "sort_pooling"
ADAPTIVEMAXPOOLING = "adaptive_max_pooling"

pooling_type = ADAPTIVEMAXPOOLING #SORTPOOLING

heads = 4 # 2
num_features = 11 #+ 8  # 8 features related to memory management
num_epochs = 500 #2000 #500 # 1000
hidden_dimension_options = [[32, 32, 32, 32]] #[[128, 64, 32, 32], [32, 32, 32, 32]] #[32, 64, 128, [128, 64, 32, 32], [32, 32, 32, 32]] # [32, 64, 128] # [[128, 64, 32, 32], 32, 64, 128]
sample_weight_value = 80 #90 #100 #80 #60 # 40


##################################################################################

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, sortpooling_k=3):
        super(GraphClassifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.classify = nn.Linear(hidden_dim, n_classes)
        # Added by Ze
        self.hidden_dim = hidden_dim
        self.sortpooling_k = sortpooling_k
        self.sortpool = SortPooling(k=sortpooling_k)
        self.conv1D = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1)


    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['features'].float() #g.in_degrees().reshape(-1, 1).float()
        #print(h.shape)

        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        g.ndata['h'] = h

        #########################################
        #print("g.ndata[h].shape (+h):", h.shape)
        h = self.sortpool(g, h)
        #print("after sortpool:", h.shape)

        current_batch_size = h.shape[0]

        h = h.reshape(current_batch_size, self.hidden_dim, self.sortpooling_k)
        #print("after resize:", h.shape)

        h = F.relu(self.conv1D(h))
        #print("after conv1d:", h.shape)

        #########################################

        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        #print(hg.shape)

        h = torch.squeeze(h)
        return self.classify(h) # era hg antes


class GATGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, sortpooling_k=3):
        super(GATGraphClassifier, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads)  # allow_zero_in_degree=True
        #self.conv2 = GATConv(hidden_dim * heads, n_classes, 1)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, 1)
        # self.classify = nn.Linear(hidden_dim * sortpooling_k, n_classes) # to use without the final convolution
        self.classify = nn.Linear(hidden_dim, n_classes)
        # Added by Ze
        self.hidden_dim = hidden_dim
        self.sortpooling_k = sortpooling_k
        self.sortpool = SortPooling(k=sortpooling_k)
        self.conv1D = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1)
        self.avgpooling = AvgPooling()
        self.drop = nn.Dropout(p = 0.3)


    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['features'].float()

        #print(h.shape)
        bs = h.shape[0]
        h = F.relu(self.conv1(g, h))
        #print(h.shape)
        h = h.reshape(bs, -1)
        #print(h.shape)
        h = F.relu(self.conv2(g, h))
        #print(h.shape)
        h = h.reshape(bs, -1)
        #print(h.shape)
        h = self.drop(h)
        #print(h.shape)
        #h = self.avgpooling(g, h)

        h = self.sortpool(g, h)
        #print("after sortpool:", h.shape)

        current_batch_size = h.shape[0]

        h = h.reshape(current_batch_size, self.hidden_dim, self.sortpooling_k)
        #print("after resize:", h.shape)

        h = F.relu(self.conv1D(h))
        #print("after Conv1d:", h.shape)

        h = torch.squeeze(h)
        #print("after squeeze:", h.shape)

        # TODO : Verficar com o Nuno: será que aqui que é para aplicar o SortPooling ?
#         hmax = self.maxpooling(g, h)
#         h = torch.cat([havg, hmax], 1)
        #print(h.shape)

        return self.classify(h)


    def forward_amp(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['features'].float()

        print(h.shape)
        bs = h.shape[0]
        h = F.relu(self.conv1(g, h))
        print("conv1:", h.shape)
        h = h.reshape(bs, -1)
        print("reshape:", h.shape)
        h = F.relu(self.conv2(g, h))
        print("conv2:", h.shape)
        h = h.reshape(bs, -1)
        print("reshape", h.shape)
        #h = self.drop(h)
        #print("after drop:", h.shape)
        ##h = self.avgpooling(g, h)

#torch.Size([657, 10])
#conv1: torch.Size([657, 4, 32])
#reshape: torch.Size([657, 128])
#conv2: torch.Size([657, 1, 32])
#reshape torch.Size([657, 32])
#after drop: torch.Size([657, 32])

        if pooling_type == SORTPOOLING:
            h = self.sortpool(g, h)
            #print("after sortpool:", h.shape)

            current_batch_size = h.shape[0]

            h = h.reshape(current_batch_size, self.hidden_dim, self.sortpooling_k)
            #h = h.reshape(current_batch_size, self.sortpooling_k, self.hidden_dim)
            #print("after resize:", h.shape)

            h = F.relu(self.conv1D(h))
            #print("after Conv1d:", h.shape)

            h = torch.squeeze(h)
            #print("after squeeze:", h.shape)
        elif pooling_type == ADAPTIVEMAXPOOLING:

            #h1 = h.reshape(bs, -1)
            #print("reshape (h1.shape):", h1.shape)
            #h1 = self.drop(h1)
            #print("drop (h1.shape):", h1.shape)
            #h1 = self.sortpool(g, h1)
            #print("sortpool (h1.shape):", h1.shape)
            h = torch.unsqueeze(h, 0)
            amp = nn.AdaptiveMaxPool2d((5,7))
            h = amp(h)
            print("After AMP:", h.shape)
            #g.ndata['h'] = h
            #hg = dgl.mean_nodes(g, 'h')


        # TODO : Verficar com o Nuno: será que aqui que é para aplicar o SortPooling ?
#         hmax = self.maxpooling(g, h)
#         h = torch.cat([havg, hmax], 1)
        #print(h.shape)

        return self.classify(h) # self.classify(h)


class GATGraphClassifier4HiddenLayers(nn.Module):
    def __init__(self, in_dim, hidden_dimensions, n_classes, sortpooling_k=3):
        super(GATGraphClassifier4HiddenLayers, self).__init__()

        self.sortpooling_k = sortpooling_k
        self.sortpool = SortPooling(k=sortpooling_k)

        self.conv1 = GATConv(in_dim, hidden_dimensions[0], heads)  # allow_zero_in_degree=True
        self.conv2 = GATConv(hidden_dimensions[0] * heads, hidden_dimensions[1], heads)
        self.conv3 = GATConv(hidden_dimensions[1] * heads, hidden_dimensions[2], heads)
        self.conv4 = GATConv(hidden_dimensions[2] * heads, hidden_dimensions[3], 1)

        self.conv1D = nn.Conv1d(in_channels=hidden_dimensions[3], out_channels=hidden_dimensions[3], kernel_size=self.sortpooling_k, stride=1) # antes o kernal size era 3

        ###############################################################
        # TESTE SORT POOL 25/03/2022 (concat)

        #ZE 11/05 self.conv1 = GraphConv(in_dim, hidden_dimensions[0], allow_zero_in_degree=True)
        #ZE 11/05 self.conv2 = GraphConv(hidden_dimensions[0], hidden_dimensions[1], allow_zero_in_degree=True)
        #ZE 11/05 self.conv3 = GraphConv(hidden_dimensions[1], hidden_dimensions[2], allow_zero_in_degree=True)
        #ZE 11/05 self.conv4 = GraphConv(hidden_dimensions[2], hidden_dimensions[3], allow_zero_in_degree=True)

        #ZE 11/05 self.conv1D = nn.Conv1d(in_channels=sum(hidden_dimensions), out_channels=hidden_dimensions[3], kernel_size=self.sortpooling_k, stride=1) # teste sort pool


        self.conv1 = GraphConv(in_dim, hidden_dimensions[0], allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dimensions[0], hidden_dimensions[1], allow_zero_in_degree=True)
        self.conv3 = GraphConv(hidden_dimensions[1], hidden_dimensions[2], allow_zero_in_degree=True)
        self.conv4 = GraphConv(hidden_dimensions[2], hidden_dimensions[3], allow_zero_in_degree=True)

        self.conv1D = nn.Conv1d(in_channels=sum(hidden_dimensions), out_channels=hidden_dimensions[3], kernel_size=self.sortpooling_k, stride=1)

        ###############################################################

        self.classify = nn.Linear(hidden_dimensions[3], n_classes)

        self.amp_shape = (5, 5)
        # self.conv1Damp = nn.Conv1d(in_channels=self.amp_shape[0] * self.amp_shape[1], out_channels=hidden_dimensions[3], kernel_size=1, stride=1)
        self.conv1Damp = nn.Conv1d(in_channels=hidden_dimensions[3]**2, out_channels=hidden_dimensions[3], kernel_size=1, stride=1)
        #self.conv1Damp = nn.Conv1d(in_channels=25, out_channels=hidden_dimensions[3], kernel_size=1, stride=1)
        self.avgpooling = AvgPooling()
        self.drop = nn.Dropout(p = 0.3)


    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        #print(type(g))

        # Katz centrality does not work (maybe related to eigen values and eigen vectors)
        nx_g = dgl.to_networkx(g)
        #centrality = nx.degree_centrality(nx_g)
        #print(type(nx_g))
        #print(nx_g.edges)
        #print(nx_g.nodes)
        #print(nx.eigenvector_centrality(nx_g))

        #print("degree_centrality", nx.degree_centrality(nx_g))
        ##print("katz_centrality", nx.katz_centrality(nx_g))
        #print("closeness_centrality", nx.closeness_centrality(nx_g))
        centrality = nx.closeness_centrality(nx_g)
        #print("centrality", type(centrality))
        #print(centrality.keys())
        #print(centrality.values())
        centrality = torch.FloatTensor(list(centrality.values()))
        #print(type(centrality), centrality.shape)

        #print(g.ndata)
        h = g.ndata['features'].float()
        if False: #True: # h.shape[0] < 500:
            #print(type(centrality), centrality.shape)

            #print("h", type(h), h.shape)
            #print("h[0]", h[0].tolist())

            #print("h", h.tolist())
            #print("centrality", centrality.tolist())
            #print("centrality[0]", centrality[0])

            #print("h.T.shape", h.T.shape)
            #print("h.T.mul(centrality)", h.T.mul(centrality).shape)

            teste_mul = h.T.mul(centrality).T
            #print("h.mul(centrality)", teste_mul.shape)
            #print(teste_mul[0].tolist())
            #print(teste_mul.tolist())

            #teste_mul = torch.matmul(centrality, h)
            #print("torch.matmul(centrality, h)", teste_mul.shape)

            h = teste_mul

        #teste_mul = torch.dot(h, centrality)
        #print("torch.dot(h, centrality)", teste_mul.shape)

        #teste_mul = torch.dot(centrality, h)
        #print("torch.dot(centrality, h)", teste_mul.shape)

        #print("Initial Shape:", h.shape)
        bs = h.shape[0]
        #13/06 print("bs", bs)
        h1 = F.relu(self.conv1(g, h))
        #print("h1", h1.shape)
        h1 = h1.reshape(bs, -1)
        #print(h.shape)
        h2 = F.relu(self.conv2(g, h1))
        #print("h2", h2.shape)
        h2 = h2.reshape(bs, -1)
        #print(h.shape)
        h3 = F.relu(self.conv3(g, h2))
        #print("h3", h3.shape)
        h3 = h3.reshape(bs, -1)
        #print("h3", h3.shape)
        h4 = F.relu(self.conv4(g, h3))
        #print("h4", h4.shape)
        h4 = h4.reshape(bs, -1)
        #print("h4.shape (after reshape):", h4.shape)

        h_cat = torch.cat((h1, h2, h3, h4), 1)
        h_concat = h_cat
        #13/06 print("h_concat.shape:", h_concat.shape)

        h4 = self.drop(h4)
        h4_output = h4
        #print("h4.shape (after drop):", h4.shape) # h4.shape (after drop): torch.Size([819, 32])
        #h = self.avgpooling(g, h)

        # TODO: isso está a mais, mas estamos usando o current_batch_size daqui
        h4 = self.sortpool(g, h4)
        #print("after h4 sortpool:", h4.shape) # after h4 sortpool: torch.Size([32, 192])
        current_batch_size = h4.shape[0]
        #print("current_batch_size:", current_batch_size) # current_batch_size: 32
        #h4 = h4.reshape(current_batch_size, self.conv4._out_feats, self.sortpooling_k)

        #print("before pooling (h_cat):", h_cat.shape)
        if pooling_type == SORTPOOLING:
            #print("before sortpool_cat:", h_cat.shape) # before sortpool_cat: torch.Size([819, 416])
            h_cat = self.sortpool(g, h_cat)
            #print("after sortpool_cat:", h_cat.shape) # after sortpool_cat: torch.Size([32, 2496])
            h_features = h_cat
            h_cat = h_cat.reshape(current_batch_size, int(h_cat.shape[1] / self.sortpooling_k), self.sortpooling_k)
            #print("after reshape:", h_cat.shape) # after reshape: torch.Size([32, 416, 6])

            #print(type(self.conv1).__name__)
            if type(self.conv1).__name__ == "GATConv":
                #print("h4", h4.shape) # h4 torch.Size([32, 32, 6])
                h4 = self.conv1D(h4)
                #print("h4 (after conv1D)", h4.shape)
                h_cat = F.relu(h4)
            else:
                h_cat = F.relu(self.conv1D(h_cat))
            #print("after relu (conv1D):", h_cat.shape) # after relu (conv1D): torch.Size([32, 32, 4])

            h_cat = torch.squeeze(h_cat)
            #print("after squeeze:", h_cat.shape) # after squeeze: torch.Size([32, 32, 4])
        elif pooling_type == ADAPTIVEMAXPOOLING:
            #13/06 print("[380] h_cat.shape", h_cat.shape)
            #h_cat = h_cat.reshape(current_batch_size, -1) # AQUI: funciona, mas dá erro: 31/03
            #print("h_cat.shape", h_cat.shape)

            #h_cat = torch.unsqueeze(h_cat, 0) # TODO: ACHO QUE AQUI QUE ESTÁ ERRADO!!!! # comentei em 06-04-2022
            #print("after.unsqueeze:", h_cat.shape)

            #h_cat = h_cat.reshape(bs, 32, 4) # AQUI: funciona, mas dá erro no conv1D: 09/06
            h_cat = h_cat.reshape(32, bs, 4) # AQUI: funciona, mas dá erro no save_features: 09/06
            #print("h_cat.shape", h_cat.shape)

            # TODO (3,3) (5,5) (7,7) -> experimentar este (valores tipicos de imagens)
            width_amp = current_batch_size #32
            height_amp = 32
            amp = nn.AdaptiveMaxPool2d((width_amp, height_amp))
            h_cat_aux = amp(h_cat)
            #print("h4_output", h4_output.shape)
            h4_output = h4_output.reshape(32, bs, 1)
            #print("h4 after reshape (seems useless)", h4_output.shape)
            h4_aux = amp(h4_output)
            #13/06 print("h_cat_aux.shape", h_cat_aux.shape)

            width_amp = self.amp_shape[0]
            height_amp = self.amp_shape[1]
            amp = nn.AdaptiveMaxPool2d((width_amp, height_amp))
            h_cat_amp = amp(h_cat)
            #13/06 print("h_cat_amp.shape (after amp):", h_cat_amp.shape)
            #print(h_cat)

            #conv2d = nn.Conv2d(5, 2, 3, stride=2)
            #h_cat2d = F.relu(conv2d(h_cat))
            #print("h_cat2d", h_cat2d.shape)

            # h_cat = h_cat.reshape(h_cat.shape[0], int(h_cat.shape[1] * h_cat.shape[2])) # 31-03-2022
            #print("before reshape, current batch_size:", current_batch_size)
            #print(h_cat)
            #h_cat = h_cat.reshape(bs, int(h_cat.shape[1] / height_amp), height_amp)
            #h_cat = h_cat.reshape(1, int((h_cat.shape[1] * h_cat.shape[2]) / 2), 2)
            #h_cat_aux = h_cat_aux.reshape(current_batch_size, h_cat_aux.shape[2], 1) # comentado em 06-04-2022
            h_cat_aux = h_cat_aux.reshape(current_batch_size, -1)# 32, 32)
            #print("after reshape (h_cat_aux):", h_cat_aux.shape)
            h_cat_aux = torch.unsqueeze(h_cat_aux, 2)
            #h_cat = torch.squeeze(h_cat, 2)
            h_cat = F.relu(self.conv1Damp(h_cat_aux))
            h_cat = torch.squeeze(h_cat)

            ######
            # Trying to transform the h4 in the same manner as h_cat
            h4_aux = h4_aux.reshape(current_batch_size, -1)
            #print("h4_aux after reshape", h4_aux.shape)
            h4_aux = torch.unsqueeze(h4_aux, 2)
            #print("h4_aux after unsqueeze", h4_aux.shape)
            h4_output = F.relu(self.conv1Damp(h4_aux))
            #print("h4_output after relu", h4_output.shape)
            h4_output = torch.squeeze(h4_output)
            #print("final h4_output", h4_output.shape)

            ######
            h_cat_amp = h_cat_amp.reshape(self.amp_shape[0] * self.amp_shape[1], -1)
            #h_cat = F.relu(self.conv1Damp(h_cat_amp))
            #print("after conv1Damp (h_cat):", h_cat.shape)
            ######

            ##h_cat_amp = h_cat_amp.reshape(current_batch_size, self.amp_shape[0] * self.amp_shape[1], -1)

            #h_cat_aux = torch.unsqueeze(h_cat_aux, 2)
            #h_cat = torch.squeeze(h_cat, 2)
            #h_cat = F.relu(self.conv1Damp(h_cat_aux))
            ##h_cat = F.relu(self.conv1Damp(h_cat_amp))
            ##print("after conv1Damp (h_cat):", h_cat.shape)

            ##h_cat = torch.squeeze(h_cat)
            #h_features = h_cat_amp ## COMENTADO EM 13/06
            h_features = torch.squeeze(h_cat_aux) # ADICIONADO EM 13/06 # alterado em 20/06

            # TODO: aqui teria o VGG, mas precisamos melhorar as features antes disso.

        #h = h.reshape(current_batch_size, self.sortpooling_k, self.hidden_dim)
        #print("after resize:", h.shape)

        #### h4 = F.relu(self.conv1D(h4)) # teste sort pooling 25/03/2021
        #print("after Conv1d:", h.shape)

        #### h4 = torch.squeeze(h4) # teste sort pooling 25/03/2021
        #print("after squeeze:", h.shape)

        # TODO : Verficar com o Nuno: será que aqui que é para aplicar o SortPooling ?
#         hmax = self.maxpooling(g, h)
#         h = torch.cat([havg, hmax], 1)
        #print(h.shape)

        ###############################################################
        # TESTE SORT POOL 25/03/2022
        #print("h4.shape:", h4.shape)

        #print(h_cat.shape)
        #print("h_cat before classify", h_cat.shape)
        classification = self.classify(h_cat)
        #print("classification:", classification.shape)
        #print("h4_output before classify", h4_output.shape)
        classification_h4 = self.classify(h4_output)
        #print("classification_h4:", classification_h4.shape)
        #13/06 print("classification.shape:", classification.shape)
        #print(h_cat_amp.squeeze().tolist())
        return classification, h_concat, h_features #self.classify(h_cat)
        ###############################################################
        # sem o sort pool
        #return self.classify(h4)
        ###############################################################


# Load and Process dataset
# %%
df['label'] = torch.tensor(df['label'].astype(np.int8))
#print("sum(df[label]): ", sum(df['label']))
df['sample_weight'] = torch.tensor(1 + (df['label'].astype(np.int8) * sample_weight_value))
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
sample_weights = df['sample_weight'].values
# %%
random_seed = 42
test_split = 0.3
# Creating data indices for training and validation splits:
dataset_size = len(df)
indices = np.arange(dataset_size)
split = int(np.floor(test_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, test_indices = indices[split : ], indices[:split]

trainset = df[['graphs', 'label']].values[train_indices]
testset = df[['graphs', 'label']].values[test_indices]

###########################################################

## Normalization

all_feature_train_data = trainset[0,0].ndata['features']
all_feature_test_data = testset[0,0].ndata['features']
print(all_feature_train_data.shape, all_feature_test_data.shape)

for i in range(1, len(trainset)):
    all_feature_train_data = torch.cat((all_feature_train_data, trainset[i, 0].ndata['features']), dim=0)

for i in range(1, len(testset)):
    all_feature_test_data = torch.cat((all_feature_test_data, testset[i, 0].ndata['features']), dim=0)

feat_amin_train = torch.amin(all_feature_train_data, 0)
feat_amax_train = torch.amax(all_feature_train_data, 0)
feat_mean_train = torch.mean(all_feature_train_data, 0)
feat_std_train = torch.std(all_feature_train_data, 0)

feat_amin_test = torch.amin(all_feature_test_data, 0)
feat_amax_test = torch.amax(all_feature_test_data, 0)
feat_mean_test = torch.mean(all_feature_test_data, 0)
feat_std_test = torch.std(all_feature_test_data, 0)


def normalize_minmax(dataset, feat_minimum, feat_maximum):
    # as the minimum is always zero, the min-max normalization can be simplified with the division by the maximum value
    for i in range(len(dataset)):
        dataset[i, 0].ndata['features'] = torch.div(torch.sub(dataset[i, 0].ndata['features'], feat_minimum), torch.div(feat_maximum, feat_minimum))
    return dataset


def normalize_znorm(dataset, feat_mean, feat_std):
    for i in range(len(dataset)):
        dataset[i, 0].ndata['features'] = torch.div(torch.sub(dataset[i, 0].ndata['features'], feat_mean), feat_std)
    return dataset


print("Normalization to be performed")
if normalization == ZNORM:
    trainset = normalize_znorm(trainset, feat_mean_train, feat_std_train)
    testset = normalize_znorm(testset, feat_mean_test, feat_std_test)
if normalization == MINMAX: 
    trainset = normalize_minmax(trainset, feat_amin_train, feat_amax_train)
    testset = normalize_minmax(testset, feat_amin_test, feat_amax_test)


###########################################################

def adjust_dataset(dataset):
    # Removes the column where all the features are zero
    for i in range(len(dataset)):
        t = dataset[i, 0].ndata['features']
        t = torch.cat((t[:,0:3], t[:,4:]), 1)
        dataset[i, 0].ndata['features'] = t
    return dataset

#  Removes one feature as it is always zero (no node was assigned to type "numeric constant")
#print(trainset.shape)
if normalization is not None or normalization == "":
    trainset = adjust_dataset(trainset)
    testset = adjust_dataset(testset)
    num_features -= 1
#print(trainset.shape)
#print(type(trainset))
#print(type(trainset[0, 0]), trainset[0,0])
#print(type(trainset[0, 1]), trainset[0,1])
#print(trainset[:1])
#aha = 0
#for bla, aaa in trainset:
#    #print(bla, aaa)
#    aha += aaa
#print("aha:", aha)
#print(len(trainset))

#print("soma apos normalizacao", sum(trainset[:, 1]))

###########################################################

#print("sample_weights[train_indices]: ", sample_weights[train_indices])
#print("sample_weights[train_indices].shape:", sample_weights[train_indices].shape)

sampler = WeightedRandomSampler(sample_weights[train_indices], num_samples=len(trainset), replacement=True)
data_loader = DataLoader(trainset, batch_size=32, collate_fn=collate, sampler=sampler)

#my_sum = 0
#for iter, (bg, label) in enumerate(data_loader):
#    #print(type(label))
#    my_sum += sum(label)
#print("my_sum", my_sum)

def write_file(filename, rows):
    #transform = transforms.Compose(
    #    [transforms.ToPILImage(),
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor()])
    with open("output/" + filename + ".csv", 'w') as output_file:
        for row in rows:
            #print(type(row))
            #print(row.shape)
            #new_row = row.reshape((int(row.shape[0] / 3), 3))
            #new_row = transform(new_row)
            #output_file.write(" ".join([str(a) for a in new_row.tolist()]) + '\n')
            output_file.write(" ".join([str(a) for a in row.tolist()]) + '\n')


def save_features(h_feats, label, dataset_type, sortpooling_k, epochs):
    #print("type(h_feats)", type(h_feats))
    #print("type(label)", type(label))

    print("[save_features] h_feats.shape", h_feats.shape)
    print("[save_features] label.shape", label.shape)

    vuln_features = []
    non_vuln_features = []

    non_vuln_indexes = (label == 0).nonzero(as_tuple=True)
    vuln_indexes = (label == 1).nonzero(as_tuple=True)
    print("non_vuln_indexes", non_vuln_indexes)
    print("vuln_indexes", vuln_indexes)
    print("h_feats.shape", h_feats.shape)
    non_vuln = h_feats[non_vuln_indexes]
    vuln = h_feats[vuln_indexes]
    #print(non_vuln)
    #print(non_vuln.shape)
    write_file("non-vuln-features-{}-k{}-ep{}".format(dataset_type, sortpooling_k, epochs), non_vuln)
    #print(h_feats[non_vuln_indexes[0]])

    #print("non_vuln[0]", non_vuln[0])
    #print(vuln)
    #print(vuln.shape)
    #print(h_feats[vuln_indexes[0]])
    #print("vuln[0]", vuln[0].tolist())
    write_file("vuln-features-{}-k{}-ep{}".format(dataset_type, sortpooling_k, epochs), vuln)


k_sortpooling = 6 #24 #16
for hidden_dimension in hidden_dimension_options:
    # %%
    # Create model
    #model = GraphClassifier(num_features, hidden_dimension, 2)
    if type(hidden_dimension) is list:
        model = GATGraphClassifier4HiddenLayers(num_features, hidden_dimension, 2, sortpooling_k=k_sortpooling)
    else:
        model = GATGraphClassifier(num_features, hidden_dimension, 2)

    #Class weighting
    #loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1,101000]))
    loss_func = nn.CrossEntropyLoss() # nn.NLLLoss() #nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # VERIFICAR COM O NUNO
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    # %%
    # Train the Model
    model.train()
    stats_dict = {
        'epoch': [],
        'epoch_losses' : [],
        'epoch_accuracy' : []
    }

    vuln_features = []
    non_vuln_features = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            #print(iter, epoch)
            # print(bg.ndata['features'].shape)
            #bg = dgl.add_self_loop(bg)
            prediction, h_concat, h_feats = model(bg)
            loss = loss_func(prediction, label)

            #print("prediction:", prediction)
            #print("h_cat:", h_cat.shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        # print(torch.argmax(prediction, dim = 1))
        accuracy = torch.mean((label == torch.argmax(prediction, dim = 1)).float())
        print('Epoch {}, loss {:.4f}, acc {:.4f}'.format(epoch, epoch_loss, accuracy))
        stats_dict['epoch'].append(epoch)
        stats_dict['epoch_losses'].append(epoch_loss)
        stats_dict['epoch_accuracy'].append(accuracy)

    print(trainset)

    artifact_suffix = "-{}-{}-{}n-{}-{}-sw{}-size1-{}-concat-k{}".format(project, version, hidden_dimension, normalization, num_epochs, sample_weight_value, type(model).__name__, k_sortpooling)

    if type(model).__name__ in ["GATGraphClassifier", "GATGraphClassifier4HiddenLayers"]:
        artifact_suffix += "-heads{}".format(heads)

    #artifact_suffix += "-closeness-centrality"

    df_stats = pd.DataFrame(stats_dict)
    df_stats.set_index('epoch', inplace=True)
    df_stats['epoch_accuracy'] = df_stats['epoch_accuracy'].astype(np.float64)
    sns.lineplot(data=df_stats)
    plt.savefig('stats/train-results{}.png'.format(artifact_suffix))
    # %%
    #Evaluate Model!
    model.eval()
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float()
    prediction, h_concat, h_feats = model(test_bg)
    save_features(h_feats, test_Y, "test", k_sortpooling, num_epochs)
    print(torch.argmax(prediction, dim = 1))
    print(test_Y)
    print(sum(test_Y))
    # accuracy = torch.mean((test_Y == torch.argmax(prediction, dim = 1)).float())
    # torch.argmax(prediction, dim = 1)
    params = test_Y.detach().numpy(), torch.argmax(prediction, dim = 1).float().detach().numpy()
    report = classification_report(*params, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('stats/classification_report{}.csv'.format(artifact_suffix))
    cm = confusion_matrix(*params, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.savefig('stats/confusion-matrix{}.png'.format(artifact_suffix))
    plt.clf()
    #plt.show()
    # print(accuracy)

    # Save Train Features
    train_X, train_Y = map(list, zip(*trainset))
    train_bg = dgl.batch(train_X)
    train_Y = torch.tensor(train_Y).float()
    prediction, h_concat, h_feats = model(train_bg)
    save_features(h_feats, train_Y, "train", k_sortpooling, num_epochs)

# %%
