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
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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
normalization = MINMAX # MINMAX #ZNORM

heads = 2
num_features = 11
num_epochs = 500
hidden_dimension_options = [32, 64, 128]
sample_weight_value = 40 #60 # 40


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


# Load and Process dataset
# %%
df['label'] = torch.tensor(df['label'].astype(np.int8))
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
if normalization is not None or normalization == "":
    trainset = adjust_dataset(trainset)
    testset = adjust_dataset(testset)
    num_features -= 1


###########################################################

sampler = WeightedRandomSampler(sample_weights[train_indices], num_samples=len(trainset), replacement=True)
data_loader = DataLoader(trainset, batch_size=32, collate_fn=collate, sampler=sampler)


for hidden_dimension in hidden_dimension_options:
    # %%
    # Create model
    model = GraphClassifier(num_features, hidden_dimension, 2)
    #model = GATGraphClassifier(num_features, hidden_dimension, 2)

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

    for epoch in range(num_epochs):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            # print(iter)
            # print(bg.ndata['features'].shape)
            #bg = dgl.add_self_loop(bg)
            prediction = model(bg)
            loss = loss_func(prediction, label)
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


    artifact_suffix = "-{}-{}-{}n-{}-{}-sw{}-size1-{}".format(project, version, hidden_dimension, normalization, num_epochs, sample_weight_value, type(model).__name__)

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
    prediction = model(test_bg)
    print(torch.argmax(prediction, dim = 1))
    print(test_Y)
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

# %%
