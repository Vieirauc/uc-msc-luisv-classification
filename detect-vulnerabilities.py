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
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
# %%
dataset_name = 'datasets/cfg-dataset-linux'
if not os.path.isfile(dataset_name + '.pkl'):
    df = load_dataset(dataset_name)
    df = df.to_pickle(sys.argv[1] + '.pkl')
else:
    df = pd.read_pickle(dataset_name + '.pkl')


# %%

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GraphClassifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.classify = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().reshape(-1, 1).float()
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

# Load and Process dataset
# %%
df['label'] = torch.tensor(df['label'].astype(np.int8))
df['sample_weight'] = torch.tensor(1 + (df['label'].astype(np.int8) * 40))
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

sampler = WeightedRandomSampler(sample_weights[train_indices], num_samples=len(trainset), replacement=True)
data_loader = DataLoader(trainset, batch_size=32, collate_fn=collate, sampler=sampler)

# %%
# Create model
model = GraphClassifier(1, 256, 2)
#Class weighting
#loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1,101000]))
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Train the Model
model.train()
epoch_losses = []
epoch_accuracy = []
for epoch in range(80):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        # print(iter)
        # print(bg)
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
    epoch_losses.append(epoch_loss)
    epoch_accuracy.append(accuracy)
# %%
#Evaluate Model!
model.eval()
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float()
prediction = model(test_bg)
print(torch.argmax(prediction, dim = 1))
print(test_Y)
accuracy = torch.mean((test_Y == torch.argmax(prediction, dim = 1)).float())
print(accuracy)

# %%
