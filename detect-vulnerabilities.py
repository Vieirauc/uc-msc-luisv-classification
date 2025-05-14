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
from sklearn.cluster import KMeans
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from torchvision.ops import sigmoid_focal_loss
from torch_geometric.nn import GCNConv

from detect_vulnerabilities_vgg import VGGnet

# %%

project = 'linux' # 'gecko-dev'#'linux'
version = 'v0.5_filtered'
graph_type = 'cfg' #

#cfg-dataset-linux-v0.5 has 101513 entries
#cfg-dataset-linux-v0.5_filtered has 65685 entries

#dataset_name = "{}-dataset-{}-{}".format(graph_type, project, version)
dataset_name = 'cfg-dataset-linux-sample1k'
dataset_path = 'datasets/'

if not os.path.isfile(dataset_path + dataset_name + '.pkl'):
    df = load_dataset(dataset_path + dataset_name)
    df.to_pickle(dataset_path + dataset_name + '.pkl')
else:
    df = pd.read_pickle(dataset_path + dataset_name  + '.pkl')

# %%

ZNORM = "znorm"
MINMAX = "minmax"
normalization = MINMAX #ZNORM
DEBUG = True
SORTPOOLING = "sort_pooling"
ADAPTIVEMAXPOOLING = "adaptive_max_pooling"
UNDERSAMPLING_STRAT= 0.2
UNDERSAMPLING_METHOD = None # "random" "kmeans" #None
pooling_type = ADAPTIVEMAXPOOLING #SORTPOOLING

USE_AUTOENCODER = True
AUTOENCODER_EPOCHS = 20
NUM_NODES = 128  # padding fixo
FREEZE_ENCODER = True


heads = 4 # 2
num_features = 11 + 8 # 8 features related to memory management
num_epochs = 15 #2000 #500 # 1000
hidden_dimension_options = [[32, 32, 32, 32]] #[[128, 64, 32, 32], [32, 32, 32, 32]] #[32, 64, 128, [128, 64, 32, 32], [32, 32, 32, 32]] # [32, 64, 128] # [[128, 64, 32, 32], 32, 64, 128]
sample_weight_value = 0 #90 #100 #80 #60 # 40
CEL_weight = [1,1]
batch_size = 10
k_sortpooling = 6 #24 #16
dropout_rate = 0.1
conv2dChannelParam = 32



##################################################################################

class GATGraphClassifier4HiddenLayers(nn.Module):
    def __init__(self, in_dim, hidden_dimensions, n_classes, sortpooling_k=3, conv2dChannel=3): # conv2dChannel era 64
        super(GATGraphClassifier4HiddenLayers, self).__init__()

        self.sortpooling_k = sortpooling_k
        self.sortpool = SortPooling(k=sortpooling_k)

        self.conv1 = GATConv(in_dim, hidden_dimensions[0], heads)  # allow_zero_in_degree=True
        self.conv2 = GATConv(hidden_dimensions[0] * heads, hidden_dimensions[1], heads)
        self.conv3 = GATConv(hidden_dimensions[1] * heads, hidden_dimensions[2], heads)
        self.conv4 = GATConv(hidden_dimensions[2] * heads, hidden_dimensions[3], 1)

        # usado apenas com o SortPooling
        self.conv1D = nn.Conv1d(in_channels=hidden_dimensions[3], out_channels=hidden_dimensions[3], kernel_size=self.sortpooling_k, stride=1) # antes o kernal size era 3
        self.conv1D = nn.Conv1d(in_channels=sum(hidden_dimensions), out_channels=hidden_dimensions[3], kernel_size=self.sortpooling_k, stride=1)

        ###############################################################

        self.conv1 = GraphConv(in_dim, hidden_dimensions[0], allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dimensions[0], hidden_dimensions[1], allow_zero_in_degree=True)
        self.conv3 = GraphConv(hidden_dimensions[1], hidden_dimensions[2], allow_zero_in_degree=True)
        self.conv4 = GraphConv(hidden_dimensions[2], hidden_dimensions[3], allow_zero_in_degree=True)

        ###############################################################

        #self.classify = nn.Linear(hidden_dimensions[3], n_classes)

        # conv2dChannel 64, to be compatible with VGG11
        self.conv2dParam = nn.Conv2d(in_channels=1,
                                     out_channels=conv2dChannel,
                                     kernel_size=13, stride=1, padding=6)

        self.amp = nn.AdaptiveMaxPool2d((30, sum(hidden_dimensions))) # 05/07: no caso deles, a ultima dimensao tem tamanho 1
        self.drop = nn.Dropout(p = dropout_rate)


    def forward(self, graphs):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        amps = []

        for g in graphs:
            
            h = g.ndata['features'].float()

            bs = h.shape[0]  # bs is the number of nodes in the graph

            h1 = F.relu(self.conv1(g, h))
            #print("h1", h1.shape)
            h1 = h1.reshape(bs, -1)
            #print(f"h1.shape: {h1.shape}")
            h2 = F.relu(self.conv2(g, h1))
            #print("h2", h2.shape)
            h2 = h2.reshape(bs, -1)
            #print(f"h2.shape: {h2.shape}")
            h3 = F.relu(self.conv3(g, h2))
            #print("h3", h3.shape)
            h3 = h3.reshape(bs, -1)
            #print("h3", h3.shape)
            h4 = F.relu(self.conv4(g, h3))
            h4 = h4.reshape(bs, -1)

            h_cat = torch.cat((h1, h2, h3, h4), 1)
            #h_concat = h_cat
            h_cat = self.drop(h_cat)

            h4 = self.drop(h4)

            if pooling_type == SORTPOOLING:
                h_cat = self.sortpool(g, h_cat)
                current_batch_size = h_cat.shape[0]
                h_features = h_cat
                h_cat = h_cat.reshape(current_batch_size, int(h_cat.shape[1] / self.sortpooling_k), self.sortpooling_k)

                if type(self.conv1).__name__ == "GATConv":
                    h4 = self.conv1D(h4)
                    h_cat = F.relu(h4)
                else:
                    h_cat = F.relu(self.conv1D(h_cat))

                h_cat = torch.squeeze(h_cat)
            elif pooling_type == ADAPTIVEMAXPOOLING:

                toConv = torch.unsqueeze(h_cat, 0)
                toConv = toConv.unsqueeze(0)
                conved = self.conv2dParam(toConv)
                apGraphs = self.amp(conved)

            amps.append(apGraphs.squeeze())

        amp_layer = torch.stack(tuple(amps))

        return amp_layer # shape: (batch_size, 30, sum(hidden_dimensions))

class GraphDecoder(nn.Module):
    def __init__(self, embedding_dim, num_nodes, feature_dim):
        super(GraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        self.reconstruct_features = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * feature_dim),
            nn.Sigmoid()  # ou nn.Identity() se os features já estiverem normalizados
        )

    def forward(self, z):
        out = self.reconstruct_features(z)
        return out.view(-1, self.num_nodes, self.feature_dim)

def pad_graph(g, target_nodes, feature_dim):
    if g.num_nodes() != g.ndata['features'].shape[0]:
        raise ValueError(f"[pad_graph] Inconsistência: g.num_nodes() = {g.num_nodes()} "
                         f"mas g.ndata['features'].shape[0] = {g.ndata['features'].shape[0]}")

    if g.num_nodes() >= target_nodes:
        # Truncar se for maior
        g = dgl.node_subgraph(g, torch.arange(target_nodes))
        g.ndata['features'] = g.ndata['features'][:target_nodes]
        return g

    # Padding com zeros
    g.add_nodes(target_nodes - g.num_nodes())
    padded_features = torch.zeros(target_nodes, feature_dim, device=g.device)
    padded_features[:g.ndata['features'].shape[0]] = g.ndata['features']
    g.ndata['features'] = padded_features

    return g


def train_autoencoder(encoder, decoder, data_loader, device, num_nodes, feature_dim, num_epochs=20):
    encoder.train()
    decoder.train()

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    loss_func = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0

        for graphs, _ in data_loader:
            padded_X = []
            for g in graphs:
                #print(f"[DEBUG] Antes do padding: g.num_nodes = {g.num_nodes()}, features = {g.ndata['features'].shape}")
                g = pad_graph(g, num_nodes,feature_dim)  # garante que todos têm o mesmo número de nós
                padded_X.append(g.ndata['features'])

            X_orig = torch.stack(padded_X).to(device)
            X_orig = X_orig.to(device)


            graphs = [g.to(device) for g in graphs]  # move cada grafo individualmente
            z = encoder(graphs)                      # output já estará no device   # (B, 30, D)
            z = z.view(z.size(0), -1)        # (B, 30*D)

            X_rec = decoder(z)               # (B, N, F)

            loss = loss_func(X_rec, X_orig)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"[Autoencoder] Epoch {epoch}, Loss: {total_loss:.4f}")



def apply_undersampling(df, strategy=0.5, method="random", n_clusters=None):
    """
    Apply undersampling to the dataset based on the 'label' column.
    
    :param df: DataFrame with a boolean or binary 'label' column (1 = vulnerable, 0 = non-vulnerable).
    :param strategy: Desired ratio of positives to the total dataset (e.g., 0.2 = 20% vulnerable).
    :param method: 'random' or 'kmeans' undersampling.
    :param n_clusters: Used if method is 'kmeans'.
    :return: Resampled DataFrame.
    """
    if method is None:
        if DEBUG:
            print("No undersampling applied.")
        return df

    df['label'] = df['label'].astype(np.bool_)  # Ensures correct type
    X = df.drop(columns=['label'])
    y = df['label'].astype(int)

    minority_count = y.sum()
    total_desired = round(minority_count / strategy)
    desired_majority_count = total_desired - minority_count

    if DEBUG:
        print(f"Minority count: {minority_count}")
        print(f"Desired total: {total_desired}")
        print(f"Desired majority: {desired_majority_count}")

    # Enforce a max cap to prevent over-removal in small datasets
    actual_majority_count = sum(y == 0)
    desired_majority_count = min(desired_majority_count, actual_majority_count)

    sampling_strategy = {0: desired_majority_count, 1: minority_count}

    if method == "random":
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif method == "kmeans":
        # NOT WORKING - Not yet implemented and may not be necessary
        undersampler = ClusterCentroids(sampling_strategy=sampling_strategy, 
                                        estimator=KMeans(n_clusters=n_clusters or 10, random_state=42))
    else:
        raise ValueError(f"Unsupported method: {method}")

    X_res, y_res = undersampler.fit_resample(X, y)
    df_resampled = pd.concat([X_res, pd.Series(y_res, name='label')], axis=1)

    return df_resampled


# %%
# Load and Process dataset
# %%

df['label'] = torch.tensor(df['label'].astype(np.int8))
#print("sum(df[label]): ", sum(df['label']))
df['sample_weight'] = torch.tensor(1 + (df['label'].astype(np.int8) * sample_weight_value))
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    #batched_graph = dgl.batch(graphs)
    #return batched_graph, torch.tensor(labels)
    return graphs, torch.tensor(labels)

print("Number of samples in the dataset before undersampling:", len(df))
# Applying undersampling
df_resampled = apply_undersampling(df, strategy=UNDERSAMPLING_STRAT, method=UNDERSAMPLING_METHOD)

true_count = df_resampled['label'].sum()  # Count of True labels (1)
false_count = len(df_resampled) - true_count  # Count of False labels (0)

ratio_true = true_count / len(df_resampled)
ratio_false = false_count / len(df_resampled)

graph_sizes = df['size'].values
max_size = np.max(graph_sizes)
percentile_95 = int(np.percentile(graph_sizes, 95))

print("Max size:", max_size)
print("95th percentile:", percentile_95)

if DEBUG:
    print("Original class distribution:")
    print(df['label'].value_counts(normalize=True))  # Check percentage distribution of True/False labels

    if UNDERSAMPLING_METHOD != None:
        # Check the class distribution after undersampling
        print("Class distribution after undersampling:")
        print(df_resampled['label'].value_counts(normalize=True))  # Should reflect the ratio

    print(f"Ratio of True: {ratio_true * 100:.2f}%")
    print(f"Ratio of False: {ratio_false * 100:.2f}%")
    print("Number of samples in the dataset after undersampling:", len(df_resampled))
    #print number of true and false labels
    
    print("Number of true labels:", true_count)
    print("Number of false labels:", false_count)
    

df = df_resampled

# %%
sample_weights = df['sample_weight'].values
# %%
random_seed = 42
test_split = 0.3
# Creating data indices for training and validation splits:
dataset_size = len(df)
print("dataset_size:", dataset_size)
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
print("train & test data shape:",all_feature_train_data.shape, all_feature_test_data.shape)

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



def normalize_minmax(dataset, feat_minimum, feat_maximum):
    denominator = feat_maximum - feat_minimum
    # Corrige casos onde max == min (sem variação)
    denominator[denominator == 0] = 1.0
    for i in range(len(dataset)):
        dataset[i, 0].ndata['features'] = torch.div(
            dataset[i, 0].ndata['features'] - feat_minimum,
            denominator
        )
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
        if num_features > 11:
            # memory management features are also available
            t = torch.cat((t[:,0:3], t[:,4:15], t[:,16:18]), 1)
        else:
            t = torch.cat((t[:,0:3], t[:,4:]), 1)
        dataset[i, 0].ndata['features'] = t
    return dataset

#  Removes one feature as it is always zero (no node was assigned to type "numeric constant")
#print(trainset.shape)
if normalization is not None or normalization == "":
    trainset = adjust_dataset(trainset)
    testset = adjust_dataset(testset)
    if num_features > 11:
        # memory management features are also available
        num_features -= 3
    else:
        num_features -= 1
print("len(trainset):", len(trainset))

###########################################################


sampler = WeightedRandomSampler(sample_weights[train_indices], num_samples=len(trainset), replacement=True)
data_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, sampler=sampler)


def write_file(filename, rows):
    with open("output/" + filename + ".csv", 'w') as output_file:
        for row in rows:
            output_file.write(" ".join([str(a) for a in row.tolist()]) + '\n')


def save_embeddings(model, model_vgg, dataset, device, embedding_dir, prediction_dir, prefix, batch_size=10):

    model.eval()
    model_vgg.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    all_dgcnn_embeddings = []
    all_vgg_features = []
    all_predictions = []
    all_labels = []

    for iter, (bg, label) in enumerate(data_loader):
        bg = [g.to(device) for g in bg]
        label = label.to(device)

        h_cat_amp = model(bg)
        all_dgcnn_embeddings.append(h_cat_amp.cpu().detach())

        h_cat_amp = adjust_to_vgg(h_cat_amp).to(device)
        prediction = model_vgg(h_cat_amp).detach()

        
        label = torch.unsqueeze(label, 1).detach()

        all_vgg_features.append(prediction.cpu())
        all_predictions.extend(torch.argmax(prediction, dim=1).cpu())
        all_labels.extend(label.cpu())

    dgcnn_embeddings = torch.cat(all_dgcnn_embeddings, dim=0)
    vgg_features = torch.cat(all_vgg_features, dim=0)
    vgg_predictions = torch.tensor(all_predictions)
    labels = torch.tensor(all_labels)

    torch.save(dgcnn_embeddings, os.path.join(embedding_dir, f"dgcnn_embeddings_{prefix}.pt"))
    torch.save(vgg_features, os.path.join(prediction_dir, f"vgg_features_{prefix}.pt"))
    torch.save(vgg_predictions, os.path.join(prediction_dir, f"vgg_predictions_{prefix}.pt"))
    torch.save(labels, os.path.join(embedding_dir, f"{prefix}_labels.pt"))
    print(f"[save_embeddings] Saved {prefix} embeddings and predictions.")

def adjust_to_vgg(samples):
    padding_size = int((224 - samples.shape[3])/2)
    if (samples.shape[3] % 2) != 0:
        # odd number
        padding_size_right = padding_size + 1
    else:
        padding_size_right = padding_size

    padding_size_dim2 = int((224 - samples.shape[2])/2)
    if (samples.shape[2] % 2) != 0:
        # odd number
        padding_size_right_dim2 = padding_size_dim2 + 1
    else:
        padding_size_right_dim2 = padding_size_dim2

    x1 = F.pad(samples, (padding_size, padding_size_right, padding_size_dim2, padding_size_right_dim2))

    return x1

# 
def focal_loss(pred, lbl, alpha=0.25, gamma=2.0):
    # Convert labels to one-hot encoding
    one_hot_lbl = F.one_hot(lbl, num_classes=2).float()
    # Compute focal loss with softmax probabilities
    return sigmoid_focal_loss(pred, one_hot_lbl, alpha=alpha, gamma=gamma, reduction="mean")




for hidden_dimension in hidden_dimension_options:
    # %%
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create model
    
    model = GATGraphClassifier4HiddenLayers(num_features, hidden_dimension, 2, sortpooling_k=k_sortpooling, conv2dChannel=conv2dChannelParam)
    model_vgg = VGGnet(in_channels=conv2dChannelParam).to(device)

    if USE_AUTOENCODER:
        encoder = model.to(device)
        decoder = GraphDecoder(embedding_dim=30 * sum(hidden_dimension) * conv2dChannelParam , num_nodes=NUM_NODES, feature_dim=num_features).to(device)

        print(f"[INFO] Starting autoencoder pretraining for {AUTOENCODER_EPOCHS} epochs...")
        train_autoencoder(encoder, decoder, data_loader, device, num_nodes=NUM_NODES, feature_dim=num_features, num_epochs=AUTOENCODER_EPOCHS)
        print(f"[INFO] Autoencoder training completed.\n")
    
    if FREEZE_ENCODER: # TRUE para usar as embeddings aprendidas, sem as alterar, e apenas treinar a VGG para aprender a classificar
        for param in model.parameters():
            param.requires_grad = False
    
    # optimizer apenas para model_vgg se FREEZE_ENCODER == True
    optimizer = optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(model_vgg.parameters()), lr=0.001
    )

    #print("[DEBUG] Parâmetros que vão ser atualizados:")
    #for name, param in list(model.named_parameters()) + list(model_vgg.named_parameters()):
    #    if param.requires_grad:
    #        print(f"  {name} - requires_grad = True")

    #loss_func = lambda pred, lbl: focal_loss(pred, lbl, alpha=0.25, gamma=2)
    if CEL_weight == 0:
        weight_values = [1, 1]

    weight = torch.tensor(CEL_weight , dtype=torch.float, device=device)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    # %%
    # Train the Model
    model.train()
    stats_dict = {
        'epoch': [],
        'epoch_losses': [],
        'epoch_accuracy': []
    }

    vuln_features = []
    non_vuln_features = []

    # Create directories to store results
    def format_hidden_dim(hd):
        return '-'.join(map(str, hd)) if isinstance(hd, list) else str(hd)

    artifact_suffix = f"{dataset_name}_hd-{format_hidden_dim(hidden_dimension)}_norm-{normalization}_e{num_epochs}_us-{UNDERSAMPLING_STRAT}{UNDERSAMPLING_METHOD or 'none'}"
    artifact_suffix += f"_w{CEL_weight[0]}-{CEL_weight[1]}_sw{sample_weight_value}_model-{type(model).__name__}_k{k_sortpooling}"
    artifact_suffix += f"_vgg-drop{dropout_rate}_c2d{conv2dChannelParam}"

    if type(model).__name__ in ["GATGraphClassifier", "GATGraphClassifier4HiddenLayers"]:
        artifact_suffix += f"_heads{heads}"

    output_base_dir = "output/runs"
    run_output_dir = os.path.join(output_base_dir, artifact_suffix)
    embedding_dir = os.path.join(run_output_dir, "embeddings")
    prediction_dir = os.path.join(run_output_dir, "predictions")
    stats_dir = os.path.join(run_output_dir, "stats")

    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    print("========= Beginning of Training Phase ===========")

    for epoch in range(num_epochs):
        epoch_loss = 0

        all_predictions = []
        all_labels = []

        for iter, (bg, label) in enumerate(data_loader):
            
            bg = [g.to(device) for g in bg]
            label = label.to(device)

            h_cat_amp = model(bg)      

            h_cat_amp = adjust_to_vgg(h_cat_amp).to(device)
            prediction = model_vgg(h_cat_amp)

            loss = loss_func(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
 
            # Store predictions and labels
            all_predictions.extend(torch.argmax(prediction, dim=1).detach().cpu())
            all_labels.extend(label.cpu().numpy())


        epoch_labels = torch.tensor(all_labels)

        epoch_loss /= (iter + 1)
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)
        accuracy = (all_predictions == all_labels).float().mean().item()
        print('Epoch {}, loss {:.4f}, acc {:.4f}'.format(epoch, epoch_loss, accuracy))
        stats_dict['epoch'].append(epoch)
        stats_dict['epoch_losses'].append(epoch_loss)
        stats_dict['epoch_accuracy'].append(accuracy)

    print("========= End of Training Phase ===========")

    stats_dict = {
        'epoch': [epoch.cpu().item() if torch.is_tensor(epoch) else epoch for epoch in stats_dict['epoch']],
        'epoch_losses': [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in stats_dict['epoch_losses']],
        'epoch_accuracy': [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in stats_dict['epoch_accuracy']]
    }

    df_stats = pd.DataFrame(stats_dict)
    df_stats.set_index('epoch', inplace=True)
    df_stats['epoch_accuracy'] = df_stats['epoch_accuracy'].astype(np.float64)
    sns.lineplot(data=df_stats)
    plt.savefig(os.path.join(stats_dir, f"train-results_epoch{num_epochs}.png"))

    # %%
    #Evaluate Model!
    model.eval()

    prediction = None
    test_Y = None
    prediction_list = []
    test_Y_list = []

    data_loader_test = DataLoader(testset, batch_size=batch_size, collate_fn=collate)

    print("========= Beginning of Test Phase ===========")
    for iter, (test_bg, test_label) in enumerate(data_loader_test):

        test_bg = [g.to(device) for g in test_bg]

        h_cat_amp = model(test_bg).to(device)

        # Prepare for VGG and predict
        h_cat_amp = adjust_to_vgg(h_cat_amp).to(device)#.detach()
        prediction_test = model_vgg(h_cat_amp).detach()

        test_label = test_label.long().squeeze().detach()

        prediction_list.append(prediction_test)
        test_Y_list.append(test_label)


        if iter % 10 == 0:
            if prediction is None:
                prediction = torch.cat(tuple(prediction_list))
                test_Y = torch.cat(tuple(test_Y_list))
            else:
                prediction_aux = torch.cat(tuple(prediction_list))
                prediction = torch.cat((prediction, prediction_aux))
                test_Y_aux = torch.cat(tuple(test_Y_list))
                test_Y = torch.cat((test_Y, test_Y_aux))
            prediction_list = []
            test_Y_list = []

    if len(prediction_list):
        prediction_aux = torch.cat(tuple(prediction_list))
        prediction = torch.cat((prediction, prediction_aux))
        test_Y_aux = torch.cat(tuple(test_Y_list))
        test_Y = torch.cat((test_Y, test_Y_aux))

    print("========= End of Test Phase ===========")


    print(f"prediction.shape: {prediction.shape}")
    print(f"test_Y.shape: {test_Y.shape}")

    #TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

    params = test_Y.squeeze().cpu().detach().numpy(), torch.argmax(prediction, dim = 1).float().cpu().detach().numpy()

    report = classification_report(*params, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(stats_dir, f"classification_report.csv"))
    cm = confusion_matrix(*params, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.savefig(os.path.join(stats_dir, f"confusion-matrix.png"))
    plt.clf()

    save_embeddings(model, model_vgg, testset, device, embedding_dir, prediction_dir, prefix="test", batch_size=batch_size)

# %%
