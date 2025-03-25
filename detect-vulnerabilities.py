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

from detect_vulnerabilities_vgg import VGGnet

# %%

project = 'linux' # 'gecko-dev'#'linux'
version = 'v0.5'

#cfg-dataset-linux-v0.5 has 101513 entries

dataset_name = 'datasets/cfg-dataset-{}-{}'.format(project, version)
if not os.path.isfile(dataset_name + '.pkl'):
    df = load_dataset(dataset_name)
    df = df.to_pickle(sys.argv[1] + '.pkl')
else:
    df = pd.read_pickle(dataset_name + '.pkl')

# %%

ZNORM = "znorm"
MINMAX = "minmax"
normalization = MINMAX #ZNORM
DEBUG = False
SORTPOOLING = "sort_pooling"
ADAPTIVEMAXPOOLING = "adaptive_max_pooling"
UNDERSAMPLING_STRAT= 0.5
UNDERSAMPLING_METHOD = "random" #"kmeans" #None
pooling_type = ADAPTIVEMAXPOOLING #SORTPOOLING


heads = 4 # 2
num_features = 11 + 8 # 8 features related to memory management
num_epochs = 10 #2000 #500 # 1000
hidden_dimension_options = [[32, 32, 32, 32]] #[[128, 64, 32, 32], [32, 32, 32, 32]] #[32, 64, 128, [128, 64, 32, 32], [32, 32, 32, 32]] # [32, 64, 128] # [[128, 64, 32, 32], 32, 64, 128]
sample_weight_value = 10 #90 #100 #80 #60 # 40
CEL_weight = [1,10]
batch_size = 10
k_sortpooling = 6 #24 #16
dropout_rate = 0.1
conv2dChannelParam = 32



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
        h = self.sortpool(g, h)
        current_batch_size = h.shape[0]
        h = h.reshape(current_batch_size, self.hidden_dim, self.sortpooling_k)
        h = F.relu(self.conv1D(h))

        #########################################

        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        h = torch.squeeze(h)
        return self.classify(h) # era hg antes


class GATGraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, sortpooling_k=3):
        super(GATGraphClassifier, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads)  # allow_zero_in_degree=True
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, 1)
        self.classify = nn.Linear(hidden_dim, n_classes)
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

        bs = h.shape[0]
        h = F.relu(self.conv1(g, h))
        h = h.reshape(bs, -1)
        h = F.relu(self.conv2(g, h))
        h = h.reshape(bs, -1)
        h = self.drop(h)
        h = self.sortpool(g, h)

        current_batch_size = h.shape[0]

        h = h.reshape(current_batch_size, self.hidden_dim, self.sortpooling_k)
        h = F.relu(self.conv1D(h))
        h = torch.squeeze(h)

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

            h = torch.unsqueeze(h, 0)
            amp = nn.AdaptiveMaxPool2d((5,7))
            h = amp(h)

        return self.classify(h)


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

        self.classify = nn.Linear(hidden_dimensions[3], n_classes)

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
            
            # Katz centrality does not work (maybe related to eigen values and eigen vectors)
            #nx_g = dgl.to_networkx(g)
            #centrality = nx.degree_centrality(nx_g)
            #print(nx.eigenvector_centrality(nx_g))

            #print("degree_centrality", nx.degree_centrality(nx_g))
            ##print("katz_centrality", nx.katz_centrality(nx_g))
            #print("closeness_centrality", nx.closeness_centrality(nx_g))
            #centrality = nx.closeness_centrality(nx_g)
            #centrality = torch.FloatTensor(list(centrality.values()))
            h = g.ndata['features'].float()
            #teste_mul = h.T.mul(centrality).T
            #h = teste_mul

            #teste_mul = torch.dot(h, centrality)
            #print("torch.dot(h, centrality)", teste_mul.shape)

            bs = h.shape[0]  # bs is the number of nodes in the graph
            #print("bs", bs)

            if DEBUG:
                print(f"Graph with {bs} nodes and feature size {h.shape}")

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
        #return classification, h_concat, amp_layer, amp_layer
        return amp_layer

def apply_undersampling(df, strategy=None, method=None, n_clusters=None):
    """
    Apply undersampling to the entire dataset based on the 'label' column (which is in np.bool_ format).
    
    :param df: The dataset containing all columns including 'label'.
    :param strategy: The ratio of True (vulnerable) to False (non-vulnerable) labels (e.g., 0.1 for 10% True, 90% False).
    :return: The resampled DataFrame.
    """

    if method == None:
        if DEBUG:
            print("No undersampling applied. Returning the original dataset.")
        return df
    
    # Ensure the label column is in boolean format
    df['label'] = df['label'].astype(np.bool_)
    
    # Define X as the entire dataset except the label column
    X = df.drop(columns=['label'])
    y = df['label'].astype(int)  # Convert boolean labels to integers (True -> 1, False -> 0)
    
    # Count of the minority class (True -> 1)
    minority_count = sum(y)
    if DEBUG:
        print("minority_count:", minority_count)

    # Total number of samples we want in the final dataset
    total_samples = minority_count / strategy
    if DEBUG:
        print("total_samples:", total_samples)
    
    # Desired majority count (False -> 0)
    desired_majority_count = int(total_samples - minority_count)
    if DEBUG:
        print("desired_majority_count:", desired_majority_count)
    
    # Define the sampling strategy
    sampling_strategy = {0: desired_majority_count, 1: minority_count}
    
    if method == "random":
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif method == "kmeans":
        # Apply KMeans clustering-based undersampling
        # NOT WORKING - Not yet implemented and may not be necessary
        undersampler = ClusterCentroids(sampling_strategy=sampling_strategy, 
                                        estimator=KMeans(n_clusters=n_clusters, random_state=42))
    
    # Apply undersampling to the dataset
    X_res, y_res = undersampler.fit_resample(X, y)
    
    # Combine resampled X and y into a new DataFrame
    df_resampled = pd.concat([X_res, pd.Series(y_res.astype(np.int8), name='label')], axis=1)  # Convert labels back to boolean
    
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


# Applying undersampling
df_resampled = apply_undersampling(df, strategy=UNDERSAMPLING_STRAT, method=UNDERSAMPLING_METHOD)

true_count = df_resampled['label'].sum()  # Count of True labels (1)
false_count = len(df_resampled) - true_count  # Count of False labels (0)

ratio_true = true_count / len(df_resampled)
ratio_false = false_count / len(df_resampled)

if DEBUG:
    print("Original class distribution:")
    print(df['label'].value_counts(normalize=True))  # Check percentage distribution of True/False labels

    if UNDERSAMPLING_METHOD != None:
        # Check the class distribution after undersampling
        print("Class distribution after undersampling:")
        print(df_resampled['label'].value_counts(normalize=True))  # Should reflect the ratio

    print(f"Ratio of True: {ratio_true * 100:.2f}%")
    print(f"Ratio of False: {ratio_false * 100:.2f}%")

df = df_resampled

# %%
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


def save_features(h_feats, label, dataset_type, sortpooling_k, epochs):
    print("[save_features] h_feats.shape", h_feats.shape)
    print("[save_features] label.shape", label.shape)

    vuln_features = []
    non_vuln_features = []

    non_vuln_indexes = (label == 0).nonzero(as_tuple=True)
    vuln_indexes = (label == 1).nonzero(as_tuple=True)
    non_vuln = h_feats[non_vuln_indexes]
    vuln = h_feats[vuln_indexes]

    write_file("non-vuln-features-{}-k{}-ep{}-{}".format(dataset_type, sortpooling_k, epochs, version), non_vuln)
    write_file("vuln-features-{}-k{}-ep{}-{}".format(dataset_type, sortpooling_k, epochs, version), vuln)


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
    # Create model
    if type(hidden_dimension) is list:
        model = GATGraphClassifier4HiddenLayers(num_features, hidden_dimension, 2, sortpooling_k=k_sortpooling, conv2dChannel=conv2dChannelParam)
    else:
        model = GATGraphClassifier(num_features, hidden_dimension, 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model_vgg = VGGnet(in_channels=batch_size).to(device)
    model_vgg = VGGnet(in_channels=conv2dChannelParam).to(device)

    #Class weighting

    
    
    #loss_func = lambda pred, lbl: focal_loss(pred, lbl, alpha=0.25, gamma=2)
    if CEL_weight == 0:
        weight_values = [1, 1]
    weight = torch.tensor(CEL_weight , dtype=torch.float, device=device)
    loss_func = nn.CrossEntropyLoss(weight=weight)
    #loss_func = nn.CrossEntropyLoss() # nn.NLLLoss() #nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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

    # Create directories to store embeddings & predictions
    embedding_dir = "output/embeddings"
    prediction_dir = "output/predictions"
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_loss = 0

        all_dgcnn_embeddings = []
        all_predictions = []
        all_vgg_features = []
        all_labels = []

        for iter, (bg, label) in enumerate(data_loader):
            
            label = label.to(device) # Move label to the same device as the model and prediction

            if DEBUG:
                print("iter, epoch:", iter, epoch)
            #bg = dgl.add_self_loop(bg)
            h_cat_amp = model(bg).to(device)
            all_dgcnn_embeddings.append(h_cat_amp.cpu().detach())

            h_cat_amp = adjust_to_vgg(h_cat_amp).to(device)
            prediction = model_vgg(h_cat_amp)
            #print(f"prediction.shape (after VGG): {prediction.shape}")

            # VERIFICAR O HEATMAP DAS FEATURES
                # valores médios das classes positivas e das classes negativas
            # VEM A VGG

            loss = loss_func(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
 
            # Store predictions and labels
            all_predictions.extend(torch.argmax(prediction, dim=1).detach().cpu())
            all_labels.extend(label.cpu().numpy())
            all_vgg_features.append(prediction.cpu().detach())

        # Aggregate batch results into single tensors
        epoch_dgcnn_embeddings = torch.cat(all_dgcnn_embeddings, dim=0)
        epoch_vgg_features = torch.cat(all_vgg_features, dim=0) # Full logits
        epoch_vgg_predictions = torch.stack(all_predictions) # Class labels (0/1)
        epoch_labels = torch.tensor(all_labels)

        # Save results for the epoch
        torch.save(epoch_dgcnn_embeddings, f"{embedding_dir}/dgcnn_embeddings_epoch{epoch}.pt")
        torch.save(epoch_vgg_features, f"{prediction_dir}/vgg_features_epoch{epoch}.pt")  # Save full VGG embeddings
        torch.save(epoch_vgg_predictions, f"{prediction_dir}/vgg_predictions_epoch{epoch}.pt")
        torch.save(epoch_labels, f"{embedding_dir}/train_labels_epoch{epoch}.pt")

        epoch_loss /= (iter + 1)
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)
        accuracy = (all_predictions == all_labels).float().mean().item()
        print('Epoch {}, loss {:.4f}, acc {:.4f}'.format(epoch, epoch_loss, accuracy))
        stats_dict['epoch'].append(epoch)
        stats_dict['epoch_losses'].append(epoch_loss)
        stats_dict['epoch_accuracy'].append(accuracy)

    artifact_suffix = f"-{project}-{version}-{hidden_dimension}-n-{normalization}-e-{num_epochs}-us-{UNDERSAMPLING_STRAT}{UNDERSAMPLING_METHOD}-w-{CEL_weight[0]}_{CEL_weight[1]}"
    #artifact_suffix = f"-{project}-{version}-{hidden_dimension}n-{normalization}-e-{num_epochs}-w-{weight_values[0]}{weight_values[1]}"
    artifact_suffix += f"-sw{sample_weight_value}-size1-{type(model).__name__}-k{k_sortpooling}"
    artifact_suffix += f"-vgg-dr{dropout_rate}-c2d{conv2dChannelParam}"

    if type(model).__name__ in ["GATGraphClassifier", "GATGraphClassifier4HiddenLayers"]:
        artifact_suffix += "-heads{}".format(heads)

    stats_dict = {
        'epoch': [epoch.cpu().item() if torch.is_tensor(epoch) else epoch for epoch in stats_dict['epoch']],
        'epoch_losses': [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in stats_dict['epoch_losses']],
        'epoch_accuracy': [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in stats_dict['epoch_accuracy']]
    }

    df_stats = pd.DataFrame(stats_dict)
    df_stats.set_index('epoch', inplace=True)
    df_stats['epoch_accuracy'] = df_stats['epoch_accuracy'].astype(np.float64)
    sns.lineplot(data=df_stats)
    plt.savefig('stats/train-results{}.png'.format(artifact_suffix))
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

        if DEBUG:
            print("[Test Phase] iter:", iter)

        h_cat_amp = model(test_bg).to(device)

        #h_cat_amp = model(test_bg).to(device)
        h_cat_amp = adjust_to_vgg(h_cat_amp).to(device)#.detach()
        prediction_test = model_vgg(h_cat_amp).detach()

        test_label = torch.unsqueeze(test_label, 1).detach()

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
    prediction, test_Y = prediction.cpu().detach().numpy(), test_Y.cpu().detach().numpy()

    params = test_Y.squeeze().detach().numpy(), torch.argmax(prediction, dim = 1).float().detach().numpy()
    report = classification_report(*params, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('stats/classification_report{}.csv'.format(artifact_suffix))
    cm = confusion_matrix(*params, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.savefig('stats/confusion-matrix{}.png'.format(artifact_suffix))
    plt.clf()

    # Save Train Features
    # 11/07 train_X, train_Y = map(list, zip(*trainset))
    # 11/07 #train_bg = dgl.batch(train_X)
    # 11/07 train_bg = train_X
    # 11/07 train_Y = torch.tensor(train_Y).float()
    # 11/07 #prediction, h_concat, h_feats,
    # 11/07 h_cat_amp = model(train_bg)
    # 11/07 #save_features(h_feats, train_Y, "train", k_sortpooling, num_epochs)

# %%
