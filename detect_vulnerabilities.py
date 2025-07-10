# %%
import numpy as np
import pandas as pd
import os
import time
from load_datasets import load_dataset
import sys
import dgl
import torch
import json
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
version = None # 'v0.5_filtered'
graph_type = 'cfg' #

#cfg-dataset-linux-v0.5 has 101513 entries
#cfg-dataset-linux-v0.5_filtered has 65685 entries

#if version:
#    dataset_name = f"{graph_type}-dataset-{project}-{version}"
#else:
#    dataset_name = f"{graph_type}-dataset-{project}"

dataset_name = 'cfg-dataset-linux-v0.5_filtered'
#dataset_name = 'cfg-dataset-linux-sample1k'
dataset_path = 'datasets/'

if not os.path.isfile(dataset_path + dataset_name + '.pkl'):
    df = load_dataset(dataset_path + dataset_name)
    df.to_pickle(dataset_path + dataset_name + '.pkl')
else:
    df = pd.read_pickle(dataset_path + dataset_name  + '.pkl')

# %%

DEBUG = False

ZNORM = "znorm"
MINMAX = "minmax"
SORTPOOLING = "sort_pooling"
ADAPTIVEMAXPOOLING = "adaptive_max_pooling"

normalization = MINMAX #ZNORM
#pooling_type = ADAPTIVEMAXPOOLING #SORTPOOLING

UNDERSAMPLING_STRAT= 0.2
UNDERSAMPLING_METHOD = "random" # "random" "kmeans" #None

USE_AUTOENCODER = True
NUM_NODES = 55  # padding fixo
FREEZE_ENCODER = True
learning_rate_ae = 0.001 #0.0001 #0.00001 #0.000001
AUTOENCODER_EPOCHS = 10

classifier_type = "conv1d"  # ou "vgg" ou "conv1d"


heads = 4 # 2
hidden_dimension = [32, 32, 32, 32] #[[128, 64, 32, 32], [32, 32, 32, 32]] #[32, 64, 128, [128, 64, 32, 32], [32, 32, 32, 32]] # [32, 64, 128] # [[128, 64, 32, 32], 32, 64, 128]
sample_weight_value = 0 #90 #100 #80 #60 # 40
CEL_weight = [1,4]
batch_size = 10
k_sortpooling = 16 #24 #16
dropout_rate = 0.3 #0.1 
conv2dChannelParam = 32
learning_rate = 0.0005 #0.0001 #0.00001 #0.000001 #0.001 #0.01 #0.1 #0.0005
num_epochs = 10 #2000 #500 # 1000

if graph_type == 'cfg':
    num_features = 19  # 11 base + 8 memory
elif graph_type == 'ast':
    num_features = 6
elif graph_type == 'pdg':
    num_features = 4
else:
    raise ValueError(f"Unsupported graph_type: {graph_type}")


##################################################################################
class DGCNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dimensions, sortpooling_k=None, use_sortpool=True):
        super(DGCNNEncoder, self).__init__()

        self.use_sortpool = use_sortpool

        self.conv1 = GraphConv(in_dim, hidden_dimensions[0], allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dimensions[0], hidden_dimensions[1], allow_zero_in_degree=True)
        self.conv3 = GraphConv(hidden_dimensions[1], hidden_dimensions[2], allow_zero_in_degree=True)
        self.conv4 = GraphConv(hidden_dimensions[2], hidden_dimensions[3], allow_zero_in_degree=True)

        #self.conv1 = GATConv(in_dim, hidden_dimensions[0], heads)  # allow_zero_in_degree=True
        #self.conv2 = GATConv(hidden_dimensions[0] * heads, hidden_dimensions[1], heads)
        #self.conv3 = GATConv(hidden_dimensions[1] * heads, hidden_dimensions[2], heads)
        #self.conv4 = GATConv(hidden_dimensions[2] * heads, hidden_dimensions[3], 1)

        self.sortpool = SortPooling(k=sortpooling_k)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.amp_pool = nn.AdaptiveMaxPool1d(sortpooling_k)

    def forward(self, graphs, return_node_embeddings=False, use_amp_pooling=False):
        batch_node_features = []

        for g in graphs:
            g = dgl.add_self_loop(g)
            h = g.ndata['features'].float()
            h = F.relu(self.conv1(g, h))
            h = F.relu(self.conv2(g, h))
            h = F.relu(self.conv3(g, h))
            h = F.relu(self.conv4(g, h))
            batch_node_features.append(h)

        batched_graph = dgl.batch(graphs)
        h_all = torch.cat(batch_node_features, dim=0)

        if return_node_embeddings:
            return h_all, batched_graph

        if use_amp_pooling:
            batch_sizes = batched_graph.batch_num_nodes()
            split_feats = torch.split(h_all, batch_sizes.tolist())
            
            # Aplica AMP pooling para garantir dimensões fixas
            amp_pooled = torch.stack([self.amp_pool(feat.T).flatten() for feat in split_feats], dim=0)
            embeddings = self.dropout(amp_pooled)
            return embeddings  # (B, k * hidden_dim[-1])

        if self.use_sortpool:
            h_pooled = self.sortpool(batched_graph, h_all)
            embeddings = self.dropout(h_pooled)
            return embeddings

        batched_graph.ndata['h'] = h_all
        return batched_graph


class DGCNNVGGAdapter(nn.Module):
    def __init__(self, embedding_dim, conv2d_channels=32):
        super(DGCNNVGGAdapter, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=conv2d_channels,
            kernel_size=13, stride=1, padding=6
        )
        self.amp = nn.AdaptiveMaxPool2d((30, embedding_dim))

    def forward(self, batched_graph):
        """
        Expects a batched graph where each graph has node features stored in 'h'.
        This version reproduces the AMP pathway: conv2d over a pseudo-image of node x embedding.
        """
        graph_feats = []
        batch_num_nodes = batched_graph.batch_num_nodes().tolist()

        # Get node features
        h_all = batched_graph.ndata['h']  # (total_nodes, hidden_dim)
        start = 0

        for num_nodes in batch_num_nodes:
            end = start + num_nodes
            h_graph = h_all[start:end]  # (num_nodes, hidden_dim)

            h_graph = h_graph.unsqueeze(0).unsqueeze(0)  # (1, 1, N, F)
            conv_out = self.conv2d(h_graph)              # (1, C, H, W)
            amp_out = self.amp(conv_out).squeeze(0)      # (C, 30, F)
            graph_feats.append(amp_out)

            start = end

        return torch.stack(graph_feats)  # (B, C, 30, F)

class DGCNNConv1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(DGCNNConv1DClassifier, self).__init__()

        # Compute L2
        L2 = input_dim - 4 - 4  # kernel_size=5 twice, stride=1

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)

        self.fc = nn.Sequential(
            nn.Flatten(),  # (B, 32, L2) → (B, 32*L2)
            nn.Linear(32 * L2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)         # (B, 1, D)
        x = F.relu(self.conv1(x))  # (B, 16, L1)
        x = F.relu(self.conv2(x))  # (B, 32, L2)
        return self.fc(x)


class GraphDecoder(nn.Module):
    def __init__(self, embedding_dim, num_nodes, feature_dim):
        super(GraphDecoder, self).__init__()
        self.num_nodes = num_nodes  # ← should be `k` from SortPooling
        self.feature_dim = feature_dim

        self.reconstruct_features = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes * feature_dim),
            nn.Sigmoid()  # Identity if not normalized
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


def train_autoencoder(encoder, decoder, data_loader, device, sortpooling_k, feature_dim, num_epochs=20, stats_dir="stats", use_amp_pooling=False):
    encoder.train()
    decoder.train()

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate_ae)
    loss_func = nn.BCELoss()

    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for graphs, _ in data_loader:
            graphs = [g.to(device) for g in graphs]

            # Usa AMP ou SortPooling dependendo do classifier
            z = encoder(graphs, use_amp_pooling=use_amp_pooling)

            # Criar ground truth com os k primeiros nós reais de cada grafo (sem padding!)
            X_orig_list = []
            for g in graphs:
                node_feats = g.ndata['features'].float()
                if node_feats.size(0) >= sortpooling_k:
                    selected = node_feats[:sortpooling_k]
                else:
                    pad = torch.zeros(sortpooling_k - node_feats.size(0), feature_dim, device=node_feats.device)
                    selected = torch.cat([node_feats, pad], dim=0)
                X_orig_list.append(selected)

            X_orig = torch.stack(X_orig_list).to(device)  # (B, k, F)

            # Reconstrução
            X_rec = decoder(z)  # (B, k, F)
            loss = loss_func(X_rec, X_orig)

            # Otimização
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"[Autoencoder] Epoch {epoch}, Loss: {avg_loss:.4f}")

    # Plot e CSV
    plt.figure()
    plt.plot(range(num_epochs), epoch_losses, marker='o', color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (BCE)")
    plt.title("Autoencoder Training Loss Curve")
    plt.grid(True)

    os.makedirs(stats_dir, exist_ok=True)
    loss_plot_path = os.path.join(stats_dir, "autoencoder_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"[Autoencoder] Loss plot saved to {loss_plot_path}")

    loss_df = pd.DataFrame({"epoch": list(range(num_epochs)), "loss": epoch_losses})
    csv_path = os.path.join(stats_dir, "autoencoder_loss.csv")
    loss_df.to_csv(csv_path, index=False)
    print(f"[Autoencoder] Loss history saved to {csv_path}")



def apply_undersampling(df, strategy=0.5, method="random", n_clusters=None):
    if method is None:
        if DEBUG:
            print("No undersampling applied.")
        return df

    X = df.drop(columns=['label'])
    y = df['label'].astype(int)

    minority_count = y.sum()
    desired_majority_count = min(
        round((minority_count / strategy) - minority_count),
        sum(y == 0)
    )

    sampler = {
        "random": RandomUnderSampler,
        "kmeans": lambda **kwargs: ClusterCentroids(
            estimator=KMeans(n_clusters=n_clusters or 10, random_state=42), **kwargs)
    }.get(method)

    if not sampler:
        raise ValueError(f"Unsupported method: {method}")

    undersampler = sampler(sampling_strategy={0: desired_majority_count, 1: minority_count}, random_state=42)
    X_res, y_res = undersampler.fit_resample(X, y)

    return pd.concat([X_res, pd.Series(y_res, name='label')], axis=1)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return graphs, torch.tensor(labels)

def write_file(filename, rows):
    with open("output/" + filename + ".csv", 'w') as output_file:
        for row in rows:
            output_file.write(" ".join([str(a) for a in row.tolist()]) + '\n')

def save_embeddings(encoder, dataset, device, embedding_dir, prefix, epoch=None,
                    batch_size=10, classifier_type=None, vgg_adapter=None, classifier_model=None):
    """
    Salva embeddings da DGCNN + previsões do classificador (se houver) + labels.
    Suporta: vgg, conv1d, ou apenas extração de embeddings.

    Args:
        encoder: modelo DGCNNEncoder
        dataset: dataset de teste ou validação
        device: cuda ou cpu
        embedding_dir: diretório de saída
        prefix: prefixo nos ficheiros
        epoch: número do epoch (opcional)
        batch_size: tamanho do batch
        classifier_type: "vgg", "conv1d" ou None
        vgg_adapter: adaptador (obrigatório se classifier_type == "vgg")
        classifier_model: classificador VGG ou Conv1D
    """
    encoder.eval()
    if classifier_model:
        classifier_model.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    all_embeddings, all_labels, all_predictions = [], [], []

    with torch.no_grad():
        for graphs, labels in data_loader:
            graphs = [g.to(device) for g in graphs]
            labels = labels.to(device)
            all_labels.append(labels.cpu())

            embeddings = encoder(graphs)

            if classifier_type == "vgg":
                assert vgg_adapter is not None, "[save_embeddings] VGG adapter required."
                batched_graph = embeddings  # encoder returned a DGLGraph
                embeddings_vgg = vgg_adapter(batched_graph)
                embeddings_vgg = adjust_to_vgg(embeddings_vgg)  # (B, C*H*W)
                all_embeddings.append(embeddings_vgg.cpu())

                if classifier_model:
                    preds = classifier_model(embeddings_vgg.to(device)).argmax(dim=1)
                    all_predictions.append(preds.cpu())

            elif classifier_type == "conv1d":
                all_embeddings.append(embeddings.cpu())

                if classifier_model:
                    preds = classifier_model(embeddings.to(device)).argmax(dim=1)
                    all_predictions.append(preds.cpu())

            else:
                # Default mode: use global mean pooling
                if isinstance(embeddings, dgl.DGLGraph):
                    node_feats = embeddings.ndata['h']
                    batch_sizes = embeddings.batch_num_nodes()
                    split_feats = torch.split(node_feats, batch_sizes.tolist())
                    pooled = torch.stack([f.mean(dim=0) for f in split_feats], dim=0)
                    all_embeddings.append(pooled.cpu())
                else:
                    all_embeddings.append(embeddings.cpu())

    # Concatenação
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    suffix = f"{prefix}" + (f"_epoch{epoch}" if epoch is not None else "")

    # Salvar
    torch.save(embeddings_tensor, os.path.join(embedding_dir, f"dgcnn_embeddings_{suffix}.pt"))
    torch.save(labels_tensor, os.path.join(embedding_dir, f"{suffix}_labels.pt"))
    print(f"[save_embeddings] Embeddings shape: {embeddings_tensor.shape}")
    print(f"[save_embeddings] Labels saved.")
    print(f"[save_embeddings] Classifier type: {classifier_type}")

    if all_predictions:
        predictions_tensor = torch.cat(all_predictions, dim=0)
        pred_filename = f"{classifier_type}_predictions_{suffix}.pt"
        torch.save(predictions_tensor, os.path.join(embedding_dir, pred_filename))
        print(f"[save_embeddings] {classifier_type.upper()} predictions saved.")

    print("[save_embeddings] Done.")




def log_hyperparameters(run_output_dir, params_dict):
    """
    Salva os hiperparâmetros usados na run em CSV, JSON e LaTeX
    dentro de um subdiretório `hyperparameters/`.
    """
    hyper_dir = os.path.join(run_output_dir, "hyperparameters")
    os.makedirs(hyper_dir, exist_ok=True)
    
    # CSV
    csv_path = os.path.join(hyper_dir, "hyperparameters.csv")
    pd.DataFrame([params_dict]).to_csv(csv_path, index=False)
    print(f"[INFO] Hiperparâmetros salvos em {csv_path}")

    # JSON
    json_path = os.path.join(hyper_dir, "hyperparameters.json")
    with open(json_path, "w") as f_json:
        json.dump(params_dict, f_json, indent=4)

    # LaTeX
    latex_path = os.path.join(hyper_dir, "hyperparameters_table.tex")
    with open(latex_path, "w") as f_latex:
        f_latex.write("\\begin{table}[H]\n\\centering\n\\begin{tabular}{ll}\n")
        f_latex.write("\\hline\n\\textbf{Hiperparâmetro} & \\textbf{Valor} \\\\\n\\hline\n")
        for key, value in params_dict.items():
            f_latex.write(f"{key} & {value} \\\\\n")
        f_latex.write("\\hline\n\\end{tabular}\n\\caption{Hiperparâmetros da execução}\n\\end{table}\n")


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

def format_hidden_dim(hd):
        return '-'.join(map(str, hd)) if isinstance(hd, list) else str(hd)


# Preprocessing and basic stats
df['label'] = df['label'].astype(int)
df['sample_weight'] = 1 + df['label'] * sample_weight_value

print(f" Dataset before undersampling: {len(df)} samples")
print(f"  - Vulnerable: {df['label'].sum()} | Non-vulnerable: {len(df) - df['label'].sum()}")

# Graph size info (for SortPooling k and decoder padding decisions)
graph_sizes = df['size'].values
print(f" Graph size stats — Max: {np.max(graph_sizes)}, 95th percentile: {int(np.percentile(graph_sizes, 95))}")

# Apply undersampling
df_resampled = apply_undersampling(df, strategy=UNDERSAMPLING_STRAT, method=UNDERSAMPLING_METHOD)

if UNDERSAMPLING_METHOD:
    print(f" Dataset after undersampling: {len(df_resampled)} samples")
    print(f"  - Vulnerable: {df_resampled['label'].sum()} | Non-vulnerable: {len(df_resampled) - df_resampled['label'].sum()}")

df = df_resampled
sample_weights = df['sample_weight'].values

# Dataset split
dataset_size = len(df)
indices = np.arange(dataset_size)
#np.random.seed(10)
np.random.seed(int(time.time()) % (2**32 - 1)) 
np.random.shuffle(indices)

split = int(np.floor(0.3 * dataset_size))
train_indices, test_indices = indices[split:], indices[:split]

trainset = df[['graphs', 'label']].values[train_indices]
testset = df[['graphs', 'label']].values[test_indices]


###########################################################

## Normalization

all_feature_train_data = trainset[0,0].ndata['features']
all_feature_test_data = testset[0,0].ndata['features']
#print("train & test data shape:",all_feature_train_data.shape, all_feature_test_data.shape)

for i in range(1, len(trainset)):
    #########
    current_feat = trainset[i, 0].ndata['features']
    if current_feat.shape[1] != all_feature_train_data.shape[1]:
        print(f"[ERROR] Mismatch at index {i}: Expected {all_feature_train_data.shape[1]} features but got {current_feat.shape[1]}")
        continue  # or raise error if you want to crash here
    #########
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
    if graph_type != 'cfg':
        return dataset  # No adjustment needed for AST/PDG

    for i in range(len(dataset)):
        t = dataset[i, 0].ndata['features']
        # Remove pre-identified zero columns in CFG
        t = torch.cat((t[:, 0:3], t[:, 4:15], t[:, 16:18]), 1) if num_features > 11 else torch.cat((t[:, 0:3], t[:, 4:]), 1)
        dataset[i, 0].ndata['features'] = t
    return dataset

#  Removes one feature as it is always zero (no node was assigned to type "numeric constant")
#print(trainset.shape)
if normalization is not None or normalization == "":
    trainset = adjust_dataset(trainset)
    testset = adjust_dataset(testset)
    if graph_type == 'cfg':
        if num_features > 11:
            # memory management features are also available
            num_features -= 3
        else:
            num_features -= 1
print("len(trainset):", len(trainset))

###########################################################


sampler = WeightedRandomSampler(sample_weights[train_indices], num_samples=len(trainset), replacement=True)
data_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, sampler=sampler)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize modules explicitly (modular setup)
if classifier_type == "vgg":
    encoder = DGCNNEncoder(num_features, hidden_dimension, sortpooling_k=k_sortpooling, use_sortpool=False).to(device)
else:
    encoder = DGCNNEncoder(num_features, hidden_dimension, sortpooling_k=k_sortpooling, use_sortpool=True).to(device)


if classifier_type == "vgg":
    vgg_adapter = DGCNNVGGAdapter(embedding_dim=sum(hidden_dimension), conv2d_channels=conv2dChannelParam).to(device)
    classifier_model = VGGnet(in_channels=conv2dChannelParam).to(device)
elif classifier_type == "conv1d":
    classifier_model = DGCNNConv1DClassifier(input_dim=k_sortpooling * hidden_dimension[-1]).to(device)
else:
    raise ValueError("Invalid classifier type. Choose 'vgg' or 'conv1d'.")

# Prepara argumentos comuns para save_embeddings
vgg_adapter_arg = vgg_adapter if classifier_type == "vgg" else None
classifier_model_arg = classifier_model if classifier_type in ["vgg", "conv1d"] else None

# Model name simplification
model_name = "GAT4_sortpool20V2"

# Hyperparameters dict clearly documented
params_dict = {
    "dataset_name": dataset_name,
    "hidden_dimensions": hidden_dimension,
    "normalization": normalization,
    "num_epochs": num_epochs,
    "undersampling_method": UNDERSAMPLING_METHOD or "None",
    "undersampling_ratio": UNDERSAMPLING_STRAT,
    "CEL_weight": CEL_weight,
    "sample_weight_value": sample_weight_value,
    "model": model_name,
    "k_sortpooling": k_sortpooling,
    "heads": heads,
    "dropout_rate": dropout_rate,
    "conv2dChannelParam": conv2dChannelParam,
    "USE_AUTOENCODER": USE_AUTOENCODER,
    "AUTOENCODER_EPOCHS": AUTOENCODER_EPOCHS,
    "FREEZE_ENCODER": FREEZE_ENCODER,
    "classifier_type": classifier_type
}

# Directory and artifacts setup
artifact_suffix = f"{dataset_name}_hd-{format_hidden_dim(hidden_dimension)}_norm-{normalization}_clf-{classifier_type}_"
artifact_suffix += f"e{num_epochs}_us-{UNDERSAMPLING_METHOD or '0'}-{UNDERSAMPLING_STRAT}_"
artifact_suffix += f"w-{CEL_weight[0]}-{CEL_weight[1]}_sw{sample_weight_value}_m-{model_name}_"
artifact_suffix += f"k-{k_sortpooling}_h{heads}_dr-{dropout_rate}_c2d-{conv2dChannelParam}_"
artifact_suffix += f"ae-{USE_AUTOENCODER}-aep-{AUTOENCODER_EPOCHS}-fz-{FREEZE_ENCODER}" if USE_AUTOENCODER else "noae"


output_base_dir = "output/runs"
run_output_dir = os.path.join(output_base_dir, artifact_suffix)
embedding_dir = os.path.join(run_output_dir, "embeddings")
prediction_dir = os.path.join(run_output_dir, "predictions")
stats_dir = os.path.join(run_output_dir, "stats")

for directory in [embedding_dir, prediction_dir, stats_dir]:
    os.makedirs(directory, exist_ok=True)

# Autoencoder training
if USE_AUTOENCODER:
    decoder = GraphDecoder(
        embedding_dim=k_sortpooling * hidden_dimension[-1],
        num_nodes=k_sortpooling,
        feature_dim=num_features
    ).to(device)

    print(f"[INFO] Starting autoencoder pretraining ({AUTOENCODER_EPOCHS} epochs)")
    train_autoencoder(
        encoder=encoder,
        decoder=decoder,
        data_loader=data_loader,
        device=device,
        sortpooling_k=k_sortpooling,
        feature_dim=num_features,
        num_epochs=AUTOENCODER_EPOCHS,
        stats_dir=stats_dir,
        use_amp_pooling=(classifier_type == "vgg") # ← define aqui claramente
    )
    print("[INFO] Autoencoder training completed.\n")

    if FREEZE_ENCODER:
        for param in encoder.parameters():
            param.requires_grad = False



# Optimizer setup
trainable_params = list(filter(lambda p: p.requires_grad, encoder.parameters()))
if classifier_type == "vgg":
    trainable_params += list(vgg_adapter.parameters()) + list(classifier_model.parameters())
else:
    trainable_params += list(classifier_model.parameters())

optimizer = optim.Adam(trainable_params, lr=learning_rate)

weight = torch.tensor(CEL_weight, dtype=torch.float, device=device)
loss_func = nn.CrossEntropyLoss(weight=weight)

# Training phase
encoder.train()
if classifier_type == "vgg":
    vgg_adapter.train()
classifier_model.train()

stats_dict = {'epoch': [], 'loss': [], 'accuracy': []}

save_embeddings(
    encoder=encoder,
    dataset=testset,
    device=device,
    embedding_dir=embedding_dir,
    prefix="before_training",
    epoch=num_epochs,
    batch_size=batch_size,
    classifier_type=classifier_type,
    vgg_adapter=vgg_adapter_arg,
    classifier_model=classifier_model_arg
)

print("\n========= Starting Training Phase ==========")
for epoch in range(num_epochs):
    total_loss, correct, total = 0, 0, 0

    for graphs, labels in data_loader:
        graphs = [g.to(device) for g in graphs]
        labels = labels.to(device)

        embeddings = encoder(graphs)
        if classifier_type == "vgg":
            batched_graph = embeddings  # returned from encoder when use_sortpool=False
            vgg_input = adjust_to_vgg(vgg_adapter(batched_graph))
            predictions = classifier_model(vgg_input)
        else:
            predictions = classifier_model(embeddings)

        loss = loss_func(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(data_loader)
    accuracy = correct / total

    stats_dict['epoch'].append(epoch)
    stats_dict['loss'].append(epoch_loss)
    stats_dict['accuracy'].append(accuracy)

    print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')


# Save stats clearly
df_stats = pd.DataFrame(stats_dict).set_index('epoch')
df_stats.plot(figsize=(8, 5))
plt.xlabel('Epoch')
plt.title('Training Loss and Accuracy')
plt.savefig(os.path.join(stats_dir, f"training_results_epoch{num_epochs}.png"))
plt.close()

save_embeddings(
    encoder=encoder,
    dataset=testset,
    device=device,
    embedding_dir=embedding_dir,
    prefix="after_training",
    epoch=num_epochs,
    batch_size=batch_size,
    classifier_type=classifier_type,
    vgg_adapter=vgg_adapter_arg,
    classifier_model=classifier_model_arg
)

# Evaluation Phase
encoder.eval()
if classifier_type == "vgg":
    vgg_adapter.eval()
classifier_model.eval()

print("\n========= Starting Evaluation Phase =========")
all_preds, all_labels = [], []

with torch.no_grad():
    for graphs, labels in DataLoader(testset, batch_size=batch_size, collate_fn=collate):
        graphs = [g.to(device) for g in graphs]
        labels = labels.cpu().numpy()

        embeddings = encoder(graphs)
        if classifier_type == "vgg":
            batched_graph = embeddings  # encoder returned the batched graph
            vgg_input = adjust_to_vgg(vgg_adapter(batched_graph))
            preds = classifier_model(vgg_input).argmax(dim=1).cpu().numpy()
        else:
            preds = classifier_model(embeddings).argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

# Metrics & Confusion Matrix
classification_df = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True, zero_division=0)).transpose()
classification_df.to_csv(os.path.join(stats_dir, "classification_report.csv"))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.savefig(os.path.join(stats_dir, "confusion_matrix.png"))
plt.close()

print("\n[CONFUSION MATRIX]")
print(cm)

# Optional: Also print the full classification report
print("\n[CLASSIFICATION REPORT]")
print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

# Log hyperparameters
log_hyperparameters(run_output_dir, params_dict)