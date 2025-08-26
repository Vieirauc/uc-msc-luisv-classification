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
import contextlib
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
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from torchvision.ops import sigmoid_focal_loss
from torch_geometric.nn import GCNConv
from torchvision.ops import sigmoid_focal_loss

from detect_vulnerabilities_vgg import VGGnet

# %%

project = 'linux' # 'gecko-dev'#'linux'
version = None # 'v0.5_filtered'
graph_type = 'pdg' #

#cfg-dataset-linux-v0.5 has 101513 entries
#cfg-dataset-linux-v0.5_filtered has 65685 entries

#if version:
#    dataset_name = f"{graph_type}-dataset-{project}-{version}"
#else:
#    dataset_name = f"{graph_type}-dataset-{project}"

#dataset_name = 'cfg-dataset-linux-v0.5_filtered'
#dataset_name = 'cfg-dataset-linux-sample1k'
dataset_name = 'pdg-dataset-linux_undersampled20k'

dataset_path = 'datasets/'

if not os.path.isfile(dataset_path + dataset_name + '.pkl'):
    df = load_dataset(dataset_path + dataset_name)
    df.to_pickle(dataset_path + dataset_name + '.pkl')
else:
    df = pd.read_pickle(dataset_path + dataset_name  + '.pkl')

# %%

N_RUNS = 5  # ou 10, conforme precisares
SEED_LIST = [42 + i*11 for i in range(N_RUNS)]  # Seeds diferentes, mas fixas

DEBUG = False
SAVE_EMBEDDINGS = True

ZNORM = "znorm"
MINMAX = "minmax"
SORTPOOLING = "sort_pooling"
ADAPTIVEMAXPOOLING = "adaptive_max_pooling"

normalization = MINMAX #ZNORM
#pooling_type = ADAPTIVEMAXPOOLING #SORTPOOLING

USE_FOCAL_LOSS = False
alpha = 0.5
gamma = 2.0

AUTO_WEIGHTING = True
USE_CLASS_WEIGHT = False
USE_BOTH_WEIGHTING = False  # ⚠️ só para testes controlados
sample_weight_value = 1
CEL_weight = [1,1]

USE_AUTOENCODER = False
NUM_NODES = 144  # padding fixo
FREEZE_ENCODER = True
learning_rate_ae = 0.001 #0.0001 #0.00001 #0.000001
AUTOENCODER_EPOCHS = 20

classifier_type = "vgg"  # ou "vgg" ou "conv1d"

#heads = 4
hidden_dimension = [32, 32, 32, 32] 
batch_size = 10
k_sortpooling = 32  # for SortPooling
k_amp = 32          # for Adaptive Max Pooling (VGG pathway)
dropout_rate = 0.3 #0.1 
conv2dChannelParam = 32
learning_rate = 0.001 #0.0001 #0.00001 #0.000001 
num_epochs = 30 #2000 #500 # 1000

TESTS = [
    {"id": "T12", "classifier_type": "vgg",    "USE_AUTOENCODER": True},
    {"id": "T10", "classifier_type": "vgg",    "USE_AUTOENCODER": False},
    {"id": "T9",  "classifier_type": "conv1d", "USE_AUTOENCODER": False},
    {"id": "T11", "classifier_type": "conv1d", "USE_AUTOENCODER": True},
]

if graph_type == 'cfg':
    num_features = 19  # 11 base + 8 memory
elif graph_type == 'ast':
    num_features = 8
elif graph_type == 'pdg':
    num_features = 7
else:
    raise ValueError(f"Unsupported graph_type: {graph_type}")


##################################################################################
class DGCNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dimensions, k_sortpooling=None, use_sortpool=True):
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

        self.sortpool = SortPooling(k=k_sortpooling)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, graphs, return_node_embeddings=False):
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

        #print(f"[INFO] DGCNNEncoder output shape: {h_all.shape}")

        if return_node_embeddings:
            return h_all, batched_graph

        if self.use_sortpool:
            h_pooled = self.sortpool(batched_graph, h_all)
            embeddings = self.dropout(h_pooled)
            return embeddings

        batched_graph.ndata['h'] = h_all
        return batched_graph


class DGCNNVGGAdapter(nn.Module):
    def __init__(self, embedding_dim, conv2d_channels=32, k_amp=30):
        super(DGCNNVGGAdapter, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=conv2d_channels,
            kernel_size=13, stride=1, padding=6
        )
        self.amp = nn.AdaptiveMaxPool2d((k_amp, embedding_dim))

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
            nn.Dropout(dropout_rate),
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
        self.num_nodes = num_nodes 
        self.feature_dim = feature_dim

        self.reconstruct_features = nn.Sequential(
            nn.Linear(embedding_dim * num_nodes, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes * feature_dim),
        )

    def forward(self, z):
        out = self.reconstruct_features(z)
        return out.view(-1, self.num_nodes, self.feature_dim)


def train_autoencoder(encoder, decoder, data_loader, device, num_nodes, feature_dim, num_epochs=20, stats_dir="stats"):
    encoder.train()
    decoder.train()

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate_ae)
    loss_func = nn.MSELoss()


    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for graphs, _ in data_loader:
            graphs = [g.to(device) for g in graphs]

            # Obtem os embeddings dos nós antes do pooling
            node_embeddings, batched_graph = encoder(graphs, return_node_embeddings=True)

            # Divide embeddings por grafo
            batch_sizes = batched_graph.batch_num_nodes()
            split_embeddings = torch.split(node_embeddings, batch_sizes.tolist())

            # Pad para tamanho fixo (num_nodes)
            padded_input = []
            ground_truth = []

            for g, embed in zip(graphs, split_embeddings):
                x_real = g.ndata['features'].float()

                if embed.size(0) >= num_nodes:
                    padded_input.append(embed[:num_nodes])
                    ground_truth.append(x_real[:num_nodes])
                else:
                    pad_embed = torch.zeros(num_nodes - embed.size(0), embed.size(1), device=device)
                    pad_real = torch.zeros(num_nodes - x_real.size(0), feature_dim, device=device)
                    padded_input.append(torch.cat([embed, pad_embed], dim=0))
                    ground_truth.append(torch.cat([x_real, pad_real], dim=0))

            Z = torch.stack(padded_input)     # (B, num_nodes, embed_dim)
            X_orig = torch.stack(ground_truth)  # (B, num_nodes, feature_dim)

            X_rec = decoder(Z.view(Z.size(0), -1))  # Flatten to (B, num_nodes * embed_dim)
            loss = loss_func(X_rec, X_orig)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"[Autoencoder] Epoch {epoch}, Loss: {avg_loss:.4f}")
        if epoch in [num_epochs-1] and DEBUG:
            print("[DEBUG] Real vs. Reconstructed (primeiro grafo)")
            print("X_orig:", X_orig[0].cpu().detach().numpy())
            print("X_rec: ", X_rec[0].cpu().detach().numpy())

    # Gravação de stats
    os.makedirs(stats_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(num_epochs), epoch_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Autoencoder Training Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "autoencoder_loss_curve.png"))
    plt.close()

    loss_df = pd.DataFrame({"epoch": list(range(num_epochs)), "loss": epoch_losses})
    loss_df.to_csv(os.path.join(stats_dir, "autoencoder_loss.csv"), index=False)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return graphs, torch.tensor(labels)

def write_file(filename, rows):
    with open("output/" + filename + ".csv", 'w') as output_file:
        for row in rows:
            output_file.write(" ".join([str(a) for a in row.tolist()]) + '\n')

def save_embeddings(encoder, dataset, device, embedding_dir, prediction_dir,
                    prefix, batch_size=10, classifier_type=None,
                    vgg_adapter=None, classifier_model=None,
                    compact_save=True):
    encoder.eval()
    if classifier_model:
        classifier_model.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    all_graph_embeddings = []   # what you'll use for PCA / RF / SVM
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for graphs, labels in data_loader:
            graphs = [g.to(device) for g in graphs]
            labels = labels.to(device)
            all_labels.append(labels.cpu())

            if classifier_type == "vgg":
                # Encoder returns a batched_graph with node embeddings in ndata['h']
                batched_graph = encoder(graphs)

                # Adapter converts per-graph node embeddings -> (B, C, k_amp, F)
                adapted = vgg_adapter(batched_graph)  # no padding yet
                B, C, H, W = adapted.shape

                # ---- Save compact or full embeddings ----
                if compact_save:
                    # Global pooling over spatial dims → (B, C)
                    pooled = adapted.mean(dim=[2, 3])  # or torch.amax(adapted, dim=[2, 3])
                    all_graph_embeddings.append(pooled.cpu())
                else:
                    # Full flattened output (B, C*k_amp*F) — large!
                    graph_vecs = adapted.view(B, -1)
                    all_graph_embeddings.append(graph_vecs.cpu())

                # Classifier forward uses the padded 224x224 image
                vgg_in = adjust_to_vgg(adapted)
                logits = classifier_model(vgg_in.to(device))
                preds = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)
                all_predictions.append(preds.cpu())
                all_probs.append(probs.cpu())

            elif classifier_type == "conv1d":
                # Encoder already returns graph-level vectors via SortPooling
                embeddings = encoder(graphs)              # (B, D)
                all_graph_embeddings.append(embeddings.cpu())

                logits = classifier_model(embeddings.to(device))
                preds = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)
                all_predictions.append(preds.cpu())
                all_probs.append(probs.cpu())

            else:
                raise ValueError("classifier_type must be 'vgg' or 'conv1d'")

    # Concatenate collected tensors
    emb_tensor = torch.cat(all_graph_embeddings, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    print(f"[save_embeddings] Shapes — embeddings:{emb_tensor.shape}, labels:{labels_tensor.shape}")

    # Save graph-level embeddings and labels
    torch.save(emb_tensor, os.path.join(embedding_dir, f"dgcnn_embeddings_{prefix}.pt"))
    torch.save(labels_tensor, os.path.join(embedding_dir, f"{prefix}_labels.pt"))

    # Save classifier outputs if available
    if classifier_model is not None and len(all_predictions) > 0:
        pred_tensor = torch.cat(all_predictions, dim=0)
        torch.save(pred_tensor, os.path.join(prediction_dir, f"{classifier_type}_predictions_{prefix}.pt"))

        probs_tensor = torch.cat(all_probs, dim=0)
        torch.save(probs_tensor, os.path.join(prediction_dir, f"{classifier_type}_probs_{prefix}.pt"))

    print(f"[save_embeddings] Saved: {prefix}")



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

def focal_loss(outputs, targets, weight=None, alpha=0.5, gamma=2.0):
    one_hot = F.one_hot(targets, num_classes=2).float()
    return sigmoid_focal_loss(outputs, one_hot, alpha=alpha, gamma=gamma, reduction="mean")

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

def adjust_dataset(trainset, testset, remove_dead_columns=True):
    """
    Remove colunas totalmente zeradas dos grafos em trainset e testset.
    Funciona para qualquer tipo de grafo (CFG, AST, PDG).
    """
    print(f"[INFO] Ajustar datasets para grafos do tipo '{graph_type}'")

    # Obter número de features dos nós
    sample_feat = trainset[0, 0].ndata['features']
    num_feat = sample_feat.shape[1]
    col_sums = torch.zeros(num_feat)

    # Somar colunas em ambos os conjuntos
    for ds in [trainset, testset]:
        for i in range(len(ds)):
            col_sums += ds[i, 0].ndata['features'].sum(dim=0)

    # Identificar colunas completamente zeradas
    zero_columns = (col_sums == 0).nonzero(as_tuple=True)[0].tolist()

    if zero_columns:
        print(f"[WARNING] Colunas vazias detectadas: {zero_columns}")

        if remove_dead_columns:
            print("[INFO] A remover colunas vazias...")
            mask = torch.ones(num_feat, dtype=torch.bool)
            mask[zero_columns] = False

            for ds in [trainset, testset]:
                for i in range(len(ds)):
                    f = ds[i, 0].ndata['features']
                    ds[i, 0].ndata['features'] = f[:, mask]

            global num_features
            num_features = mask.sum().item()
            print(f"[INFO] Num features após limpeza: {num_features}")
    else:
        print("[OK] Nenhuma coluna vazia detectada.")
        num_features = num_feat

    return trainset, testset


# Preprocessing and basic stats
df['label'] = df['label'].astype(int)
df['sample_weight'] = 1 + df['label'] * sample_weight_value

print(f" Dataset before undersampling: {len(df)} samples")
print(f"  - Vulnerable: {df['label'].sum()} | Non-vulnerable: {len(df) - df['label'].sum()}")

# Graph size info (for SortPooling k and decoder padding decisions)
graph_sizes = df['size'].values
print(f" Graph size stats — Max: {np.max(graph_sizes)}, 95th percentile: {int(np.percentile(graph_sizes, 95))}")

for test in TESTS:
    # Flip the two knobs per test
    classifier_type = test["classifier_type"]     # "vgg" or "conv1d"
    USE_AUTOENCODER = test["USE_AUTOENCODER"]     # True or False
    TEST_ID = test["id"]                          # "T9"..."T12"

    print(f"\n================= {TEST_ID} — clf={classifier_type}, AE={USE_AUTOENCODER} =================\n")

    run_metrics = []

    for run_idx in range(N_RUNS):
        print(f"\n========== RUN {run_idx + 1}/{N_RUNS} ({TEST_ID}) ==========\n")
        seed = SEED_LIST[run_idx]

        run_df = df.copy(deep=True)
        run_df['graphs'] = run_df['graphs'].apply(lambda g: g.clone())

        # Dataset split
        trainset_df, testset_df = train_test_split(
            run_df[['graphs', 'label']],
            test_size=0.3,
            stratify=run_df['label'],
            random_state=seed
        )

        trainset = trainset_df.values
        testset = testset_df.values

        # Recalcula os pesos com base nos labels do trainset
        train_labels = trainset[:, 1].astype(int)

        if AUTO_WEIGHTING:
            class_counts = np.bincount(train_labels)
            total = sum(class_counts)
            class_weights = [total / (2 * class_counts[i]) for i in range(len(class_counts))]
            CEL_weight = class_weights  # [w_non_vuln, w_vuln]
            sample_weight_value = class_counts[0] / class_counts[1]
            print(f"[INFO] Auto-weighting: {CEL_weight}, sample_weight_value: {sample_weight_value}")
        else:
            class_weights = CEL_weight  # já definido no topo
            sample_weight_value = sample_weight_value  # não usado se USE_CLASS_WEIGHT=True

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


        print("Normalization to be performed")
        if normalization == ZNORM:
            trainset = normalize_znorm(trainset, feat_mean_train, feat_std_train)
            testset = normalize_znorm(testset, feat_mean_test, feat_std_test)
        if normalization == MINMAX: 
            trainset = normalize_minmax(trainset, feat_amin_train, feat_amax_train)
            testset = normalize_minmax(testset, feat_amin_test, feat_amax_test)

        #  Removes one feature as it is always zero (no node was assigned to type "numeric constant")
        #print(trainset.shape)
        if normalization is not None or normalization == "":
            trainset, testset = adjust_dataset(trainset, testset)
            #num_features = trainset[0][0].ndata['features'].shape[1]
            #if graph_type == 'cfg':
            #    if num_features > 11:
            #        # memory management features are also available
            #        num_features -= 3
            #    else:
            #       num_features -= 1
        print("len(trainset):", len(trainset))

        ###########################################################

        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = torch.tensor(CEL_weight, dtype=torch.float, device=device)

        ### Data balacing and DataLoader setup
        if USE_BOTH_WEIGHTING:
            sample_weights = 1 + train_labels * sample_weight_value
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(trainset), replacement=True)
            data_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, sampler=sampler)

            if USE_FOCAL_LOSS:
                loss_func = lambda outputs, targets: focal_loss(outputs, targets, weight)
                print(f"[INFO] Using BOTH sample weights and Focal Loss: {CEL_weight}, sample_weights: {sample_weights[:10]}...")
            else:
                loss_func = nn.CrossEntropyLoss(weight=weight)
                print(f"[INFO] Using BOTH sample weights (sampler) and class weights (loss): {CEL_weight}, sample_weights: {sample_weights[:10]}...")

        elif not USE_CLASS_WEIGHT:
            sample_weights = 1 + train_labels * sample_weight_value
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(trainset), replacement=True)
            data_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, sampler=sampler)

            if USE_FOCAL_LOSS:
                loss_func = lambda outputs, targets: focal_loss(outputs, targets, None)
                print(f"[INFO] Using ONLY sample weights and Focal Loss: {sample_weights[:10]}...")
            else:
                loss_func = nn.CrossEntropyLoss()
                print(f"[INFO] Using ONLY sample weights (sampler): {sample_weights[:10]}...")

        else:
            data_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate)

            if USE_FOCAL_LOSS:
                loss_func = lambda outputs, targets: focal_loss(outputs, targets, weight)
                print(f"[INFO] Using ONLY class weights and Focal Loss: {CEL_weight}")
            else:
                loss_func = nn.CrossEntropyLoss(weight=weight)
                print(f"[INFO] Using ONLY class weights (loss): {CEL_weight}")



        # (Opcional) debug para garantir que as proporções estão corretas
        print(f"[DEBUG] Trainset: {sum(train_labels)} vulnerable / {len(train_labels)} total")
        print(f"[DEBUG] Testset:  {sum(testset[:,1])} vulnerable / {len(testset)} total")

        # Initialize modules explicitly (modular setup)
        if classifier_type == "vgg":
            encoder = DGCNNEncoder(num_features, hidden_dimension, k_sortpooling=k_sortpooling, use_sortpool=False).to(device)
        else:
            encoder = DGCNNEncoder(num_features, hidden_dimension, k_sortpooling=k_sortpooling, use_sortpool=True).to(device)


        if classifier_type == "vgg":
            vgg_adapter = DGCNNVGGAdapter(embedding_dim=sum(hidden_dimension), conv2d_channels=conv2dChannelParam, k_amp=k_amp).to(device)
            classifier_model = VGGnet(in_channels=conv2dChannelParam).to(device)
        elif classifier_type == "conv1d":
            classifier_model = DGCNNConv1DClassifier(input_dim=k_sortpooling * hidden_dimension[-1]).to(device)
        else:
            raise ValueError("Invalid classifier type. Choose 'vgg' or 'conv1d'.")

        # Prepara argumentos comuns para save_embeddings
        vgg_adapter_arg = vgg_adapter if classifier_type == "vgg" else None
        classifier_model_arg = classifier_model if classifier_type in ["vgg", "conv1d"] else None

        # Hyperparameters dict clearly documented
        # Hyperparameters dict clearly documented
        params_dict = {
            "TEST_ID": TEST_ID,
            "dataset_name": dataset_name,
            "hidden_dimensions": hidden_dimension,
            "normalization": normalization,
            "num_epochs": num_epochs,

            # Weighting strategy
            "auto_weighting": AUTO_WEIGHTING,
            "use_class_weight": USE_CLASS_WEIGHT,
            "use_both_weighting": USE_BOTH_WEIGHTING,
            "CEL_weight": CEL_weight,
            "sample_weight_value": sample_weight_value,

            # Focal Loss
            "use_focal_loss": USE_FOCAL_LOSS,
            "focal_alpha": alpha,
            "focal_gamma": gamma,

            # Model architecture
            "classifier_type": classifier_type,
            "k_amp": k_amp if classifier_type == "vgg" else "N/A",
            "k_sortpooling": k_sortpooling,
            #"heads": heads,
            "dropout_rate": dropout_rate,
            "conv2dChannelParam": conv2dChannelParam if classifier_type == "vgg" else "N/A",

            # Autoencoder
            "USE_AUTOENCODER": USE_AUTOENCODER,
            "AUTOENCODER_EPOCHS": AUTOENCODER_EPOCHS,
            "FREEZE_ENCODER": FREEZE_ENCODER,

            # Learning rates
            "learning_rate_ae": learning_rate_ae,
            "learning_rate_classifier": learning_rate
        }

        artifact_suffix = (
            f"{TEST_ID}_"
            f"{dataset_name}"
            f"_hd-{format_hidden_dim(hidden_dimension)}"
            f"_norm-{normalization}"
            f"_clf-{classifier_type}"
            f"_ep{num_epochs}"
        )

        # Estratégia de balanceamento
        if USE_BOTH_WEIGHTING:
            artifact_suffix += f"_wboth_cw-{int(CEL_weight[0])}-{int(CEL_weight[1])}_swv{round(sample_weight_value, 2)}"
        elif USE_CLASS_WEIGHT:
            artifact_suffix += f"_wcw-{int(CEL_weight[0])}-{int(CEL_weight[1])}"
        else:
            artifact_suffix += f"_wsw_swv{round(sample_weight_value, 2)}"

        # Focal loss
        if USE_FOCAL_LOSS:
            artifact_suffix += f"_focal-a{alpha}-g{gamma}"

        # Componentes da arquitetura
        artifact_suffix += (
            (f"_k{k_sortpooling}" if classifier_type == "conv1d" else "")
            + (f"-ampk{k_amp}" if classifier_type == "vgg" else "")
        )
        artifact_suffix += f"_dr{dropout_rate}"

        if classifier_type == "vgg":
            artifact_suffix += f"_c2d{conv2dChannelParam}"

        # Autoencoder
        artifact_suffix += (
            f"_ae{int(USE_AUTOENCODER)}"
            f"-aep{AUTOENCODER_EPOCHS}-fz{int(FREEZE_ENCODER)}" if USE_AUTOENCODER else "_noae"
        )

        if SAVE_EMBEDDINGS:
            artifact_suffix += "_embsave"

        output_base_dir = "output/runs"
        run_output_dir = os.path.join(output_base_dir, artifact_suffix, f"run_{run_idx + 1}")
        embedding_dir = os.path.join(run_output_dir, "embeddings")
        prediction_dir = os.path.join(run_output_dir, "predictions")
        stats_dir = os.path.join(run_output_dir, "stats")

        for directory in [embedding_dir, prediction_dir, stats_dir]:
            os.makedirs(directory, exist_ok=True)

        if USE_AUTOENCODER:
            decoder = GraphDecoder(
                embedding_dim=hidden_dimension[-1],  # ← novo: só a última camada do encoder
                num_nodes=NUM_NODES,
                feature_dim=num_features
            ).to(device)

            print(f"[INFO] Starting autoencoder pretraining ({AUTOENCODER_EPOCHS} epochs)")
            train_autoencoder(
                encoder=encoder,
                decoder=decoder,
                data_loader=data_loader,
                device=device,
                num_nodes=NUM_NODES,
                feature_dim=num_features,
                num_epochs=AUTOENCODER_EPOCHS,
                stats_dir=stats_dir
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

        #weight = torch.tensor(CEL_weight, dtype=torch.float, device=device) if USE_CLASS_WEIGHT else None
        #loss_func = nn.CrossEntropyLoss(weight=weight)

        # Training phase
        if not FREEZE_ENCODER:
            encoder.train()
        if classifier_type == "vgg":
            vgg_adapter.train()
        classifier_model.train()

        stats_dict = {'epoch': [], 'loss': [], 'accuracy': []}

        # Antes do treino
        '''
        if SAVE_EMBEDDINGS:
            save_embeddings(
                encoder=encoder,
                dataset=testset,
                device=device,
                embedding_dir=embedding_dir,
                prediction_dir=prediction_dir,
                prefix="test_before",
                batch_size=batch_size,
                classifier_type=classifier_type,
                vgg_adapter=vgg_adapter_arg,
                classifier_model=classifier_model_arg
            )
        else:
            print("[INFO] SAVE_EMBEDDINGS=False → Skipping .pt saving")
        '''

        print("\n========= Starting Training Phase ==========")
        train_start = time.time()
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

        train_end = time.time()
        print(f"[INFO] Training completed in {(train_end - train_start):.2f} seconds.\n")

        # Save stats clearly
        df_stats = pd.DataFrame(stats_dict).set_index('epoch')
        df_stats.to_csv(os.path.join(stats_dir, "training_stats.csv"))

        # === Gráfico separado de Loss e Accuracy, lado a lado ===
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss por epoch
        axes[0].plot(df_stats.index, df_stats['loss'], label='Loss', color='darkred')
        axes[0].set_title('Training Loss per Epoch')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)

        # Accuracy por epoch
        axes[1].plot(df_stats.index, df_stats['accuracy'], label='Accuracy', color='darkgreen')
        axes[1].set_title('Training Accuracy per Epoch')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f"training_loss_vs_accuracy_epoch{num_epochs}.png"))
        plt.close()

        print(f"[INFO] Saved training loss and accuracy plots to: {os.path.join(stats_dir, f'training_loss_vs_accuracy_epoch{num_epochs}.png')}")

        with open(os.path.join(stats_dir, "training_metadata.txt"), "w") as f:
            f.write(f"Training time (seconds): {train_end - train_start:.2f}\n")
            f.write(f"Epochs: {num_epochs}\n")
            f.write(f"Final loss: {epoch_loss:.4f}\n")
            f.write(f"Final accuracy: {accuracy:.4f}\n")

        # After training completes
        if SAVE_EMBEDDINGS:
            # Save TRAIN embeddings and final predictions
            save_embeddings(
                encoder=encoder, dataset=trainset, device=device,
                embedding_dir=embedding_dir, prediction_dir=prediction_dir,
                prefix="train_after", batch_size=batch_size,
                classifier_type=classifier_type, vgg_adapter=vgg_adapter_arg,
                classifier_model=classifier_model_arg
            )
            # Save TEST embeddings and final predictions
            save_embeddings(
                encoder=encoder, dataset=testset, device=device,
                embedding_dir=embedding_dir, prediction_dir=prediction_dir,
                prefix="test_after", batch_size=batch_size,
                classifier_type=classifier_type, vgg_adapter=vgg_adapter_arg,
                classifier_model=classifier_model_arg
            )
        else:
            print("[INFO] SAVE_EMBEDDINGS=False → Skipping .pt saving")

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
        cm_sum = np.sum(cm)
        cm_percent = cm / cm_sum * 100

        # Create labels with both count and percentage
        labels = np.array([
            [f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False,
                    xticklabels=["Neutral (0)", "Vulnerable (1)"],
                    yticklabels=["Neutral (0)", "Vulnerable (1)"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix with Percentages")
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f"confusion_matrix_run{run_idx + 1}.png"))
        plt.close()

        print("\n[CONFUSION MATRIX]")
        print(cm)

        # Optional: Also print the full classification report
        print("\n[CLASSIFICATION REPORT]")
        print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        run_metrics.append({
            "run": run_idx + 1,
            "seed": seed,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log hyperparameters
        log_hyperparameters(run_output_dir, params_dict)

    df_metrics = pd.DataFrame(run_metrics)
    test_base_dir = os.path.join(output_base_dir, artifact_suffix)
    df_metrics.to_csv(os.path.join(test_base_dir,"multi_run_summary.csv"), index=False)

    print("\n===== MÉDIA FINAL =====")
    print(df_metrics.mean(numeric_only=True).to_string())
    #df_metrics.to_csv(os.path.join(output_base_dir, artifact_suffix, "multi_run_summary.csv"), index=False)
    print("\n===== DESVIO PADRÃO =====")
    print(df_metrics.std(numeric_only=True).to_string())
    df_metrics.describe().to_csv(os.path.join(output_base_dir, artifact_suffix, "summary_stats.csv"))

    # ==== Multi-run plots (Boxplot + Mean ± Std Dev Barplot) ====
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    summary_plot_path = os.path.join(output_base_dir, artifact_suffix, "multi_run_summary_plots.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    sns.boxplot(data=df_metrics[metrics_to_plot], ax=axes[0])
    axes[0].set_title("Distribution of Metrics Across Runs")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)

    # Barplot com média ± std
    mean_vals = df_metrics[metrics_to_plot].mean()
    std_vals = df_metrics[metrics_to_plot].std()
    axes[1].bar(mean_vals.index, mean_vals.values, yerr=std_vals.values, capsize=5, color='skyblue')
    axes[1].set_title("Mean ± Std Dev of Metrics")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close()
    print(f"[INFO] Multi-run performance plots saved to: {summary_plot_path}")
