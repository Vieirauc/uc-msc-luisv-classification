import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import umap.umap_ as umap

# Load embeddings
embedding_dir = "output/embeddings"
epoch = 1  # Change epoch as needed

# Load stored embeddings
tp_embeddings = torch.load(f"{embedding_dir}/tp_embeddings_epoch{epoch}.pt")
fp_embeddings = torch.load(f"{embedding_dir}/fp_embeddings_epoch{epoch}.pt")
tn_embeddings = torch.load(f"{embedding_dir}/tn_embeddings_epoch{epoch}.pt")
fn_embeddings = torch.load(f"{embedding_dir}/fn_embeddings_epoch{epoch}.pt")

# Convert to numpy
tp_np = tp_embeddings.numpy()
fp_np = fp_embeddings.numpy()
tn_np = tn_embeddings.numpy()
fn_np = fn_embeddings.numpy()

# Create labels
labels = np.concatenate([
    np.full(len(tp_np), 0),  # True Positive (0)
    np.full(len(fp_np), 1),  # False Positive (1)
    np.full(len(tn_np), 2),  # True Negative (2)
    np.full(len(fn_np), 3),  # False Negative (3)
])

# Concatenate all embeddings
all_embeddings = np.concatenate([tp_np, fp_np, tn_np, fn_np])

# Define color mapping
label_names = ["True Positive", "False Positive", "True Negative", "False Negative"]
colors = ["green", "red", "blue", "orange"]

# ==============================
# 1. PCA Visualization
# ==============================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_embeddings)

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(labels)):
    plt.scatter(pca_result[labels == label, 0], pca_result[labels == label, 1], 
                label=label_names[label], color=colors[label], alpha=0.6)
plt.title("PCA Visualization of Embeddings")
plt.legend()
plt.savefig(f"{embedding_dir}/pca_epoch{epoch}.png")
plt.show()

# ==============================
# 2. t-SNE Visualization
# ==============================
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(labels)):
    plt.scatter(tsne_result[labels == label, 0], tsne_result[labels == label, 1], 
                label=label_names[label], color=colors[label], alpha=0.6)
plt.title("t-SNE Visualization of Embeddings")
plt.legend()
plt.savefig(f"{embedding_dir}/tsne_epoch{epoch}.png")
plt.show()

# ==============================
# 3. UMAP Visualization
# ==============================
#umap_reducer = umap.UMAP(n_components=2, random_state=42)
#umap_result = umap_reducer.fit_transform(all_embeddings)

#plt.figure(figsize=(8, 6))
#for i, label in enumerate(np.unique(labels)):
#    plt.scatter(umap_result[labels == label, 0], umap_result[labels == label, 1], 
#                label=label_names[label], color=colors[label], alpha=0.6)
#plt.title("UMAP Visualization of Embeddings")
#plt.legend()
#plt.savefig(f"{embedding_dir}/umap_epoch{epoch}.png")
#plt.show()
