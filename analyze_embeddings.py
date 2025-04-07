import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load embeddings
embedding_dir = "output/embeddings"
prediction_dir = "output/predictions"
epoch = 5  # Change epoch as needed

# Load stored embeddings, predictions, and labels
dgcnn_embeddings = torch.load(f"{embedding_dir}/dgcnn_embeddings_epoch{epoch}.pt")
vgg_features = torch.load(f"{prediction_dir}/vgg_features_epoch{epoch}.pt")  # New feature embeddings
vgg_predictions = torch.load(f"{prediction_dir}/vgg_predictions_epoch{epoch}.pt")
labels = torch.load(f"{embedding_dir}/train_labels_epoch{epoch}.pt")

# Convert to numpy
dgcnn_np = dgcnn_embeddings.numpy()
vgg_features_np = vgg_features.numpy()
vgg_predictions_np = vgg_predictions.numpy()
labels_np = labels.numpy()

# Flatten embeddings (Convert from 4D to 2D)

#Shape before: (3709, 32, 30, 128)
dgcnn_np = dgcnn_np.reshape(dgcnn_np.shape[0], -1)  # Reshape to (samples, features)
#Shape after: (3709, 122880)

# Define classification categories based on VGG output (Boolean mask)
true_positive = (vgg_predictions_np == 1) & (labels_np == 1)
false_positive = (vgg_predictions_np == 1) & (labels_np == 0)
true_negative = (vgg_predictions_np == 0) & (labels_np == 0)
false_negative = (vgg_predictions_np == 0) & (labels_np == 1)

# Extract corresponding embeddings from DGCNN
tp_dgcnn = dgcnn_np[true_positive]
fp_dgcnn = dgcnn_np[false_positive]
tn_dgcnn = dgcnn_np[true_negative]
fn_dgcnn = dgcnn_np[false_negative]

# Flatten if necessary
#vgg_features_np = vgg_features_np.reshape(vgg_features_np.shape[0], -1)

tp_vgg = vgg_features_np[true_positive]
fp_vgg = vgg_features_np[false_positive]
tn_vgg = vgg_features_np[true_negative]
fn_vgg = vgg_features_np[false_negative]

# Create labels for visualization
labels_dgcnn = np.concatenate([
    np.full(len(tp_dgcnn), 0),  # True Positive
    np.full(len(fp_dgcnn), 1),  # False Positive
    np.full(len(tn_dgcnn), 2),  # True Negative
    np.full(len(fn_dgcnn), 3),  # False Negative
])

labels_vgg = np.concatenate([
    np.full(len(tp_vgg), 0),  # True Positive
    np.full(len(fp_vgg), 1),  # False Positive
    np.full(len(tn_vgg), 2),  # True Negative
    np.full(len(fn_vgg), 3),  # False Negative
])

# Concatenate embeddings for visualization
all_dgcnn_embeddings = np.concatenate([tp_dgcnn, fp_dgcnn, tn_dgcnn, fn_dgcnn])
all_vgg_embeddings = np.concatenate([tp_vgg, fp_vgg, tn_vgg, fn_vgg])

# Define color mapping
label_names = ["True Positive", "False Positive", "True Negative", "False Negative"]
colors = ["green", "red", "blue", "orange"]

# ==============================
# 1. PCA Visualization - DGCNN Embeddings
# ==============================
pca = PCA(n_components=2)
pca_result_dgcnn = pca.fit_transform(all_dgcnn_embeddings)

print(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(labels_dgcnn)):
    plt.scatter(pca_result_dgcnn[labels_dgcnn == label, 0], pca_result_dgcnn[labels_dgcnn == label, 1], 
                label=label_names[label], color=colors[label], alpha=0.6)
plt.title("PCA - DGCNN Embeddings")
plt.legend()
plt.savefig(f"{embedding_dir}/pca_dgcnn_epoch{epoch}.png")
plt.show()

# ==============================
# 2.1 PCA Visualization - VGG Feature Space
# ==============================
pca_result_vgg = pca.fit_transform(all_vgg_embeddings)

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(labels_vgg)):
    plt.scatter(pca_result_vgg[labels_vgg == label, 0], pca_result_vgg[labels_vgg == label, 1], 
                label=label_names[label], color=colors[label], alpha=0.6)
plt.title("PCA - VGG Feature Space")
plt.legend()
plt.savefig(f"{embedding_dir}/pca_vgg_epoch{epoch}.png")
plt.show()



# ==============================
# 3. t-SNE Visualization - DGCNN Embeddings
# ==============================
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result_dgcnn = tsne.fit_transform(all_dgcnn_embeddings)

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(labels_dgcnn)):
    plt.scatter(tsne_result_dgcnn[labels_dgcnn == label, 0], tsne_result_dgcnn[labels_dgcnn == label, 1], 
                label=label_names[label], color=colors[label], alpha=0.6)
plt.title("t-SNE - DGCNN Embeddings")
plt.legend()
plt.savefig(f"{embedding_dir}/tsne_dgcnn_epoch{epoch}.png")
plt.show()

# ==============================
# 4. t-SNE Visualization - VGG Feature Space
# ==============================
tsne_result_vgg = tsne.fit_transform(all_vgg_embeddings)

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(labels_vgg)):
    plt.scatter(tsne_result_vgg[labels_vgg == label, 0], tsne_result_vgg[labels_vgg == label, 1], 
                label=label_names[label], color=colors[label], alpha=0.6)
plt.title("t-SNE - VGG Feature Space")
plt.legend()
plt.savefig(f"{embedding_dir}/tsne_vgg_epoch{epoch}.png")
plt.show()
