import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load embeddings
embedding_dir = "output/output_US_0_2/output_filtered/embeddings"
prediction_dir = "output/output_US_0_2/output_filtered/predictions"
epoch = 10  # Change epoch as needed

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

# NOTE: vgg_predictions is basically equal to torch.argmax(vgg_features_np, dim=1)

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
    np.full(len(fn_dgcnn), 3),  # False Negative
    np.full(len(tn_dgcnn), 2),  # True Negative
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
label_names = ["True Positive", "False Positive", "False Negative", "True Negative"]
colors = ["green", "red", "orange", "blue"]


def generate_trainset_report_and_confusion_matrix(predictions, labels, save_path="output/output_US_0_2/output_filtered/", suffix="train"):
    os.makedirs(save_path, exist_ok=True)

    # Converte para tensores, se necessário
    y_true = labels.squeeze()
    y_pred = predictions

    # Se for logit, aplica argmax
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    # Garante que está tudo no formato certo
    y_pred = y_pred.float()
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()

    # Gera classification report
    report = classification_report(y_true_np, y_pred_np, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"{save_path}/classification_report_{suffix}.csv")

    # Gera confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title(f"Confusion Matrix ({suffix})")
    plt.savefig(f"{save_path}/confusion_matrix_{suffix}.png")
    plt.clf()

# Chamada direta (após carregar os .pt)
generate_trainset_report_and_confusion_matrix(torch.tensor(vgg_predictions_np), torch.tensor(labels_np), 
                                              save_path="output/output_US_0_2/output_non_filtered/", suffix=f"train_epoch{epoch}")


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

'''
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
'''


'''
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
'''

'''
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
'''

# PCA com n componentes (por exemplo, 20)
n_components = 20
pca = PCA(n_components=n_components)
pca_dgcnn = pca.fit_transform(dgcnn_np)

# Treino/teste + Random Forest
X_train, X_test, y_train, y_test = train_test_split(pca_dgcnn, labels_np, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# Salvar os resultados da Random Forest
rf_dir = "output/output_US_0_2/output_filtered/random_forest"
#rf_dir = os.path.join(os.path.dirname(embedding_dir), "random_forest")
os.makedirs(rf_dir, exist_ok=True)

# Salva classification report
report_rf = classification_report(y_test, y_pred, output_dict=True)
df_report_rf = pd.DataFrame(report_rf).transpose()
df_report_rf.to_csv(os.path.join(rf_dir, f"classification_report_random_forest_test_epoch{epoch}.csv"))

# Salva confusion matrix
cm_rf = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0, 1])
disp_rf.plot()
plt.title(f"Confusion Matrix - Random Forest (Test Epoch {epoch})")
plt.savefig(os.path.join(rf_dir, f"confusion_matrix_random_forest_test_epoch{epoch}.png"))
plt.clf()


'''
# PCA com n componentes (por exemplo, 20)
n_components = 20
pca = PCA(n_components=n_components)
pca_dgcnn = pca.fit_transform(dgcnn_np)

# Treino/teste + Random Forest
X_train, X_test, y_train, y_test = train_test_split(pca_dgcnn, labels_np, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
'''
