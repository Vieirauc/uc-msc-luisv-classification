import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# === CONFIGURATION ===
classifier_type = "conv1d"  # or "vgg"
USE_PCA_FOR_CLASSIFIERS = False  # Set to True if you want to use PCA for classifiers

run_dir = r"C:\Users\luka3\Desktop\UC\MSI\Tese\code\pamela_runs\docker_outputs\runs\PDG_multirun_embs\T9_pdg-dataset-linux_undersampled20k_hd-32-32-32-32_norm-minmax_clf-conv1d_ep30_wsw_swv37.25_k32_dr0.3_noae_embsave\run_4"
embedding_dir = os.path.join(run_dir, "embeddings")
prediction_dir = os.path.join(run_dir, "predictions")
output_dir = os.path.join(run_dir, "embedding_analysis")
os.makedirs(output_dir, exist_ok=True)

# Prefixes produced by your training script
TRAIN_PREFIX = "train_after"
TEST_PREFIX  = "test_after"
AE_TRAIN_PREFIX = "train_ae"   # may not exist
AE_TEST_PREFIX  = "test_ae"    # may not exist

def _load_tensor(path):
    t = torch.load(path)
    return t

def _load_pack(prefix):
    """Return (X, y, y_pred or None). X is 2D: [n_samples, feat_dim]."""
    X_t = _load_tensor(os.path.join(embedding_dir, f"dgcnn_embeddings_{prefix}.pt"))
    if X_t.ndim > 2:
        X = X_t.reshape(X_t.shape[0], -1).numpy()
    else:
        X = X_t.numpy()

    y = _load_tensor(os.path.join(embedding_dir, f"{prefix}_labels.pt")).numpy()

    pred_path = os.path.join(prediction_dir, f"{classifier_type}_predictions_{prefix}.pt")
    y_pred = _load_tensor(pred_path).numpy() if os.path.exists(pred_path) else None
    return X, y, y_pred

# --- Load AFTER-training splits
X_train, y_train, y_pred_train = _load_pack(TRAIN_PREFIX)
X_test,  y_test,  y_pred_test  = _load_pack(TEST_PREFIX)

#--- Fit scaler + PCA on TRAIN, transform both
#scaler = StandardScaler().fit(X_train)              # NEW: scale features
#X_train_std = scaler.transform(X_train)
#X_test_std  = scaler.transform(X_test)

# --- Fit PCA on TRAIN, transform both
pca_dim = 20
pca = PCA(n_components=pca_dim).fit(X_train)
Z_train = pca.transform(X_train)
Z_test  = pca.transform(X_test)
#pca = PCA(n_components=pca_dim).fit(X_train_std)
#Z_train = pca.transform(X_train_std)
#Z_test  = pca.transform(X_test_std)

# --- (Optional) Export tidy CSV with PCA coords + metadata
df_train = pd.DataFrame(Z_train, columns=[f"pc{i+1}" for i in range(Z_train.shape[1])])
df_train["split"] = "train"
df_train["label"] = y_train

df_test = pd.DataFrame(Z_test, columns=[f"pc{i+1}" for i in range(Z_test.shape[1])])
df_test["split"] = "test"
df_test["label"] = y_test
if y_pred_test is not None:
    df_test["pred"] = y_pred_test

df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all.to_csv(os.path.join(output_dir, "embeddings_pca20.csv"), index=False)

# --- 2D plots (PC1 vs PC2) on TEST split
def _scatter_2d(Z, labels, names, colors, title, out_png):
    plt.figure(figsize=(8, 6))
    for val, name, color in zip(range(len(names)), names, colors):
        mask = (labels == val)
        if np.any(mask):
            plt.scatter(Z[mask, 0], Z[mask, 1], label=name, alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, out_png))
    plt.close()

# Labels view: Neutral (0) vs Vulnerable (1) on TEST
Zt2 = Z_test[:, :2]
labels_names = ["Neutral", "Vulnerable"]
labels_colors = ["blue", "red"]
_scatter_2d(Zt2, y_test, labels_names, labels_colors, "PCA (TEST) — Ground Truth", "pca_test_labels.png")

# Predictions view: TP/FP/TN/FN (only if y_pred_test exists)
if y_pred_test is not None:
    tp = (y_pred_test == 1) & (y_test == 1)
    fp = (y_pred_test == 1) & (y_test == 0)
    tn = (y_pred_test == 0) & (y_test == 0)
    fn = (y_pred_test == 0) & (y_test == 1)

    # Map each sample to class id 0..3
    view_labels = np.full_like(y_test, fill_value=-1)
    view_labels[tp] = 0
    view_labels[fp] = 1
    view_labels[tn] = 2
    view_labels[fn] = 3

    names = ["TP", "FP", "TN", "FN"]
    colors = ["green", "red", "blue", "orange"]
    _scatter_2d(Zt2, view_labels, names, colors, "PCA (TEST) — TP/FP/TN/FN", "pca_test_predclasses.png")

# --- Compute balanced class weights from TRAIN labels
classes = np.unique(y_train)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

# --- Conventional classifiers: train on TRAIN, evaluate on TEST
if USE_PCA_FOR_CLASSIFIERS:
   CLF_train, CLF_test, feature_space = Z_train, Z_test, "PCA(20)"
else:
    CLF_train, CLF_test, feature_space = X_train, X_test, "Raw (standardized)"
print(f"[INFO] RF + SVM on {feature_space} with train→test protocol (balanced sample weights)")

sw = compute_sample_weight(class_weight='balanced', y=y_train)

# RF
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(CLF_train, y_train, sample_weight=sw)
y_pred_rf = rf.predict(CLF_test)
pd.DataFrame(classification_report(y_test, y_pred_rf, output_dict=True)).T.to_csv(
    os.path.join(output_dir, "classification_report_rf.csv"))
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[0, 1])
ConfusionMatrixDisplay(cm_rf, display_labels=["Neutral", "Vulnerable"]).plot()
plt.title("Random Forest — Confusion Matrix (TEST)")
plt.savefig(os.path.join(output_dir, "confusion_matrix_rf.png"))
plt.close()

# SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(CLF_train, y_train, sample_weight=sw)
y_pred_svm = svm.predict(CLF_test)
pd.DataFrame(classification_report(y_test, y_pred_svm, output_dict=True)).T.to_csv(
    os.path.join(output_dir, "classification_report_svm.csv"))
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=[0, 1])
ConfusionMatrixDisplay(cm_svm, display_labels=["Neutral", "Vulnerable"]).plot()
plt.title("SVM — Confusion Matrix (TEST)")
plt.savefig(os.path.join(output_dir, "confusion_matrix_svm.png"))
plt.close()

# --- (Optional) AE stage plots, if you saved train_ae/test_ae
ae_train_path = os.path.join(embedding_dir, f"dgcnn_embeddings_{AE_TRAIN_PREFIX}.pt")
ae_test_path  = os.path.join(embedding_dir, f"dgcnn_embeddings_{AE_TEST_PREFIX}.pt")
if os.path.exists(ae_train_path) and os.path.exists(ae_test_path):
    X_train_ae, y_train_ae, _ = _load_pack(AE_TRAIN_PREFIX)
    X_test_ae,  y_test_ae,  _ = _load_pack(AE_TEST_PREFIX)
    pca_ae = PCA(n_components=2).fit(X_train_ae)
    Z_test_ae_2d = pca_ae.transform(X_test_ae)[:, :2]
    _scatter_2d(Z_test_ae_2d, y_test_ae, ["Neutral", "Vulnerable"], ["blue", "red"],
                "PCA (TEST) — Post-AE Encoder (no classifier)", "pca_test_labels_postae.png")
