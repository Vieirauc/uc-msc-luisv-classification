import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re

AUTOENCONDER_EPOCHS = 10
num_epochs = 10  # Number of epochs for VGG training, adjust as needed

# === CONFIGURATION ===
run_dir = "C:/Users/luka3/Desktop/UC/MSI/Tese/code/pamela_runs/docker_outputs/runs/cfg-dataset-linux-v0.5_filtered_hd-32-32-32-32_norm-minmax_e10_us-0-0.2_w-1-1_sw0_m-GAT4_k-6_h4_dr-0.3_c2d-32_ae-True-aep-10-fz-True"
embedding_dir = os.path.join(run_dir, "embeddings")
prediction_dir = os.path.join(run_dir, "predictions")
output_dir = os.path.join(run_dir, "embedding_analysis")
os.makedirs(output_dir, exist_ok=True)

# === Define os modos a analisar ===
MODES = {
    "before": 0,
    "after": AUTOENCONDER_EPOCHS,
    "after_vgg": num_epochs
}

def load_data(epoch, mode):
    prefix = f"test_{mode}_autoencoder_epoch{epoch}"
    dgcnn_file = os.path.join(embedding_dir, f"dgcnn_embeddings_{prefix}.pt")
    vgg_features_file = os.path.join(prediction_dir, f"vgg_features_{prefix}.pt")
    vgg_predictions_file = os.path.join(prediction_dir, f"vgg_predictions_{prefix}.pt")
    labels_file = os.path.join(embedding_dir, f"{prefix}_labels.pt")

    dgcnn_embeddings = torch.load(dgcnn_file)
    vgg_features = torch.load(vgg_features_file)
    vgg_predictions = torch.load(vgg_predictions_file)
    labels = torch.load(labels_file)

    return dgcnn_embeddings.numpy().reshape(dgcnn_embeddings.size(0), -1), \
           vgg_features.numpy(), \
           vgg_predictions.numpy(), \
           labels.numpy()


def analyze_epoch(epoch, mode, run_random_forest=False):
    print(f"[INFO] Analyzing epoch {epoch} ({mode})")
    dgcnn_np, vgg_features_np, vgg_predictions_np, labels_np = load_data(epoch, mode)

    if mode == "after_vgg":

        true_positive = (vgg_predictions_np == 1) & (labels_np == 1)
        false_positive = (vgg_predictions_np == 1) & (labels_np == 0)
        true_negative = (vgg_predictions_np == 0) & (labels_np == 0)
        false_negative = (vgg_predictions_np == 0) & (labels_np == 1)

        tp = dgcnn_np[true_positive]
        fp = dgcnn_np[false_positive]
        tn = dgcnn_np[true_negative]
        fn = dgcnn_np[false_negative]

        all_dgcnn_embeddings = np.concatenate([tp, fp, tn, fn])
        labels_dgcnn = np.concatenate([
            np.full(len(tp), 0),
            np.full(len(fp), 1),
            np.full(len(fn), 3),
            np.full(len(tn), 2),
        ])

        num_positive = np.sum(labels_np == 1)
        num_negative = np.sum(labels_np == 0)
        print(f"[INFO] Epoch {epoch} - Positives: {num_positive}, Negatives: {num_negative}, Ratio: {num_positive / (num_positive + num_negative + 1e-6):.2f}")

        label_names = ["TP", "FP", "FN", "TN"]
        colors = ["green", "red", "orange", "blue"]

    else:
        # Just use raw labels
        vulnerable = dgcnn_np[labels_np == 1]
        neutral = dgcnn_np[labels_np == 0]
        all_dgcnn_embeddings = np.concatenate([vulnerable, neutral])
        labels_dgcnn = np.concatenate([
            np.full(len(vulnerable), 1),
            np.full(len(neutral), 0)
        ])
        label_names = ["Neutral", "Vulnerable"]
        colors = ["blue", "red"]

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_dgcnn_embeddings)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(labels_dgcnn)):
        plt.scatter(pca_result[labels_dgcnn == label, 0], pca_result[labels_dgcnn == label, 1],
                    label=label_names[label], color=colors[label], alpha=0.6)
    plt.title(f"PCA - DGCNN Embeddings (Epoch {epoch}, {mode})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"pca_dgcnn_epoch{epoch}_{mode}.png"))
    plt.close()

    # Optional Random Forest
    if run_random_forest:
        print("[INFO] Running Random Forest classifier on embeddings...")
        pca_full = PCA(n_components=20).fit_transform(dgcnn_np)
        X_train, X_test, y_train, y_test = train_test_split(pca_full, labels_np, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, f"classification_report_random_forest_epoch{epoch}_{mode}.csv"))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
        disp.plot()
        plt.title(f"Random Forest Confusion Matrix (Epoch {epoch}, {mode})")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_rf_epoch{epoch}_{mode}.png"))
        plt.close()


# Corre ambos sem Random Forest
analyze_epoch(epoch=0, mode="before")
analyze_epoch(epoch=AUTOENCONDER_EPOCHS, mode="after")

#analyze_epoch(epoch=AUTOENCONDER_EPOCHS, mode="after", run_random_forest=True)