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

# === CONFIGURATION ===
run_dir = "output/runs/cfg-dataset-linux-sample1k_hd-32-32-32-32_norm-minmax_e2_us-0-0.2_w-1-1_sw0_m-GAT4_k-6_h4_dr-0.3_c2d-32_noae"
embedding_dir = os.path.join(run_dir, "embeddings")
prediction_dir = os.path.join(run_dir, "predictions")
output_dir = os.path.join(run_dir, "embedding_analysis")
os.makedirs(output_dir, exist_ok=True)

# === Find available epochs ===
pattern = re.compile(r'dgcnn_embeddings_test_epoch(\d+)\.pt')
epochs = sorted(int(pattern.search(f).group(1)) for f in os.listdir(embedding_dir) if pattern.search(f))
print(f"[INFO] Found epochs: {epochs}")
first_epoch, last_epoch = epochs[0], epochs[-1]

def load_data(epoch):
    dgcnn_embeddings = torch.load(f"{embedding_dir}/dgcnn_embeddings_test_epoch{epoch}.pt")
    vgg_features = torch.load(f"{prediction_dir}/vgg_features_test_epoch{epoch}.pt")
    vgg_predictions = torch.load(f"{prediction_dir}/vgg_predictions_test_epoch{epoch}.pt")
    labels = torch.load(f"{embedding_dir}/test_epoch{epoch}_labels.pt")

    return dgcnn_embeddings.numpy().reshape(dgcnn_embeddings.size(0), -1), \
           vgg_features.numpy(), \
           vgg_predictions.numpy(), \
           labels.numpy()

def analyze_epoch(epoch):
    print(f"[INFO] Analyzing epoch {epoch}")
    dgcnn_np, vgg_features_np, vgg_predictions_np, labels_np = load_data(epoch)

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

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_dgcnn_embeddings)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(labels_dgcnn)):
        plt.scatter(pca_result[labels_dgcnn == label, 0], pca_result[labels_dgcnn == label, 1],
                    label=label_names[label], color=colors[label], alpha=0.6)
    plt.title(f"PCA - DGCNN Embeddings (Epoch {epoch})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"pca_dgcnn_epoch{epoch}.png"))
    plt.close()

    # Random Forest (only for last epoch)
    if epoch == last_epoch:
        print("[INFO] Running Random Forest classifier on final embeddings...")
        pca_full = PCA(n_components=20).fit_transform(dgcnn_np)
        X_train, X_test, y_train, y_test = train_test_split(pca_full, labels_np, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Save report
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, f"classification_report_random_forest_epoch{epoch}.csv"))
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
        disp.plot()
        plt.title(f"Random Forest Confusion Matrix (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_rf_epoch{epoch}.png"))
        plt.close()

# === Run analysis for first and last epochs
analyze_epoch(first_epoch)
analyze_epoch(last_epoch)