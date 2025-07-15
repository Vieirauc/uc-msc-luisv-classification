import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === CONFIGURATION ===
AUTOENCONDER_EPOCHS = 10
num_epochs = 10
classifier_type = "conv1d"  # ou "vgg"

run_dir = r"output\runs\cfg-dataset-linux-sample1k_hd-32-32-32-32_norm-minmax_clf-conv1d_ep20_us-0-0.2_wcw-1-3_k16_dr0.3_noae"
embedding_dir = os.path.join(run_dir, "embeddings")
prediction_dir = os.path.join(run_dir, "predictions")
output_dir = os.path.join(run_dir, "embedding_analysis")
os.makedirs(output_dir, exist_ok=True)

# === Define os modos a analisar ===
MODES = {
    "before": "test_before",
    "after": "test_after"
}

def load_data(prefix):
    dgcnn_file = os.path.join(embedding_dir, f"dgcnn_embeddings_{prefix}.pt")
    predictions_file = os.path.join(prediction_dir, f"{classifier_type}_predictions_{prefix}.pt")
    labels_file = os.path.join(embedding_dir, f"{prefix}_labels.pt")

    dgcnn_embeddings = torch.load(dgcnn_file)
    predictions = torch.load(predictions_file)
    labels = torch.load(labels_file)

    return dgcnn_embeddings.numpy().reshape(dgcnn_embeddings.size(0), -1), \
           predictions.numpy(), \
           labels.numpy()

def analyze_epoch(mode, prefix, view="predictions", run_random_forest=False):
    """
    mode: "before" ou "after"
    prefix: prefixo usado nos ficheiros
    view: "labels" → Neutral vs Vulnerable | "predictions" → TP, FP, TN, FN
    """
    print(f"[INFO] Analyzing: {mode} ({prefix}) — View: {view}")
    dgcnn_np, predictions_np, labels_np = load_data(prefix)

    if view == "predictions":
        true_positive = (predictions_np == 1) & (labels_np == 1)
        false_positive = (predictions_np == 1) & (labels_np == 0)
        true_negative = (predictions_np == 0) & (labels_np == 0)
        false_negative = (predictions_np == 0) & (labels_np == 1)

        tp = dgcnn_np[true_positive]
        fp = dgcnn_np[false_positive]
        tn = dgcnn_np[true_negative]
        fn = dgcnn_np[false_negative]

        all_dgcnn_embeddings = np.concatenate([tp, fp, tn, fn])
        labels_dgcnn = np.concatenate([
            np.full(len(tp), 0),  # TP
            np.full(len(fp), 1),  # FP
            np.full(len(tn), 2),  # TN
            np.full(len(fn), 3),  # FN
        ])

        label_names = ["TP", "FP", "TN", "FN"]
        colors = ["green", "red", "blue", "orange"]

        if mode == "after":
            num_positive = np.sum(labels_np == 1)
            num_negative = np.sum(labels_np == 0)
            print(f"[INFO] Positives: {num_positive}, Negatives: {num_negative}, Ratio: {num_positive / (num_positive + num_negative + 1e-6):.2f}")

    elif view == "labels":
        vulnerable = dgcnn_np[labels_np == 1]
        neutral = dgcnn_np[labels_np == 0]

        all_dgcnn_embeddings = np.concatenate([neutral, vulnerable])
        labels_dgcnn = np.concatenate([
            np.full(len(neutral), 0),
            np.full(len(vulnerable), 1)
        ])

        label_names = ["Neutral", "Vulnerable"]
        colors = ["blue", "red"]

    else:
        raise ValueError(f"Invalid view: {view}. Choose 'labels' or 'predictions'.")

    # PCA visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_dgcnn_embeddings)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(labels_dgcnn)):
        plt.scatter(pca_result[labels_dgcnn == label, 0],
                    pca_result[labels_dgcnn == label, 1],
                    label=label_names[label], color=colors[label], alpha=0.6)

    plt.title(f"PCA - DGCNN Embeddings ({mode}, {view})")
    plt.legend()
    filename = f"pca_dgcnn_{prefix}_{view}.png"
    plt.savefig(os.path.join(output_dir, filename))
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
        pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, f"classification_report_rf_{prefix}.csv"))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
        disp.plot()
        plt.title(f"Random Forest Confusion Matrix ({mode})")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_rf_{prefix}.png"))
        plt.close()


# Analisar antes e depois com as duas visões
for mode, prefix in MODES.items():
    analyze_epoch(mode=mode, prefix=prefix, view="labels")       # Ground-truth
    analyze_epoch(mode=mode, prefix=prefix, view="predictions")  # TP/FP/TN/FN

