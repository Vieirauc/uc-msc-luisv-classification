import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load embeddings
embedding_dir = "output/embeddings"
prediction_dir = "output/predictions"
epoch = 9  # Change epoch as needed


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

# Generate confusion matrix
cm = confusion_matrix(labels_np, vgg_predictions_np, labels=[0, 1])

'''
# Display and save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title(f"Confusion Matrix - Epoch {epoch}")
plt.savefig(f"stats/confusion-matrix_epoch{epoch}.png")
plt.show()

# Generate classification report
report = classification_report(labels_np, vgg_predictions_np, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(f"stats/classification_report_epoch{epoch}.csv")

print(f"Confusion Matrix and classification report saved for epoch {epoch}.")
'''
# Flatten embeddings (Convert from 4D to 2D)
#dgcnn_np = dgcnn_np.reshape(dgcnn_np.shape[0], -1)  # Reshape to (samples, features)
# Flatten if necessary
#vgg_features_np = vgg_features_np.reshape(vgg_features_np.shape[0], -1)
#vgg_predictions_np = vgg_predictions_np.reshape(vgg_predictions_np.shape[0], -1)

print(f'VGG Features head: {vgg_features_np[:10]}')
print(f'VGG Predictions head: {vgg_predictions_np[:10]}')
print(f'Labels head: {labels_np[:10]}')

# Define classification categories based on VGG output
true_positive = (vgg_predictions_np == 1) & (labels_np == 1)
false_positive = (vgg_predictions_np == 1) & (labels_np == 0)
true_negative = (vgg_predictions_np == 0) & (labels_np == 0)
false_negative = (vgg_predictions_np == 0) & (labels_np == 1)

# Extract corresponding embeddings from VGG predictions
tp_vgg = vgg_features_np[true_positive]
fp_vgg = vgg_features_np[false_positive]
tn_vgg = vgg_features_np[true_negative]
fn_vgg = vgg_features_np[false_negative]

#print(f"tp_vgg: {tp_vgg.shape}, fp_vgg: {fp_vgg.shape}, tn_vgg: {tn_vgg.shape}, fn_vgg: {fn_vgg.shape}")
#print(f"tp_vgg: {tp_vgg[:10]}, fp_vgg: {fp_vgg[:10]}, tn_vgg: {tn_vgg[:10]}, fn_vgg: {fn_vgg[:10]}")


# Concatenate embeddings for visualization
#all_dgcnn_embeddings = np.concatenate([tp_dgcnn, fp_dgcnn, tn_dgcnn, fn_dgcnn])
all_vgg_embeddings = np.concatenate([tp_vgg, fp_vgg, tn_vgg, fn_vgg])
print(f"Fixed Shape of all_vgg_embeddings: {all_vgg_embeddings.shape}")


print(f"Shape of all_vgg_embeddings: {all_vgg_embeddings.shape}")

# Ensure boolean masks are 1D (remove extra dimension)
#true_positive = true_positive.flatten()
#false_positive = false_positive.flatten()
#true_negative = true_negative.flatten()
#false_negative = false_negative.flatten()

print("Shapes:")
print(f"DGCNN Embeddings: {dgcnn_np.shape}")
print(f"Labels: {labels_np.shape}")
print(f"Predictions: {vgg_predictions_np.shape}")
print(f"Boolean Mask: {true_positive.shape}")
print(f"Boolean Mask first 10 (true positives): {true_positive[:10]}")

# Extract corresponding embeddings from DGCNN
tp_dgcnn = dgcnn_np[true_positive]
fp_dgcnn = dgcnn_np[false_positive]
tn_dgcnn = dgcnn_np[true_negative]
fn_dgcnn = dgcnn_np[false_negative]


# print first 10 dgcnn embeddings
#print(dgcnn_np[:10])

# print first 10 of each category of dgcnn embeddings
#print(tp_dgcnn[:10])
#print(fp_dgcnn[:10])
#print(tn_dgcnn[:10])
#print(fn_dgcnn[:10])

# print first 10 vgg features and predictions
#print(vgg_features_np[:10])
#print(vgg_predictions_np[:10])



