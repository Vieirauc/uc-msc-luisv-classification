# Classification Pipeline

Graph-based vulnerability detection with a DGCNN encoder and pluggable classifiers.
You can train, evaluate, and analyze embeddings from saved runs.

## Contents

* detect\_vulnerabilities.py – main training and evaluation script
* load\_datasets.py – CSV to DGL graph loader
* analyze\_embeddings.py – PCA plots and classic classifiers on saved embeddings
* ---

## Quick Start

### 1. Build the Docker image

```
docker build -t msiproj .
```
Note: "msiproj" is an example name for the image.

### 2. Run a container with GPU

Pick one of these:

```
# Single GPU 0
docker run --gpus '"device=0"' -it --name msiproj_container msiproj

# All GPUs
docker run --gpus all -it --name msiproj_container msiproj
```
Note: "msiproj_container" is an example name for the container.

Re-attach later:

```
docker start -ai msiproj_container
docker exec -it msiproj_container bash
```

### 3. Copy datasets into the container

```
docker cp /path/to/your_dataset.csv msiproj_container:/workspace/uc-msc-luisv-classification/datasets/
```

The loader expects a semicolon-separated CSV with columns:
`label size adjacency_matrix feature_matrix`

Each row → one DGL graph.

* adjacency\_matrix is parsed and the diagonal is set to one
* feature\_matrix is reshaped to size × feature\_dim

See load\_datasets.py for details.

---
## Dependencies and Setup Notes

Some required libraries are not fully covered by `requirements.txt`. If you run into DGL or PyG related errors, install them manually with the proper CUDA version:

```
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html

pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

Check the URLs for compatibility with your installed `torch` and `CUDA` versions.

---

## Workflow Tips

When experimenting with hyperparameters, you don’t need to commit every change in `detect_vulnerabilities.py`. After running your test, you can restore the file to its clean state:

```
# Edit hyperparameters and run your experiment
git restore detect_vulnerabilities.py
git pull
```

This avoids polluting the repository history with temporary edits.

---
## Train and Evaluate

Open detect\_vulnerabilities.py and set the top-level parameters:

* Dataset: dataset\_name, dataset\_path
* Graph type: graph\_type one of cfg, ast, pdg
* Classifier: classifier\_type conv1d or vgg
* Autoencoder: USE\_AUTOENCODER, FREEZE\_ENCODER
* Normalization: MINMAX or ZNORM
* Pooling / sizes: k\_sortpooling, k\_amp
* Class balancing: weighted sampler or class weights

Run training:

```
python detect_vulnerabilities.py
```

### Outputs

Each run is saved under output/runs/\<auto\_named\_run>/ with:

* stats/ → training curves (loss, accuracy) and CSVs
* classification\_report.csv, confusion\_matrix.png
* hyperparameters/ → CSV, JSON, LaTeX table with run config
* embeddings/ and predictions/ (if saving enabled)

The run folder name encodes hyperparameters for easy comparison.

---

## Analyze Embeddings

Use analyze\_embeddings.py to visualize and probe embeddings.

Config options:

* run\_dir → path to a trained run under output/runs
* classifier\_type → must match the run
* run\_random\_forest and run\_svm → enable RF or SVM probes

Run analysis:

```
python analyze_embeddings.py
```

Outputs go to output/runs/<run>/embedding\_analysis/ and include PCA plots, confusion matrices, and reports.

---

## Dataset Format

CSV format expected by load\_datasets.py:

* label → 0/1 or boolean
* size → number of nodes
* adjacency\_matrix → stringified list, becomes size × size matrix (diagonal = 1)
* feature\_matrix → stringified list, becomes size × feature\_dim matrix

Each row becomes a DGL graph with node features under "features".

---

## GPU and Library Check

Inside the container, verify CUDA:

```
python3 -c "import torch; print('PyTorch', torch.__version__); print('CUDA', torch.version.cuda); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## Screen Tool (for remote runs)

Useful commands:

```
# Start new session
screen -S train

# Detach (keeps process running)
Ctrl + A, then D

# List sessions
screen -ls

# Reattach
screen -r train

# Kill session
screen -XS train quit
```

---

## Clean Outputs

Delete old run outputs:

Iinside the container:

```
rm -rf /workspace/uc-msc-luisv-cfg-dataset/output/runs/*
```

---

## Notes

* Modify hyperparameters in detect\_vulnerabilities.py, rerun, and commit changes if needed.
* Embeddings and results are structured for reproducibility and comparison across runs.
* External classifiers (RF, SVM) and PCA visualization are optional but help interpret embeddings.

---
