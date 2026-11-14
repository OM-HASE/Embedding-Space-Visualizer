# Embedding Space Visualizer

An interactive NLP application for exploring pre-trained word embeddings in 2D. It supports GloVe and Word2Vec, PCA and t-SNE projections, K-Means clustering, cosine similarity, semantic-neighbor lookup, custom vocabulary input, and downloadable results.

## Features

- **Pre-trained embeddings:** GloVe Wiki-Gigaword 100-dimensional vectors and Word2Vec Google News 300-dimensional vectors.
- **Dimensionality reduction:** switch between PCA and t-SNE.
- **K-Means clustering:** color the projected vocabulary by cluster assignment.
- **Semantic search:** retrieve nearest words using the embedding model's similarity method.
- **Cosine similarity:** compare two selected word vectors.
- **Custom vocabulary:** enter words or upload TXT/CSV files.
- **Exports:** download plotted coordinates and cluster labels as CSV, or the interactive Plotly chart as HTML.

## Requirements

- Python 3.10 or 3.11 recommended.
- Internet connection for the first model download.
- Enough disk space for the selected model.
- A modern browser.

### Model download sizes

The GloVe 100d model is the lighter option and is recommended for the first run. The Google News Word2Vec model is very large (roughly 1.6 GB compressed / several GB after extraction, depending on the cache and package version). Download it only if you have sufficient bandwidth and disk space. The application loads only the selected model and caches it during the Streamlit session.

## Setup (Windows PowerShell)

```powershell
# 1. Move into the project directory
cd Embedding-Space-Visualizer

# 2. Create a virtual environment
py -3.11 -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Start the app
streamlit run app.py
```

If PowerShell blocks activation, run this for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Setup (macOS / Linux)

```bash
cd Embedding-Space-Visualizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## How to use

1. Choose **GloVe Wiki Gigaword 100d** first to validate the setup.
2. Select a built-in topic, type your own vocabulary, or upload a TXT/CSV file.
3. Choose PCA or t-SNE in the sidebar.
4. Adjust the vocabulary limit and number of K-Means clusters.
5. Hover over points to inspect words and cluster membership.
6. Select a word to view its nearest semantic neighbors.
7. Select two words to calculate cosine similarity.
8. Download the coordinate/cluster CSV or interactive HTML plot.

### Custom vocabulary format

TXT (one word per line):

```text
computer
algorithm
database
python
robot
```

CSV: the first column should contain one word per row. Headerless CSV is easiest.

## Project structure

```text
Embedding-Space-Visualizer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── uploads/
│       └── .gitkeep
└── src/
    ├── __init__.py
    ├── embeddings.py
    ├── reduction.py
    ├── analysis.py
    └── utils.py
```

## Methodology

### 1. Word embeddings

Each word is represented by a dense numeric vector. Words used in similar contexts tend to have related vector representations. GloVe learns vectors from global word co-occurrence statistics; Word2Vec learns them by predicting surrounding/context words.

### 2. PCA

Principal Component Analysis is a linear transformation that finds directions of maximum variance. This project retains two principal components for plotting.

### 3. t-SNE

t-distributed Stochastic Neighbor Embedding is a nonlinear technique designed to preserve local neighborhoods. It can reveal local groupings, but distances between far-apart groups and cluster sizes should not be over-interpreted.

### 4. K-Means

K-Means partitions the original high-dimensional vectors into K groups. The resulting labels are then displayed as plot colors. Clustering is performed on the original vectors, not the 2D coordinates.

### 5. Cosine similarity

Cosine similarity compares the angle between two vectors. A higher score indicates more similar vector directions. Similarity scores depend on the embedding model and its training data.

## Suggested evaluation / experiments

For a meaningful academic report, run and record these experiments:

1. Compare PCA and t-SNE using the same vocabulary and embedding model.
2. Compare GloVe and Word2Vec using the same supported vocabulary.
3. Test multiple K-Means values (for example K=3, 4, 5, 6).
4. Compare nearest-neighbor lists for selected words.
5. Record runtime, number of in-vocabulary words, and qualitative cluster coherence.

Do not claim one method is universally better based only on how attractive a plot looks. Use the same words and random seed when comparing methods.

## Troubleshooting

- **First launch is slow:** model files are downloaded and cached. Wait for completion.
- **Word2Vec download is too large:** use GloVe, or select Word2Vec only when disk space and bandwidth permit.
- **Word not found:** the model may not contain it. Try lowercase/common vocabulary or another model.
- **t-SNE error:** t-SNE requires at least three samples; use at least three supported words.
- **SciPy / Gensim import errors:** use a clean Python 3.10/3.11 environment and reinstall requirements.
- **CSV upload parsing:** use a headerless CSV with the word in its first column, or upload TXT.

## Important limitations

- The 2D projection is an approximation of high-dimensional relationships.
- t-SNE layouts can change with vocabulary, perplexity and random seed.
- A word embedding usually has one vector per token and may not distinguish different meanings of polysemous words.
- Pre-trained embeddings can contain social and cultural biases inherited from their training corpora.

## License and attribution

GloVe and Word2Vec model files are distributed by their respective publishers/providers and may have their own terms. Review the source model terms before redistributing model files. This repository contains code only and downloads models on demand via `gensim.downloader`.

## Academic project abstract (editable)

**Embedding Space Visualizer** is an interactive natural language processing application that explores semantic relationships encoded in pre-trained word embeddings. The system loads GloVe and Word2Vec vectors and applies Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE) to project high-dimensional representations into two dimensions. K-Means clustering highlights groups of related vocabulary, while cosine similarity and nearest-neighbor search allow users to inspect semantic relationships. Users can select built-in vocabularies, enter custom words, or upload a vocabulary file, then export visualization coordinates and interactive plots. The application provides an accessible environment for comparing embedding models and dimensionality-reduction techniques.

## Future enhancements

- Add FastText for subword-aware representations.
- Add automatic topic extraction from documents.
- Add adjustable t-SNE perplexity and UMAP.
- Add silhouette score and cluster evaluation.
- Add contextual embeddings from transformer models.
