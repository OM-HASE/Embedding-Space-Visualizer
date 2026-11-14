import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def reduce_embeddings(vectors, algorithm="PCA", random_state=42):
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError("At least three word vectors are required.")
    if algorithm == "PCA":
        return PCA(n_components=2, random_state=random_state).fit_transform(matrix)
    if algorithm == "t-SNE":
        # Perplexity must be smaller than the number of samples.
        perplexity = min(30, max(2, (matrix.shape[0] - 1) // 3))
        perplexity = min(perplexity, matrix.shape[0] - 1)
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state
        ).fit_transform(matrix)
    raise ValueError("Algorithm must be PCA or t-SNE.")
