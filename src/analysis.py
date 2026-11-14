import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

def cluster_embeddings(vectors, n_clusters=4, random_state=42):
    matrix = np.asarray(vectors, dtype=np.float32)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return model.fit_predict(matrix)

def find_similar_words(model, word, top_n=10):
    return [(w, float(score)) for w, score in model.most_similar(word, topn=top_n)]

def cosine_similarity(vector_a, vector_b):
    a = np.asarray(vector_a, dtype=np.float32).reshape(1, -1)
    b = np.asarray(vector_b, dtype=np.float32).reshape(1, -1)
    return float(sklearn_cosine_similarity(a, b)[0, 0])
