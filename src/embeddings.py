import gensim.downloader as api

MODEL_IDS = {
    "GloVe Wiki Gigaword 100d": "glove-wiki-gigaword-100",
    "Word2Vec Google News 300d": "word2vec-google-news-300",
}

def available_models():
    return list(MODEL_IDS.keys())

def load_embedding_model(display_name):
    """Download a model from the public Gensim-data repository and load it."""
    if display_name not in MODEL_IDS:
        raise ValueError(f"Unknown model: {display_name}")
    return api.load(MODEL_IDS[display_name])

def get_vectors(model, words):
    found_words, vectors, missing = [], [], []
    for word in words:
        try:
            vectors.append(model[word])
            found_words.append(word)
        except (KeyError, TypeError):
            missing.append(word)
    return found_words, vectors, missing
