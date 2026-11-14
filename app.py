
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.embeddings import (
    load_embedding_model,
    get_vectors,
    available_models
)

from src.reduction import reduce_embeddings

from src.analysis import (
    cluster_embeddings,
    find_similar_words,
    cosine_similarity
)

from src.utils import parse_custom_words


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Embedding Space Visualizer",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Embedding Space Visualizer")

st.markdown(
    """
    Explore semantic relationships between words using
    **GloVe, Word2Vec, PCA, t-SNE, K-Means and Cosine Similarity.**
    """
)

st.divider()


# ============================================================
# 2. BUILT-IN VOCABULARY
# ============================================================

TOPICS = {

    "Mixed Topics": """
    computer software internet python programming database
    algorithm robot dog cat lion tiger elephant horse bird
    fish school college student teacher university book learning
    king queen prince princess kingdom apple bread rice coffee
    city country river mountain
    """,

    "Technology": """
    computer laptop keyboard mouse software hardware internet
    website server database cloud network security python java
    coding programming artificial intelligence machine learning
    deep learning robot algorithm data smartphone application
    processor developer
    """,

    "Programming Languages": """
    python java javascript typescript c cpp csharp ruby php
    swift kotlin golang rust scala perl html css sql r matlab
    programming coding compiler interpreter software developer
    """,

    "Artificial Intelligence": """
    artificial intelligence machine learning deep learning
    neural network dataset training testing prediction
    classification regression clustering algorithm model
    accuracy precision recall chatbot transformer computer
    vision language learning data
    """,

    "Animals": """
    dog cat lion tiger elephant horse monkey bird fish whale
    dolphin bear wolf fox rabbit snake eagle penguin cow goat
    sheep giraffe zebra crocodile
    """,

    "Education": """
    school college student teacher university book knowledge
    learning research mathematics science history classroom
    examination degree graduation engineering professor
    computer physics chemistry biology
    """,

    "Royalty": """
    king queen prince princess royal kingdom emperor empress
    crown palace throne monarch kingdom castle
    """,

    "Food": """
    apple banana orange bread rice pizza pasta coffee tea milk
    cheese vegetable fruit chocolate mango potato tomato
    """,

    "Geography": """
    city country river mountain ocean forest desert island
    continent village capital climate india germany france
    japan china pune mumbai berlin paris tokyo london
    """
}


# ============================================================
# 3. CACHE EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading embedding model. First download may take time..."
)
def get_model_cached(model_name):

    return load_embedding_model(model_name)


# ============================================================
# 4. CACHE DIMENSIONALITY REDUCTION
# ============================================================

@st.cache_data(show_spinner=False)
def calculate_projection(vectors, algorithm, seed):

    matrix = np.asarray(vectors, dtype=np.float32)

    return reduce_embeddings(
        matrix,
        algorithm,
        seed
    )


# ============================================================
# 5. SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.subheader("Embedding Model")

    model_name = st.selectbox(
        "Choose embedding model",
        available_models(),
        index=0
    )

    st.divider()

    st.subheader("Visualization Settings")

    method = st.selectbox(
        "Dimensionality Reduction",
        ["PCA", "t-SNE"]
    )

    max_words = st.slider(
        "Maximum vocabulary size",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

    clusters = st.slider(
        "Number of K-Means Clusters",
        min_value=2,
        max_value=12,
        value=4
    )

    random_state = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42
    )

    st.divider()

    st.subheader("Vocabulary Source")

    vocabulary_source = st.radio(
        "Choose vocabulary",
        [
            "Built-in Topic",
            "Enter Words",
            "Upload TXT / CSV"
        ]
    )

    if vocabulary_source == "Built-in Topic":

        topic = st.selectbox(
            "Select Topic",
            list(TOPICS.keys())
        )

    else:

        topic = None


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

try:

    model = get_model_cached(model_name)

    st.sidebar.success("Embedding model loaded successfully")

except Exception as exc:

    st.error("Unable to load the embedding model.")

    st.exception(exc)

    st.info(
        "Check your internet connection, available memory, "
        "disk space and Gensim installation."
    )

    st.stop()


# ============================================================
# 7. GET USER VOCABULARY
# ============================================================

raw_words = []

if vocabulary_source == "Built-in Topic":

    raw_words = TOPICS[topic].split()

elif vocabulary_source == "Enter Words":

    entered_words = st.text_area(
        "Enter words separated by commas, spaces or new lines",
        value=(
            "computer, software, internet, python, java, "
            "algorithm, database, robot, dog, cat, lion, "
            "tiger, india, germany, student, teacher"
        ),
        height=130
    )

    raw_words = entered_words.replace(",", " ").split()

else:

    uploaded_file = st.file_uploader(
        "Upload your vocabulary file",
        type=["txt", "csv"]
    )

    if uploaded_file:

        try:

            raw_words = parse_custom_words(uploaded_file)

        except Exception as exc:

            st.error(f"Unable to read uploaded file: {exc}")

    st.caption(
        "TXT: One word per line or comma-separated. "
        "CSV: First column is used."
    )


# ============================================================
# 8. PROCESS VOCABULARY
# ============================================================

if raw_words:

    words = list(
        dict.fromkeys(
            word.strip().lower()
            for word in raw_words
            if word.strip()
        )
    )[:max_words]

    st.info(f"Processing {len(words)} unique input words.")

    valid_words, vectors, missing_words = get_vectors(
        model,
        words
    )

    if missing_words:

        with st.expander(
            f"⚠️ {len(missing_words)} words not found in model"
        ):

            st.write(", ".join(missing_words))

    if len(valid_words) < 3:

        st.warning(
            "Please provide at least three words supported "
            "by the selected embedding model."
        )

        st.stop()

    # Convert vectors into NumPy array

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    # Effective number of clusters

    effective_clusters = min(
        int(clusters),
        len(valid_words)
    )

    # ========================================================
    # 9. DIMENSIONALITY REDUCTION
    # ========================================================

    st.header("📊 Dimensionality Reduction")

    try:

        coords = calculate_projection(
            vectors,
            method,
            int(random_state)
        )

    except Exception as exc:

        st.error("Dimensionality reduction failed.")

        st.exception(exc)

        st.stop()

    st.success(
        f"{method} successfully reduced "
        f"{vectors.shape[1]} dimensions to 2 dimensions."
    )

    # ========================================================
    # 10. ACTUAL K-MEANS CLUSTERING
    # ========================================================

    st.header("🎯 Actual K-Means Clustering")

    try:

        # K-Means is performed on original high-dimensional vectors.
        # It does NOT use the 2D PCA/t-SNE coordinates.

        labels = cluster_embeddings(
            vectors,
            effective_clusters,
            int(random_state)
        )

    except Exception as exc:

        st.error("K-Means clustering failed.")

        st.exception(exc)

        st.stop()

    # Create dataframe

    frame = pd.DataFrame({

        "word": valid_words,

        "x": coords[:, 0],

        "y": coords[:, 1],

        "cluster_id": labels

    })

    frame["cluster"] = frame["cluster_id"].apply(
        lambda value: f"Cluster {int(value) + 1}"
    )

    # ========================================================
    # 11. MAIN CLUSTER VISUALIZATION
    # ========================================================

    st.subheader("K-Means Cluster Visualization")

    st.caption(
        f"{len(valid_words)} words | "
        f"{model_name} | "
        f"{method} | "
        f"{effective_clusters} clusters"
    )

    fig = px.scatter(

        frame,

        x="x",

        y="y",

        color="cluster",

        text="word",

        hover_name="word",

        hover_data={

            "cluster": True,

            "x": ":.4f",

            "y": ":.4f",

            "cluster_id": False

        },

        title=f"Actual K-Means Clustering using {model_name}",

        color_discrete_sequence=px.colors.qualitative.Bold

    )

    fig.update_traces(

        textposition="top center",

        marker=dict(

            size=12,

            opacity=0.85,

            line=dict(

                width=1,

                color="white"

            )

        )

    )

    # Calculate visual centers in the 2D projected space.
    # These are NOT the original high-dimensional K-Means centroids.

    centers_2d = (

        frame.groupby("cluster_id")[["x", "y"]]

        .mean()

        .reset_index()

    )

    centers_2d["cluster"] = centers_2d["cluster_id"].apply(

        lambda value: f"Cluster {int(value) + 1} center"

    )

    # Add visual centers to the scatter plot

    fig.add_trace(

        go.Scatter(

            x=centers_2d["x"],

            y=centers_2d["y"],

            mode="markers+text",

            marker=dict(

                symbol="star",

                size=22,

                color="black",

                line=dict(

                    color="white",

                    width=2

                )

            ),

            text=centers_2d["cluster"],

            textposition="bottom center",

            name="2D Visual Centers",

            hovertemplate=(

                "%{text}<br>"

                "X: %{x:.4f}<br>"

                "Y: %{y:.4f}"

                "<extra></extra>"

            )

        )

    )

    fig.update_layout(

        height=700,

        template="plotly_white",

        legend_title_text="K-Means Clusters",

        xaxis_title=f"{method} Dimension 1",

        yaxis_title=f"{method} Dimension 2",

        hovermode="closest"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.caption(
        "Colored points show actual K-Means assignments. "
        "Star markers show the average 2D position of each cluster, "
        "not the original high-dimensional K-Means centroid."
    )

    # ========================================================
    # 12. PROJECT METRICS
    # ========================================================

    st.subheader("📈 Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Words Plotted",
        len(valid_words)
    )

    col2.metric(
        "K-Means Clusters",
        effective_clusters
    )

    col3.metric(
        "Embedding Dimensions",
        vectors.shape[1]
    )

    col4.metric(
        "Missing Words",
        len(missing_words)
    )

    # ========================================================
    # 13. CLUSTER-WISE WORD DISTRIBUTION
    # ========================================================

    st.divider()

    st.subheader("📋 Cluster-wise Word Distribution")

    cluster_summary = (

        frame.groupby("cluster")

        .agg(

            Word_Count=("word", "count"),

            Words=("word", lambda words: ", ".join(words))

        )

        .reset_index()

    )

    st.dataframe(
        cluster_summary,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # 14. CLUSTER SIZE BAR CHART
    # ========================================================

    st.subheader("Cluster Size Comparison")

    size_fig = px.bar(

        cluster_summary,

        x="cluster",

        y="Word_Count",

        color="cluster",

        text="Word_Count",

        title="Number of Words in Each Cluster",

        color_discrete_sequence=px.colors.qualitative.Bold

    )

    size_fig.update_layout(

        height=420,

        template="plotly_white",

        xaxis_title="K-Means Cluster",

        yaxis_title="Number of Words",

        showlegend=False

    )

    st.plotly_chart(
        size_fig,
        width="stretch"
    )

    # ========================================================
    # 15. INDIVIDUAL CLUSTER EXPLORATION
    # ========================================================

    st.subheader("🔍 Explore Individual Clusters")

    selected_cluster = st.selectbox(

        "Select a cluster",

        sorted(frame["cluster"].unique())

    )

    selected_cluster_words = frame[

        frame["cluster"] == selected_cluster

    ]["word"].tolist()

    st.write(
        f"**Words in {selected_cluster}:**"
    )

    st.write(", ".join(selected_cluster_words))

    st.metric(
        "Words in selected cluster",
        len(selected_cluster_words)
    )

    # ========================================================
    # 16. COSINE SIMILARITY SEARCH
    # ========================================================

    st.divider()

    st.header("🔎 Semantic Similarity Analysis")

    left, right = st.columns(2)

    # --------------------------------------------------------
    # Similar words
    # --------------------------------------------------------

    with left:

        st.subheader("Find Similar Words")

        selected_word = st.selectbox(

            "Choose a word",

            valid_words,

            key="similar_word"

        )

        top_n = st.slider(

            "Number of similar words",

            min_value=3,

            max_value=20,

            value=10

        )

        try:

            similar = find_similar_words(

                model,

                selected_word,

                top_n

            )

            similar_frame = pd.DataFrame(

                similar,

                columns=[

                    "Word",

                    "Cosine Similarity"

                ]

            )

            st.dataframe(

                similar_frame,

                width="stretch",

                hide_index=True

            )

        except Exception as exc:

            st.warning(
                f"Similarity search unavailable: {exc}"
            )

    # --------------------------------------------------------
    # Compare two words
    # --------------------------------------------------------

    with right:

        st.subheader("Compare Two Words")

        word_a = st.selectbox(

            "First word",

            valid_words,

            index=0,

            key="word_a"

        )

        word_b = st.selectbox(

            "Second word",

            valid_words,

            index=min(1, len(valid_words) - 1),

            key="word_b"

        )

        score = cosine_similarity(

            model[word_a],

            model[word_b]

        )

        st.metric(

            "Cosine Similarity",

            f"{score:.4f}"

        )

        st.caption(
            "Higher values indicate more similar vector directions. "
            "Similarity scores depend on the selected embedding model."
        )

    # ========================================================
    # 17. DOWNLOAD OUTPUTS
    # ========================================================

    st.divider()

    st.header("⬇️ Download Results")

    csv_bytes = frame.to_csv(
        index=False
    ).encode("utf-8")

    html_bytes = fig.to_html(
        include_plotlyjs="cdn"
    ).encode("utf-8")

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        st.download_button(

            label="⬇ Download Cluster Data (CSV)",

            data=csv_bytes,

            file_name="embedding_cluster_results.csv",

            mime="text/csv",

            width="stretch"

        )

    with download_col2:

        st.download_button(

            label="⬇ Download Interactive Visualization (HTML)",

            data=html_bytes,

            file_name="embedding_space_visualization.html",

            mime="text/html",

            width="stretch"

        )

    # ========================================================
    # 18. COMPLETE DATA TABLE
    # ========================================================

    with st.expander("View Complete Embedding and Cluster Data"):

        st.dataframe(

            frame.sort_values(

                "cluster_id"

            ),

            width="stretch",

            hide_index=True

        )

    # ========================================================
    # 19. INTERPRETATION
    # ========================================================

    st.divider()

    st.subheader("💡 How to Interpret the Results")

    st.markdown(
        """
        1. **Colored points:** Each point represents a word.
        2. **Cluster colors:** Words assigned to the same K-Means
           cluster share a cluster label.
        3. **Star markers:** These indicate the average 2D position
           of the words in each cluster.
        4. **PCA:** Provides a linear two-dimensional projection.
        5. **t-SNE:** Emphasizes local neighborhood structure.
        6. **Cosine similarity:** Compares the direction of two
           original word vectors.

        **Important:** K-Means is applied to the original
        high-dimensional vectors, not the 2D visualization.
        A 2D projection can distort distances and relationships.
        """
    )

else:

    st.info(
        "👈 Choose a vocabulary source from the sidebar "
        "and provide words to generate the visualization."
    )


# ============================================================
# 20. FOOTER
# ============================================================

st.divider()

st.caption(
    "Embedding Space Visualizer | "
    "Python | GloVe | Word2Vec | PCA | t-SNE | "
    "K-Means | Cosine Similarity"
)
