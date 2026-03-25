from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st


def _canonicalize_cluster_labels(labels, centers):
    """
    Remap arbitrary KMeans label IDs to a stable order based on centroid location.
    """
    ordered_old_ids = centers[:, 0].argsort()
    label_map = {old_id: new_id for new_id, old_id in enumerate(ordered_old_ids)}
    return pd.Series(labels).map(label_map).to_numpy()

@st.cache_data(show_spinner=False)
def perform_kmeans_clustering(_dataset_vectors, n_clusters=8, seed=42):
    """
    Groups games into semantic clusters using KMeans over the embedding vectors.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        algorithm='lloyd'
    )
    cluster_labels = kmeans.fit_predict(_dataset_vectors)
    return _canonicalize_cluster_labels(cluster_labels, kmeans.cluster_centers_)

