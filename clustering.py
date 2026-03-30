from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def perform_kmeans_clustering(_dataset_vectors, n_clusters=8):
    """
    Groups games into semantic clusters using KMeans over the embedding vectors.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(_dataset_vectors)
    return cluster_labels

