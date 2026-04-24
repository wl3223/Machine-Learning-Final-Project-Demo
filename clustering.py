import numpy as np
import streamlit as st
from sklearn.metrics import silhouette_score


def _kmeans_plus_plus_init(data, n_clusters, rng):
    """
    Initialize centroids using the k-means++ strategy.
    """
    n_samples = data.shape[0]
    centers = np.empty((n_clusters, data.shape[1]), dtype=data.dtype)

    first_idx = int(rng.integers(0, n_samples))
    centers[0] = data[first_idx]

    closest_dist_sq = np.sum((data - centers[0]) ** 2, axis=1)

    for center_idx in range(1, n_clusters):
        total_dist = closest_dist_sq.sum()
        if total_dist <= 0:
            random_idx = int(rng.integers(0, n_samples))
            centers[center_idx] = data[random_idx]
            continue

        probs = closest_dist_sq / total_dist
        probs = probs / probs.sum() # Ensure probabilities sum to strictly 1.0
        next_idx = int(rng.choice(n_samples, p=probs))
        centers[center_idx] = data[next_idx]

        new_dist_sq = np.sum((data - centers[center_idx]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centers


def _assign_labels(data, centers):
    """
    Assign each sample to the nearest center using squared Euclidean distance.
    """
    data_sq = np.sum(data ** 2, axis=1, keepdims=True)
    centers_sq = np.sum(centers ** 2, axis=1)
    distances = data_sq + centers_sq - 2 * (data @ centers.T)
    return np.argmin(distances, axis=1)


def _recompute_centers(data, labels, centers, rng):
    """
    Recompute cluster centers from assigned labels.
    Reinitialize empty clusters to random data points.
    """
    n_clusters = centers.shape[0]
    new_centers = np.empty_like(centers)

    for cluster_idx in range(n_clusters):
        members = data[labels == cluster_idx]
        if len(members) == 0:
            random_idx = int(rng.integers(0, data.shape[0]))
            new_centers[cluster_idx] = data[random_idx]
        else:
            new_centers[cluster_idx] = members.mean(axis=0)

    return new_centers


def _canonicalize_cluster_labels(labels, centers):
    """
    Remap arbitrary K-means labels into a stable ordering for UI display.
    """
    order = np.argsort(centers[:, 0])
    mapping = np.empty_like(order)
    mapping[order] = np.arange(len(order))
    return mapping[labels]


@st.cache_data(show_spinner=False)
def perform_kmeans_clustering(_dataset_vectors, n_clusters=8, seed=42, max_iter=100, tol=1e-4):
    """
    Groups vectors into semantic clusters using a from-scratch K-means implementation.
    """
    data = np.asarray(_dataset_vectors, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError("_dataset_vectors must be a 2D array")
    if data.shape[0] == 0:
        raise ValueError("_dataset_vectors must contain at least one row")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0")

    n_clusters = min(n_clusters, data.shape[0])
    rng = np.random.default_rng(seed)

    centers = _kmeans_plus_plus_init(data, n_clusters, rng)
    labels = np.zeros(data.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        labels = _assign_labels(data, centers)
        new_centers = _recompute_centers(data, labels, centers, rng)

        center_shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        if center_shift <= tol:
            break

    labels = _assign_labels(data, centers)
    return _canonicalize_cluster_labels(labels, centers)

def compute_clustering_metrics(data, labels, n_samples=3000):
    """
    Computes clustering evaluation metrics.
    Inertia is calculated from scratch.
    Silhouette Score uses sklearn (with sampling to ensure fast front-end rendering).
    """
    data = np.asarray(data, dtype=np.float32)
    
    # 1. From-scratch Inertia (Within-cluster sum of squares)
    unique_labels = np.unique(labels)
    centers = np.array([data[labels == l].mean(axis=0) for l in unique_labels])
    
    labels_mapped = np.searchsorted(unique_labels, labels)
    
    # Same vectorization technique used for distance calculation
    data_sq = np.sum(data ** 2, axis=1, keepdims=True)
    centers_sq = np.sum(centers ** 2, axis=1)
    distances = data_sq + centers_sq - 2 * (data @ centers.T)
    
    # Get the minimum squared distance for each point to its assigned cluster
    min_distances_sq = distances[np.arange(len(data)), labels_mapped]
    inertia = np.sum(min_distances_sq)
    
    # 2. Silhouette Score
    if len(unique_labels) > 1:
        # Sample points if dataset is too large to maintain fast UI performance
        sample_size = min(len(data), n_samples)
        sil_score = silhouette_score(data, labels, sample_size=sample_size, random_state=42)
    else:
        sil_score = -1.0 # Invalid configuration
        
    return inertia, sil_score

@st.cache_data(show_spinner=False)
def find_optimal_k(_dataset_vectors, k_min=2, k_max=15, seed=42):
    """
    Runs K-Means for a range of k values to compute Inertia and Silhouette scores.
    Returns a DataFrame with the results.
    """
    import pandas as pd
    results = []
    # To keep it fast for UI, if dataset is huge we could subsample, but our k-means is fast enough
    for k in range(k_min, k_max + 1):
        labels = perform_kmeans_clustering(_dataset_vectors, n_clusters=k, seed=seed, max_iter=50)
        inertia, sil_score = compute_clustering_metrics(_dataset_vectors, labels)
        results.append({
            'k': k,
            'inertia': inertia,
            'silhouette': sil_score
        })
    return pd.DataFrame(results)


def get_cluster_profiles(df, cluster_col='Cluster', n_top=3):
    """
    Extracts the top n genres, categories, and numeric stats for each cluster 
    to create a rich mathematical profile.
    Returns a dictionary mapping cluster names to a formatted string.
    """
    profiles = {}
    
    # Precompute global tag counts to calculate inverse frequency (TF-IDF style)
    df['combined_terms'] = df['genres'] + "," + df['tags']
    all_terms = df['combined_terms'].str.split(',').explode().str.strip()
    all_terms = all_terms[(all_terms != '') & (all_terms.str.lower() != 'nan')]
    global_counts = all_terms.value_counts()

    for cluster_name in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster_name]
        
        # 1. Top Tags/Genres using Mathematical Term Frequency Importance
        c_terms = cluster_df['combined_terms'].str.split(',').explode().str.strip()
        c_terms = c_terms[(c_terms != '') & (c_terms.str.lower() != 'nan')]
        c_counts = c_terms.value_counts()
        
        # Calculate score: Frequency in Cluster / sqrt(Global Frequency)
        # This naturally suppresses generic tags (Indie, Strategy) and boosts unique cluster signatures
        scores = {}
        for tag, count in c_counts.items():
            if count >= 2: # Min threshold to avoid ultra-rare outliers
                scores[tag] = count / (global_counts[tag] ** 0.6) # 0.6 smoothing factor
                
        top_user_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_top]
        top_genres = [k for k, v in top_user_tags]
        if not top_genres:
             top_genres = ['General / Mixed']
        
        # 2. Top Categories (Filter generic noise like Single-player/Family Sharing)
        cats = cluster_df['categories'].str.split(',').explode().str.strip()
        cats = cats[cats != '']
        meaningful_cats = cats[~cats.isin({'Single-player', 'Family Sharing', 'Steam Achievements', 'Steam Cloud', 'Profile Features Limited'})]
        if len(meaningful_cats.unique()) >= 2:
            top_cats = meaningful_cats.value_counts().nlargest(2).index.tolist()
        else:
            top_cats = cats.value_counts().nlargest(2).index.tolist()
        
        # 3. Numeric averages
        avg_price = cluster_df['price'].mean()
        
        # Format richer string
        genres_str = ", ".join(top_genres)
        cats_str = ", ".join(top_cats)
        
        profiles[cluster_name] = f"🎮 {genres_str}  🏷️ {cats_str}  💰 ${avg_price:.2f}"
        
    # Clean up temporary column
    df.drop(columns=['combined_terms'], inplace=True, errors='ignore')
    return profiles
