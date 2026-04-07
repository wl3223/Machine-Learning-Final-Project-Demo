import numpy as np
import streamlit as st


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

