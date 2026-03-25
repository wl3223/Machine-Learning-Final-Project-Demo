# Group Member Progress Log
## Kylie Lin
1. Implemented deterministic reproducibility controls across the ML pipeline using a shared seed (42) and startup seed initialization.
2. Added set_reproducibility(seed=42) in utils.py to seed Python random, NumPy, and Torch (including CUDA when available).
3. Updated PCA flow in viz.py to use deterministic solver settings and fixed PCA sign indeterminacy so 2D map orientation is stable across runs.
Updated K-means in clustering.py to deterministic settings (random_state, fixed n_init, lloyd) and added canonical label remapping so cluster IDs stay stable.
4. Wired reproducibility controls into app startup in app.py via RANDOM_SEED = 42 and passed the seed into clustering.
5. Provided a Git Bash validation function check_reproducibility; you ran it successfully (Exit Code: 0), confirming deterministic outputs (labels_equal True, pca_equal True).