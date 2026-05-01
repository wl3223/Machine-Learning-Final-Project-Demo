# Steam Game Discovery: A Hybrid NLP and Vector Search Approach

**Contributors:**
Weiyi Lu, Kylie Lin, Zipei He

A Streamlit web application to discover Steam games through natural text search, dealbreakers, and advanced three-stage vector mapping. This application utilizes an industry-grade **Hybrid Retrieval Architecture**. It merges dense semantic vectors (`Sentence-Transformers`) with sparse keyword matrices (`TF-IDF`), unifies them mathematically using **Reciprocal Rank Fusion (RRF)**, and achieves optimal human-like precision by reranking candidates through a deep learning **Cross-Encoder**. It also includes robust UI filtering, global cache pre-loading, and zero-latency quantitative evaluation workflows.

## Data Source & Acknowledgements
This project builds its semantic corpus upon the public **`FronkonGames/steam-games-dataset`** hosted natively on Hugging Face. The application automatically downloads, filters, and curates a robust catalog of over 10,000+ top-rated indie and mainstream games from this repository to power the discovery engine.

## Core Features
1. **Natural Language Search**: Describe what you want to play ("A relaxing farming simulator with multiplayer").
2. **Ultimate Hybrid Retrieval (Three-Stage Pipeline)**: Combines pure 'vibe' text embeddings (Bi-Encoders) with precise keyword tracking (TF-IDF Matrices) via RRF math. The top 60 broad candidates are then definitively sorted by a sequence-classifying Cross-Encoder.
3. **Dealbreakers (Negative Vectors)**: Specify elements you do not want ("no microtransactions"). The engine maps these to geometry and subtracts them from your search query.
4. **Custom K-Means++ Clustering Engine**: Segmented the game library grouped natively via NumPy to build mathematical Discovery Maps without `sklearn` black-box wrappers.
5. **Similar Game Mapping**: Bypass rerankers to analyze pure Geometric Cosine Distances and find titles perfectly sharing multidimensional mechanics.
6. **Cross-Genre "Surprise Me"**: Combine wildly different concepts (e.g. "cute animal crossing" + "violent shooter") to triangulate specific midpoint vector games.
7. **Pre-Loaded Zero-Latency Evaluation**: Quantitative validation metrics (MRR, Recall@k, WCSS, Silhouette) are rigorously computed on full neural-networks precisely during initialization, exposing instantaneous testing scores without lagging the frontend.

## Directory Structure
- `app.py`: Streamlit entry point.
- `data.py`: Dataset loading and preprocessing (from HuggingFace).
- `embed.py`: Embedding generation and combination logic.
- `retrieval.py`: From-scratch math for vector normalization and similarities.
- `viz.py`: Data visualizations and charts.
- `clustering.py`: K-means logic, segmentations, and Streamlit filters.
- `utils.py`: Helpers for caching, layout, and timing metrics.
- `assets/`: Static assets.
- `notebooks/`: Exploratory analysis notebooks.

## Installation Pipeline

1. Clone or download this repository.
2. Setup your virtual environment using `uv` and install the necessary dependencies:
	```bash
	uv venv
	source .venv/bin/activate
	uv pip install -r requirements.txt
	```
	On Windows (PowerShell):
	```powershell
	uv venv
	./.venv/Scripts/Activate.ps1
	$env:UV_LINK_MODE = "copy"
	uv pip install -r requirements.txt
	```

	On Windows (Git Bash):
	```bash
	uv venv
	source .venv/Scripts/activate
	export UV_LINK_MODE=copy
	uv pip install -r requirements.txt
	```

	If `uv pip` fails (common in some OneDrive setups), install with the venv python directly:
	```powershell
	./.venv/Scripts/python.exe -m ensurepip --upgrade
	./.venv/Scripts/python.exe -m pip install -r requirements.txt
	```
3. Run the Streamlit application using the virtual environment's python directly or while activated:
	```bash
	./.venv/bin/python -m streamlit run app.py
	```

	On Windows/Git Bash (as Windows does not have `bin`):
	```powershell
	./.venv/Scripts/python.exe -m streamlit run app.py
	```

	Optional (if port 8501 is busy):
	```bash
	python -m streamlit run app.py --server.port 8502
	```

Note: On the first launch, the application will automatically download the dataset `FronkonGames/steam-games-dataset`, load embedding assets, and generate cached vectors. Please allow a few minutes for this initialization process.
