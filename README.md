# A Semantic Vector-Based Approach to Steam Game Discovery

A Streamlit web application to discover Steam games through natural text search, dealbreakers, and vector similarity mapping. This application utilizes a from-scratch cosine similarity retrieval algorithm to match plain-English queries with games embedded using Sentence-Transformers.

## Features
1. **Natural Language Search**: Describe what you want to play ("A relaxing farming simulator with multiplayer").
2. **Dealbreakers**: Specify elements you do not want ("no microtransactions", "not turn-based").
3. **Algorithm Comparison**: Test our manual implementation of cosine similarity against the standard `sklearn` performance.
4. **Similar Games**: Find games semantically close to your favorites.
5. **Cross-Genre "Surprise Me"**: Combine wildly different genres to find games living in the midpoint vector space.
6. **Visual Map**: Interactive scatterplot visualizing games grouped by genre and mathematical similarity.

## Directory Structure
- `app.py`: Streamlit entry point.
- `data.py`: Dataset loading and preprocessing (from HuggingFace).
- `embed.py`: Embedding generation and combination logic.
- `retrieval.py`: From-scratch math for vector normalization and similarities.
- `viz.py`: Data visualizations and charts.
- `clustering.py`: K-means logic, segmentations, and Streamlit filters.
- `utils.py`: Helpers for caching, layout, and timing metrics.
- `data/`: Storage for raw dataset artifacts and mapped vectors.

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
   .\.venv\Scripts\Activate.ps1
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

   If `uv pip` still fails due to OneDrive hardlink restrictions, install with the venv python directly:
   ```powershell
   .\.venv\Scripts\python.exe -m ensurepip --upgrade
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Run the Streamlit application using the virtual environment's python directly or while activated:
   ```bash
   ./.venv/bin/python -m streamlit run app.py
   ```

   On Windows:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

Note: On the first launch, the application will automatically download the dataset `FronkonGames/steam-games-dataset` from Hugging Face and generate vector embeddings. Please allow a few minutes for this initialization process.
