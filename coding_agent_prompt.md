# Steam Game Discovery Explorer - Coding Agent Prompt

You are an expert AI coding assistant. Your task is to build a web application called the **Steam Game Discovery Explorer**. 

This application will help users discover Steam games through natural-language search, similar-game retrieval, and interactive exploration of the game space. It will be built in Python using Streamlit, relying on an initial Minimum Viable Product (MVP) core with text embeddings and a from-scratch cosine similarity retrieval algorithm.

Please follow the detailed implementation steps below meticulously.

---

## 1. Project Parameters
- **Core Technology**: Python, Streamlit, Pandas, NumPy, Scikit-learn, Sentence-Transformers (or equivalent).
- **Core Algorithm**: You MUST implement text embeddings and a **from-scratch** cosine similarity search algorithm (do not use `sklearn.NearestNeighbors` or vector databases for the primary grading algorithm).
- **Goal**: "Given a game idea or preference written in plain English, what Steam games best match it, and what other games are similar?"

---

## 2. Directory and File Structure
Set up the following core files in the working repository.

- `app.py`: Streamlit interface and page routing.
- `data.py`: Load, clean, and prepare the Steam dataset.
- `embed.py`: Build the text representation used for embeddings.
- `retrieval.py`: From-scratch cosine similarity and top-k retrieval.
- `viz.py`: Charts and interactive plots.
- `clustering.py`: Optional clustering logic and cluster labels.
- `utils.py`: Shared helper functions.
- `requirements.txt`: Project dependencies.
- `README.md`: Setup instructions and project overview.
- Directories: `data/raw/`, `data/processed/`, `notebooks/`, `assets/`

---

## 3. Implementation Phases

You must execute the following phases in strictly this order. Do not skip to Phase 5 without completing Phase 4. 

### Phase 1: Data Audit & Acquisition (`data.py`)
1. **Download Dataset**: Use the `datasets` library from Hugging Face to download the required dataset: `FronkonGames/steam-games-dataset`. Ensure your code handles loading this programmatically.
2. Provide functions to load the downloaded dataset into a pandas DataFrame.
3. Clean the data (e.g. handle missing values for descriptions, tags, and prices).
4. Identify and isolate the text fields to be searched: `name` (or `title`), `short_description`, `detailed_description`, `genres`, `tags`, `categories`.
5. Identify numerical/categorical fields for display/filtering: `price`, `release_date`, `metacritic_score`, `user_score`, `developers`, `publishers`.

### Phase 2: Embedding Generation (`embed.py`)
1. Create a function to combine textual data into one dense text string per game. For example: `title + short_description + selected tags + genres + categories`.
2. Generate fixed-length text embeddings for this combined text using a pre-trained embedding model (e.g., `sentence-transformers`).

### Phase 3: From-Scratch Retrieval Engine (`retrieval.py`)
**CRITICAL**: You must build the backend algorithm without relying on pre-packaged library implementations for the actual similarity search.

Create the following functions:
- `normalize_vectors(vectors)`: Ensure vectors have length 1.
- `cosine_similarity(vec_a, vec_b)`: Compute the dot product between two normalized vectors mathematically.
- `rank_games_for_query(query_string, negative_query_string)`: Embed the user's natural language query (what they want) and optionally map a negative query (what they don't want). Combine them (e.g. `Vector(query) - a * Vector(negative_query)`) and map against the dataset. Return the top-k matches with sorted similarity scores.
- `get_similar_games(game_id)`: Look up an existing game's vector and run a brute-force nearest neighbor search across the dataset based on cosine similarity.

### Phase 4: Minimum Viable Product (MVP) Interface (`app.py`)
Build the Streamlit interface with the core functionalities:
1. Title and Text Input box for freeform text search.
2. **"Dealbreakers" (Anti-Recommendation)**: Add an optional text input field for "What you absolutely DON'T want" (e.g., "microtransactions", "turn-based"). Feed this into `rank_games_for_query` as the `negative_query_string` to dynamically push results away from these concepts in the vector space.
3. **Algorithm Toggle Switch**: Provide a UI option (e.g., radio button) allowing the user to switch between the "From-Scratch Algorithm" and a "Scikit-Learn (`sklearn.NearestNeighbors`) Baseline". 
4. **Performance Metrics**: Display the execution time (latency) for the retrieval so users can naturally compare the speed difference between the custom python implementation and the optimized C-backend library.
5. Ranked results list rendering matches.
6. Detail view for a selected game.
7. "Similar Games" component populating from `get_similar_games()`.
8. **"Surprise Me" (Cross-Genre Explorer)**: Add an interface allowing users to select two heavily contrasting genres/concepts. Calculate the midpoint vector between these two clusters and return games residing in that bizarre middle ground.

### Phase 5: Visual Exploration (`viz.py` & `app.py`)
*Only begin this phase after the MVP is stable.*
1. Add a 2D map projection of games (use PCA, t-SNE, or an autoencoder).
2. Create scatter plots.
3. Enable point coloring based on Genre or Cluster.
4. Build standard EDA charts (price distribution, common genres).

### Phase 6: Extended Polishing (`clustering.py` & `app.py`)
1. Build optional k-means clustering logic to segment games.
2. Add interactive Streamlit filter widgets so users can narrow results by: Price range, Release year, and precise Genres.

---

## 4. Code Quality & Formatting
1. Ensure all code is extremely readable, cleanly structured, and formatted consistently.
2. Provide docstrings for all non-obvious logic and mathematical computations in `retrieval.py`.
3. Avoid redundant logic (use `utils.py` wherever helpful).
4. Update `README.md` periodically with execution commands (`streamlit run app.py`) and setup configuration.

When you are ready, please acknowledge these instructions and start generating the initial project structure and `README.md`.
