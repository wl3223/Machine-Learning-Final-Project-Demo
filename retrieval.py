import numpy as np
import re


QUERY_EXPANSIONS = {
    'relaxing': ['cozy', 'calm', 'laid-back', 'low-stress', 'casual'],
    'fun': ['enjoyable', 'engaging', 'entertaining', 'addictive'],
    'good': ['well-reviewed', 'popular', 'polished', 'solid'],
    'bad': ['frustrating', 'rough', 'unfair', 'clunky'],
    'story': ['narrative', 'plot-driven', 'dialogue-rich', 'cinematic'],
    'multiplayer': ['co-op', 'online', 'friends', 'social'],
    'single player': ['single-player', 'solo', 'story mode'],
    'co-op': ['multiplayer', 'team-based', 'friends'],
    'horror': ['scary', 'tense', 'survival', 'suspenseful'],
    'puzzle': ['logic', 'brain-teaser', 'thinking', 'problem-solving'],
    'strategy': ['tactical', 'management', 'planning', 'turn-based'],
    'action': ['fast-paced', 'combat', 'intense', 'adrenaline'],
    'rpg': ['role-playing', 'character progression', 'quests', 'builds'],
    'farming': ['cozy', 'life sim', 'crafting', 'management'],
    'casual': ['accessible', 'easy-going', 'simple', 'pick-up-and-play'],
    'open world': ['exploration', 'sandbox', 'adventure', 'free roam'],
}


def build_robust_query_vector(model, query_string):
    """
    Builds a query vector by blending the raw query with a few lightweight
    semantic expansions so vague input still retrieves sensible matches.
    """
    query = str(query_string).strip()
    if not query:
        return model.encode([query], convert_to_numpy=True)[0]

    lower_query = query.lower()
    variants = [query]

    # Generic prompt-style variants help when the input is short or underspecified.
    variants.extend([
        f"A video game about {query}",
        f"A game with {query}",
        f"A game that feels {query}",
    ])

    # Add a few targeted expansions when the query contains broad concepts.
    for term, synonyms in QUERY_EXPANSIONS.items():
        if term in lower_query:
            variants.append(f"{query}, {' '.join(synonyms)}")
            variants.append(f"{', '.join(synonyms)} game")

    # If the query is very short, broaden it a bit more.
    content_words = re.findall(r"\b[a-zA-Z]{3,}\b", lower_query)
    if len(content_words) <= 2:
        variants.extend([
            "popular well-reviewed video game",
            "engaging game with broad appeal",
        ])

    # Deduplicate while preserving order.
    seen = set()
    unique_variants = []
    for variant in variants:
        normalized = variant.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_variants.append(variant)

    query_vectors = model.encode(unique_variants, convert_to_numpy=True)
    weights = np.ones(len(unique_variants), dtype=np.float32)
    weights[0] = 2.0  # keep the user's exact words dominant
    q_vec = np.average(query_vectors, axis=0, weights=weights)

    return q_vec

def normalize_vectors(vectors):
    """
    Normalizes a 2D matrix of vectors (or a single 1D vector) to have length 1 (L2 norm). 
    Aims to eliminate bias towards longer descriptons
    Math: v_norm = v / ||v||
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0:
            return vectors
        return vectors / norm
    else:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return (vectors / norms).astype(np.float32)

def cosine_similarity(vec_a, vec_b):
    """
    Computes the cosine similarity between two vectors.
    Used specfically to calculate similarity between a user query and game vectors, or between two game vectors.
    Math: (A dot B) / (||A|| * ||B||)
    """
    a_norm = normalize_vectors(vec_a)
    b_norm = normalize_vectors(vec_b)
    # For normalized vectors, dot product = cosine similarity
    return np.dot(a_norm, b_norm.T)

def batch_cosine_similarity(query_vector, dataset_vectors):
    """
    Computes cosine similarity between a 1D query vector and a 2D matrix of dataset vectors.
    """
    q_norm = normalize_vectors(query_vector)
    d_norm = normalize_vectors(dataset_vectors)
    # Dot product of 2D dataset with 1D query -> 1D array of similarities
    return np.dot(d_norm, q_norm)

def rank_games_for_query(query_string, negative_query_string, model, dataset_vectors, df, top_k=10, alpha=0.5, cross_encoder=None, tfidf_vectorizer=None, tfidf_matrix=None, filtered_indices=None):
    """
    Ranks games based on a positive query, optionally shifting away from a negative query.
    Implements a robust Three-Stage NLP Pipeline (Dense + Sparse/RRF + Cross-Encoder).
    
    Args:
        query_string (str): Primary search intent.
        negative_query_string (str): Dealbreakers intent.
        model (SentenceTransformer): The bi-encoder text embedding model.
        dataset_vectors (np.ndarray): (N, D) matrix of pre-computed game embeddings.
        df (pd.DataFrame): Dataset dataframe for returning results.
        top_k (int): Number of top matches to return.
        alpha (float): Weight for subtracting the negative vector.
        cross_encoder (CrossEncoder): Optional cross-encoder model for precision reranking.
        tfidf_vectorizer (TfidfVectorizer): Optional Sparse Vectorizer for keyword matching.
        tfidf_matrix (sparse matrix): Optional Global Sparse dataset.
        filtered_indices (list): List of real dataset row indices currently active via sidebar.
        
    Returns:
        pd.DataFrame: Top matching games containing their data and scores.
    """
    # 1. Embed query with lightweight semantic expansion for vague input
    q_vec = build_robust_query_vector(model, query_string)
    
    # 2. Scale dealbreakers consistently with the custom alpha setting
    if negative_query_string:
        neg_vec = model.encode([negative_query_string], convert_to_numpy=True)[0]
        q_vec = q_vec - (alpha * neg_vec)
        
    # 3. Calculate Dense similarities (Semantic Vibe) matching entirely from scratch
    dense_similarities = batch_cosine_similarity(q_vec, dataset_vectors)
    
    # 4. Determine base candidate pool sizes
    fetch_k = top_k * 3 if cross_encoder is not None else top_k
    
    # 5. Hybrid Search & Reciprocal Rank Fusion (RRF) Logic (Stage 1)
    if tfidf_vectorizer is not None and tfidf_matrix is not None and filtered_indices is not None and len(filtered_indices) > 0:
        fetch_k = min(fetch_k, len(filtered_indices))
        
        # Calculate Dense Ranks
        dense_sim_filtered = dense_similarities[filtered_indices]
        dense_ranks = np.zeros_like(dense_sim_filtered, dtype=float)
        dense_ranks[np.argsort(dense_sim_filtered)[::-1]] = np.arange(len(dense_sim_filtered))

        # Calculate Sparse Scores & Ranks
        sparse_query_vec = tfidf_vectorizer.transform([str(query_string)])
        sparse_matrix_filtered = tfidf_matrix[filtered_indices]
        sparse_similarities = sparse_query_vec.dot(sparse_matrix_filtered.T).toarray()[0]
        
        sparse_ranks = np.zeros_like(sparse_similarities, dtype=float)
        sparse_ranks[np.argsort(sparse_similarities)[::-1]] = np.arange(len(sparse_similarities))

        # RRF Math: Fuse both independent semantic/keyword signals perfectly safely
        rrf_scores = (1.0 / (60.0 + dense_ranks)) + (1.0 / (60.0 + sparse_ranks))
        
        best_local_indices = np.argsort(rrf_scores)[::-1][:fetch_k]
        top_indices = [filtered_indices[i] for i in best_local_indices]
        
        results = df.iloc[top_indices].copy()
        results['similarity_score'] = rrf_scores[best_local_indices]
        
    else:
        # Fallback to pure Dense ranking if no sparse objects passed
        top_indices = np.argsort(dense_similarities)[::-1][:fetch_k]
        results = df.iloc[top_indices].copy()
        results['similarity_score'] = dense_similarities[top_indices]
    
    # 6. Apply Cross-Encoder Reranking if enabled (Stage 2)
    if cross_encoder is not None:
        pairs = []
        for _, row in results.iterrows():
            doc_text = f"{row.get('name', '')}: {row.get('short_description', '')}"
            pairs.append([str(query_string), str(doc_text)])
        
        # Cross_encoder outputs raw logits (can be 5.0, -10.0, etc.)
        raw_logits = cross_encoder.predict(pairs)
        
        # Apply Sigmoid function to normalize mathematically to continuous [0.0, 1.0] probability
        ce_scores = 1 / (1 + np.exp(-raw_logits))
        
        # Overwrite the similarity score so the UI naturally picks up the final stage precision
        results['cross_encoder_score'] = ce_scores
        results['similarity_score'] = ce_scores
        results = results.sort_values(by='cross_encoder_score', ascending=False).head(top_k)
    
    return results

def get_similar_games(target_idx, dataset_vectors, df, top_k=5):
    """
    Looks up an existing game's vector by dataframe index and runs a brute-force
    cosine similarity search against the entire dataset.
    """
    target_vec = dataset_vectors[target_idx]
    
    similarities = batch_cosine_similarity(target_vec, dataset_vectors)
    
    # Fetch top_k + 1 because the 1st result will be the game itself (score ~ 1.0)
    top_indices = np.argsort(similarities)[::-1][:top_k+1]
    
    # Filter out the exact target_idx
    top_indices = [idx for idx in top_indices if idx != target_idx][:top_k]
    
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    
    return results

def evaluate_retrieval_mrr(model, dataset_vectors, df, sample_size=50, top_k=5, query_field='detailed_description'):
    """
    Evaluates the retrieval model using Known-Item Search.
    Randomly samples games, uses a held-out text field as the query, and checks
    how well the model ranks the original game.

    Default behavior uses `detailed_description` so the validation does not reuse
    the same field that is already part of the embedding input.
    """

    if query_field not in df.columns:
        query_field = 'short_description'

    eval_df = df[df[query_field].astype(str).str.strip() != ''].copy()
    if len(eval_df) == 0:
        eval_df = df.copy()
        query_field = 'short_description' if 'short_description' in df.columns else df.columns[0]

    if len(eval_df) < sample_size:
        sample_size = len(eval_df)
        
    # Randomly sample games
    sample_df = eval_df.sample(n=sample_size, random_state=42)
    
    queries = sample_df[query_field].tolist()
    target_indices = sample_df.index.tolist()
    
    # Batch encode all queries for performance
    query_vectors = model.encode(queries, convert_to_numpy=True)
    
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    
    # For each query, see where the target game ranks
    for i, target_idx in enumerate(target_indices):
        q_vec = query_vectors[i]
        
        # calculate similarities
        similarities = batch_cosine_similarity(q_vec, dataset_vectors)
        
        # rank all items
        ranked_dataset_indices = np.argsort(similarities)[::-1]
        
        # Find the rank of our target game
        # Note: df.index maps directly to dataset_vectors index in our setup
        df_index_pos = df.index.get_loc(target_idx)
        
        # np.where returns a tuple of arrays, [0][0] gets the integer index
        rank = np.where(ranked_dataset_indices == df_index_pos)[0][0] + 1 
        
        reciprocal_ranks.append(1.0 / rank)
        if rank == 1:
            hits_at_1 += 1
        if rank <= top_k:
            hits_at_5 += 1
            
    mrr = np.mean(reciprocal_ranks)
    recall_at_1 = hits_at_1 / sample_size
    recall_at_5 = hits_at_5 / sample_size
    
    return {
        "mrr": mrr,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "sample_size": sample_size
    }

def evaluate_ultimate_pipeline(model, dataset_vectors, df, cross_encoder, tfidf_vectorizer, tfidf_matrix, sample_size=30, top_k=5, query_field='detailed_description'):
    """
    Evaluates the full Three-Stage Hybrid Pipeline ensuring testing fidelity.
    Passes queries exactly through the final rank_games_for_query logic.
    """
    if query_field not in df.columns:
        query_field = 'short_description'

    eval_df = df[df[query_field].astype(str).str.strip() != ''].copy()
    if len(eval_df) < sample_size:
        sample_size = len(eval_df)
        
    sample_df = eval_df.sample(n=sample_size, random_state=42)
    target_indices = sample_df.index.tolist()
    
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    all_indices = df.index.tolist()
    
    for target_idx in target_indices:
        query_str = eval_df.loc[target_idx, query_field]
        eval_fetch_depth = 50 
        
        # Heavy computation: Execute the entire dense + sparse + cross pipeline 
        results = rank_games_for_query(
            query_string=query_str, 
            negative_query_string="", 
            model=model, 
            dataset_vectors=dataset_vectors, 
            df=df, 
            top_k=eval_fetch_depth, 
            alpha=0.0, 
            cross_encoder=cross_encoder, 
            tfidf_vectorizer=tfidf_vectorizer, 
            tfidf_matrix=tfidf_matrix, 
            filtered_indices=all_indices
        )
        
        df_index_pos = results.index.tolist()
        
        if target_idx in df_index_pos:
            rank = df_index_pos.index(target_idx) + 1
        else:
            rank = eval_fetch_depth + 1
            
        reciprocal_ranks.append(1.0 / rank)
        if rank == 1:
            hits_at_1 += 1
        if rank <= top_k:
            hits_at_5 += 1
            
    return {
        "mrr": np.mean(reciprocal_ranks),
        "recall_at_1": hits_at_1 / sample_size,
        "recall_at_5": hits_at_5 / sample_size,
        "sample_size": sample_size
    }
