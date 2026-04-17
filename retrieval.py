import numpy as np

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

def rank_games_for_query(query_string, negative_query_string, model, dataset_vectors, df, top_k=10, alpha=0.5):
    """
    Ranks games based on a positive query, optionally shifting away from a negative query.
    
    Args:
        query_string (str): Primary search intent.
        negative_query_string (str): Dealbreakers intent.
        model (SentenceTransformer): The text embedding model.
        dataset_vectors (np.ndarray): (N, D) matrix of pre-computed game embeddings.
        df (pd.DataFrame): Dataset dataframe for returning results.
        top_k (int): Number of top matches to return.
        alpha (float): Weight for subtracting the negative vector.
        
    Returns:
        pd.DataFrame: Top matching games containing their data and 'similarity_score'.
    """
    # 1. Embed positive query
    q_vec = model.encode([query_string], convert_to_numpy=True)[0]
    
    # 2. Vector Math for dealbreakers: Q_combined = Q_pos - alpha * Q_neg
    if negative_query_string:
        neg_vec = model.encode([negative_query_string], convert_to_numpy=True)[0]
        q_vec = q_vec - (alpha * neg_vec)
        
    # 3. Calculate similarities matching entirely from scratch
    similarities = batch_cosine_similarity(q_vec, dataset_vectors)
    
    # 4. Sort distances and return top_k index list
    # Unpack np.argsort indexing descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 5. Prep output dataframe
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    
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
